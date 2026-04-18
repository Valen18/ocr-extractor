"""
Matcher de referencia externa — busca el localizador del cliente en un Excel
de reservas propio, usando fuzzy matching sobre el nombre del cliente.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

BASE_DIR = Path(__file__).resolve().parent
REF_FILE = BASE_DIR / "reference_data.json"
REF_EXCEL = BASE_DIR / "reference_current.xlsx"

DB_URL = os.environ.get("DATABASE_URL", "")
_lock = Lock()
_table_ready = False


# ── Name normalization ─────────────────────────────────────────

def format_client_name(name: str) -> str:
    """
    Convierte 'Apellido, Nombre' → 'Nombre Apellidos' (sin coma).
    Si no hay coma, devuelve el nombre tal cual.
    """
    if not name:
        return ""
    name = str(name).strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[1]} {parts[0]}"
    return name


def normalize_name(name: str) -> str:
    """Normaliza para matching: sin acentos, lowercase, palabras ordenadas."""
    if not name:
        return ""
    # Apply comma swap first
    n = format_client_name(name)
    # Remove accents, lowercase
    n = unidecode(n).lower().strip()
    # Remove common prefixes/titles
    for prefix in ("sr ", "sra ", "don ", "dna ", "d. ", "dr ", "dra "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    # Sort words (so "Juan Perez" matches "Perez Juan")
    words = sorted(n.split())
    return " ".join(words)


# ── DB helpers ─────────────────────────────────────────────────

def _get_db():
    import psycopg2
    return psycopg2.connect(DB_URL)


def _ensure_table():
    global _table_ready
    if _table_ready or not DB_URL:
        return
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reference_bookings (
                    id SERIAL PRIMARY KEY,
                    client_name TEXT NOT NULL,
                    client_name_norm TEXT NOT NULL,
                    localizador TEXT NOT NULL,
                    extra JSONB DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ref_bookings_norm
                ON reference_bookings(client_name_norm)
            """)
            conn.commit()
        _table_ready = True
    finally:
        conn.close()


# ── Storage ─────────────────────────────────────────────────────

def save_reference(records: list[dict]) -> int:
    """
    Guarda la referencia. records es una lista de dicts con keys:
    - client_name
    - localizador
    - extra (dict con datos adicionales)

    Returns: número de registros guardados
    """
    with _lock:
        if DB_URL:
            try:
                _ensure_table()
                conn = _get_db()
                try:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM reference_bookings")
                        for r in records:
                            cur.execute("""
                                INSERT INTO reference_bookings
                                    (client_name, client_name_norm, localizador, extra)
                                VALUES (%s, %s, %s, %s)
                            """, (
                                r["client_name"],
                                normalize_name(r["client_name"]),
                                r["localizador"],
                                json.dumps(r.get("extra", {})),
                            ))
                        conn.commit()
                finally:
                    conn.close()
                return len(records)
            except Exception as e:
                print(f"DB save failed, falling back to JSON: {e}")

        # Fallback: JSON file
        data = [{
            "client_name": r["client_name"],
            "client_name_norm": normalize_name(r["client_name"]),
            "localizador": r["localizador"],
            "extra": r.get("extra", {}),
        } for r in records]
        REF_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=None), encoding="utf-8")
        return len(data)


def load_reference() -> list[dict]:
    """Carga todos los registros de referencia."""
    if DB_URL:
        try:
            _ensure_table()
            conn = _get_db()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT client_name, client_name_norm, localizador, extra
                        FROM reference_bookings
                    """)
                    rows = cur.fetchall()
            finally:
                conn.close()
            return [{
                "client_name": r[0],
                "client_name_norm": r[1],
                "localizador": r[2],
                "extra": r[3] if isinstance(r[3], dict) else (json.loads(r[3]) if r[3] else {}),
            } for r in rows]
        except Exception:
            pass

    if REF_FILE.exists():
        try:
            return json.loads(REF_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def get_reference_status() -> dict:
    """Devuelve el estado de la referencia cargada."""
    records = load_reference()
    return {
        "loaded": len(records) > 0,
        "count": len(records),
    }


def clear_reference() -> bool:
    """Borra toda la referencia."""
    with _lock:
        if DB_URL:
            try:
                _ensure_table()
                conn = _get_db()
                try:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM reference_bookings")
                        conn.commit()
                finally:
                    conn.close()
            except Exception:
                pass

        if REF_FILE.exists():
            REF_FILE.unlink()
        if REF_EXCEL.exists():
            REF_EXCEL.unlink()
    return True


# ── Parsing Excel ──────────────────────────────────────────────

def parse_excel_to_records(file_path: Path) -> list[dict]:
    """
    Parsea el Excel de referencia y extrae nombre + localizador.

    Estrategia:
    - Busca columna "Nombre completo" (case-insensitive) o columna J (índice 9)
    - Busca columna "Localizador" (case-insensitive) o columna D (índice 3)
    """
    df = pd.read_excel(str(file_path), engine="openpyxl")

    # Find columns by name (flexible matching)
    name_col = None
    loc_col = None

    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "nombre completo" in col_lower:
            name_col = col
        elif name_col is None and col_lower == "nombre completo":
            name_col = col
        if "localizador" in col_lower:
            loc_col = col

    # Fallback to column positions (D=3, J=9)
    if name_col is None and len(df.columns) > 9:
        name_col = df.columns[9]
    if loc_col is None and len(df.columns) > 3:
        loc_col = df.columns[3]

    if name_col is None or loc_col is None:
        raise ValueError(
            f"No se encontro columna 'Nombre completo' ni 'Localizador'. "
            f"Columnas disponibles: {list(df.columns)[:15]}"
        )

    records = []
    for _, row in df.iterrows():
        name = row.get(name_col)
        loc = row.get(loc_col)

        if pd.isna(name) or pd.isna(loc):
            continue

        name_str = str(name).strip()
        loc_str = str(loc).strip()

        # Filter out headers, empty, numeric-only nonsense
        if not name_str or not loc_str or name_str.lower() in ("nan", "none"):
            continue
        if loc_str.lower() in ("nan", "none"):
            continue

        # Clean localizador (it might be a float like 436911.0)
        try:
            loc_str = str(int(float(loc_str)))
        except (ValueError, TypeError):
            pass

        records.append({
            "client_name": name_str,
            "localizador": loc_str,
            "extra": {},
        })

    return records


# ── Matching ───────────────────────────────────────────────────

def find_localizador(client_name: str, reference: list[dict], threshold: int = 85) -> tuple[str, int]:
    """
    Busca el localizador para un nombre de cliente.

    Returns: (localizador, score). score es 0-100, 100 = exacto.
    Si no encuentra nada sobre el threshold, devuelve ("", 0).
    """
    if not client_name or not reference:
        return "", 0

    # Handle multiple passengers separated by |
    # Take the first one (titular/main passenger)
    first_name = client_name.split("|")[0].strip()
    norm_query = normalize_name(first_name)

    if not norm_query:
        return "", 0

    # Pre-compute norms if missing
    for r in reference:
        if "client_name_norm" not in r or not r["client_name_norm"]:
            r["client_name_norm"] = normalize_name(r["client_name"])

    # Exact match first (by normalized name)
    for r in reference:
        if r["client_name_norm"] == norm_query:
            return r["localizador"], 100

    # Fuzzy match
    best_score = 0
    best_loc = ""
    for r in reference:
        score = fuzz.token_sort_ratio(norm_query, r["client_name_norm"])
        if score > best_score:
            best_score = score
            best_loc = r["localizador"]

    if best_score >= threshold:
        return best_loc, best_score
    return "", best_score


def enrich_facturas(facturas: list[dict]) -> list[dict]:
    """
    Añade el campo 'nuestro_localizador' a cada factura, buscando
    en la referencia cargada. Devuelve la lista enriquecida.
    """
    reference = load_reference()
    if not reference:
        # No reference loaded — just format names
        for fac in facturas:
            if fac.get("pasajeros"):
                # Apply name formatting to first passenger
                names = [format_client_name(n.strip()) for n in fac["pasajeros"].split("|")]
                fac["pasajeros"] = " | ".join(names)
        return facturas

    for fac in facturas:
        pasajeros = fac.get("pasajeros", "")
        # Reformat names
        if pasajeros:
            names = [format_client_name(n.strip()) for n in pasajeros.split("|")]
            fac["pasajeros"] = " | ".join(names)

        # Find localizador using first passenger
        loc, score = find_localizador(pasajeros, reference)
        fac["nuestro_localizador"] = loc
        fac["_match_score"] = score

    return facturas
