"""
Tracker de uso de API — registra cada llamada a Gemini con tokens y coste.
Persiste en PostgreSQL (Neon) si DATABASE_URL está configurada, si no en JSON local.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, date
from pathlib import Path
from threading import Lock

# Gemini 2.5 Flash pricing (USD per million tokens)
# Fuente: https://ai.google.dev/gemini-api/docs/pricing
# Output price ($2.50/M) already includes thinking tokens
PRICING = {
    "input": 0.30 / 1_000_000,     # $0.30 per 1M tokens (text/image/video)
    "output": 2.50 / 1_000_000,    # $2.50 per 1M tokens (includes thinking)
    "thinking": 0.0,               # included in output price
}

_lock = Lock()
_db_url = os.environ.get("DATABASE_URL", "")
_table_created = False


# ── PostgreSQL ──────────────────────────────────────────────────

def _get_conn():
    import psycopg2
    return psycopg2.connect(_db_url)


def _ensure_table():
    global _table_created
    if _table_created:
        return
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id SERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    usage_date DATE NOT NULL,
                    operation VARCHAR(50) NOT NULL,
                    filename VARCHAR(500) DEFAULT '',
                    prompt_tokens INT DEFAULT 0,
                    output_tokens INT DEFAULT 0,
                    thinking_tokens INT DEFAULT 0,
                    total_tokens INT DEFAULT 0,
                    num_facturas INT DEFAULT 0,
                    cost_usd NUMERIC(12,6) DEFAULT 0
                )
            """)
            conn.commit()
        _table_created = True
    finally:
        conn.close()


def _db_log(entry: dict):
    _ensure_table()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO api_usage
                    (ts, usage_date, operation, filename, prompt_tokens, output_tokens,
                     thinking_tokens, total_tokens, num_facturas, cost_usd)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                entry["ts"], entry["date"], entry["op"], entry["file"],
                entry["prompt_tokens"], entry["output_tokens"],
                entry["thinking_tokens"], entry["total_tokens"],
                entry["num_facturas"], entry["cost_usd"],
            ))
            conn.commit()
    finally:
        conn.close()


def _db_get_daily() -> list[dict]:
    _ensure_table()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT usage_date, COUNT(*) as requests,
                       COALESCE(SUM(num_facturas),0),
                       COALESCE(SUM(prompt_tokens),0),
                       COALESCE(SUM(output_tokens),0),
                       COALESCE(SUM(thinking_tokens),0),
                       COALESCE(SUM(total_tokens),0),
                       COALESCE(SUM(cost_usd),0)
                FROM api_usage
                GROUP BY usage_date
                ORDER BY usage_date DESC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    return [{
        "date": str(r[0]),
        "requests": r[1],
        "facturas": int(r[2]),
        "prompt_tokens": int(r[3]),
        "output_tokens": int(r[4]),
        "thinking_tokens": int(r[5]),
        "total_tokens": int(r[6]),
        "cost_usd": round(float(r[7]), 4),
    } for r in rows]


def _db_get_entries(limit: int = 200) -> list[dict]:
    _ensure_table()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts, usage_date, operation, filename, prompt_tokens,
                       output_tokens, thinking_tokens, total_tokens,
                       num_facturas, cost_usd
                FROM api_usage
                ORDER BY id DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [{
        "ts": r[0].isoformat() if r[0] else "",
        "date": str(r[1]),
        "op": r[2],
        "file": r[3] or "",
        "prompt_tokens": r[4] or 0,
        "output_tokens": r[5] or 0,
        "thinking_tokens": r[6] or 0,
        "total_tokens": r[7] or 0,
        "num_facturas": r[8] or 0,
        "cost_usd": float(r[9] or 0),
    } for r in rows]


def _db_get_totals() -> dict:
    _ensure_table()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*),
                       COALESCE(SUM(num_facturas),0),
                       COALESCE(SUM(prompt_tokens),0),
                       COALESCE(SUM(output_tokens),0),
                       COALESCE(SUM(thinking_tokens),0),
                       COALESCE(SUM(total_tokens),0),
                       COALESCE(SUM(cost_usd),0),
                       MIN(usage_date),
                       MAX(usage_date)
                FROM api_usage
            """)
            r = cur.fetchone()
    finally:
        conn.close()

    return {
        "requests": r[0] or 0,
        "facturas": int(r[1]),
        "prompt_tokens": int(r[2]),
        "output_tokens": int(r[3]),
        "thinking_tokens": int(r[4]),
        "total_tokens": int(r[5]),
        "cost_usd": round(float(r[6]), 4),
        "first_date": str(r[7]) if r[7] else None,
        "last_date": str(r[8]) if r[8] else None,
    }


# ── JSON fallback (local, sin DB) ──────────────────────────────

USAGE_FILE = Path(__file__).resolve().parent / "usage_log.json"


def _json_load() -> list[dict]:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _json_save(entries: list[dict]):
    USAGE_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )


# ── Public API ──────────────────────────────────────────────────

def _use_db() -> bool:
    return bool(_db_url)


def log_usage(
    operation: str,
    filename: str,
    prompt_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    total_tokens: int = 0,
    num_facturas: int = 0,
):
    """Registra una llamada a la API."""
    cost_input = prompt_tokens * PRICING["input"]
    cost_output = output_tokens * PRICING["output"]
    cost_total = cost_input + cost_output

    entry = {
        "ts": datetime.now().isoformat(),
        "date": date.today().isoformat(),
        "op": operation,
        "file": filename,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens or (prompt_tokens + output_tokens + thinking_tokens),
        "num_facturas": num_facturas,
        "cost_usd": round(cost_total, 6),
    }

    with _lock:
        if _use_db():
            try:
                _db_log(entry)
            except Exception:
                # Fallback to JSON if DB fails
                entries = _json_load()
                entries.append(entry)
                _json_save(entries)
        else:
            entries = _json_load()
            entries.append(entry)
            _json_save(entries)

    return entry


def get_daily_summary() -> list[dict]:
    if _use_db():
        try:
            return _db_get_daily()
        except Exception:
            pass

    entries = _json_load()
    by_day: dict[str, dict] = {}
    for e in entries:
        d = e["date"]
        if d not in by_day:
            by_day[d] = {
                "date": d, "requests": 0, "facturas": 0,
                "prompt_tokens": 0, "output_tokens": 0,
                "thinking_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
            }
        s = by_day[d]
        s["requests"] += 1
        s["facturas"] += e.get("num_facturas", 0)
        s["prompt_tokens"] += e.get("prompt_tokens", 0)
        s["output_tokens"] += e.get("output_tokens", 0)
        s["thinking_tokens"] += e.get("thinking_tokens", 0)
        s["total_tokens"] += e.get("total_tokens", 0)
        s["cost_usd"] += e.get("cost_usd", 0.0)

    days = sorted(by_day.values(), key=lambda x: x["date"], reverse=True)
    for d in days:
        d["cost_usd"] = round(d["cost_usd"], 4)
    return days


def get_all_entries() -> list[dict]:
    if _use_db():
        try:
            return _db_get_entries()
        except Exception:
            pass

    entries = _json_load()
    entries.reverse()
    return entries


def get_totals() -> dict:
    if _use_db():
        try:
            return _db_get_totals()
        except Exception:
            pass

    entries = _json_load()
    return {
        "requests": len(entries),
        "facturas": sum(e.get("num_facturas", 0) for e in entries),
        "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in entries),
        "output_tokens": sum(e.get("output_tokens", 0) for e in entries),
        "thinking_tokens": sum(e.get("thinking_tokens", 0) for e in entries),
        "total_tokens": sum(e.get("total_tokens", 0) for e in entries),
        "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in entries), 4),
        "first_date": entries[0]["date"] if entries else None,
        "last_date": entries[-1]["date"] if entries else None,
    }
