"""
Tracker de uso de API — registra cada llamada a Gemini con tokens y coste.
Persiste en un JSON local (usage_log.json).
"""

from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from threading import Lock

USAGE_FILE = Path(__file__).resolve().parent / "usage_log.json"

# Gemini 2.5 Flash pricing (USD per million tokens)
PRICING = {
    "input": 0.15 / 1_000_000,
    "output": 0.60 / 1_000_000,
    "thinking": 0.35 / 1_000_000,
}

_lock = Lock()


def _load() -> list[dict]:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save(entries: list[dict]):
    USAGE_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )


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
    cost_thinking = thinking_tokens * PRICING["thinking"]
    cost_total = cost_input + cost_output + cost_thinking

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
        entries = _load()
        entries.append(entry)
        _save(entries)

    return entry


def get_daily_summary() -> list[dict]:
    """Devuelve resumen diario de uso."""
    entries = _load()

    by_day: dict[str, dict] = {}
    for e in entries:
        d = e["date"]
        if d not in by_day:
            by_day[d] = {
                "date": d,
                "requests": 0,
                "facturas": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        s = by_day[d]
        s["requests"] += 1
        s["facturas"] += e.get("num_facturas", 0)
        s["prompt_tokens"] += e.get("prompt_tokens", 0)
        s["output_tokens"] += e.get("output_tokens", 0)
        s["thinking_tokens"] += e.get("thinking_tokens", 0)
        s["total_tokens"] += e.get("total_tokens", 0)
        s["cost_usd"] += e.get("cost_usd", 0.0)

    # Sort by date descending
    days = sorted(by_day.values(), key=lambda x: x["date"], reverse=True)
    for d in days:
        d["cost_usd"] = round(d["cost_usd"], 4)
    return days


def get_all_entries() -> list[dict]:
    """Devuelve todas las entradas (más recientes primero)."""
    entries = _load()
    entries.reverse()
    return entries


def get_totals() -> dict:
    """Devuelve totales globales."""
    entries = _load()
    total = {
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
    return total
