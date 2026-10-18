"""Structured JSON logs + a SQLite run store. The dashboard and the Optimization Agent read it."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
import threading

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"), format="%(message)s")
log = logging.getLogger("hiring_intel")

DB_PATH = os.getenv("HI_DB", "data/runs.db")
_lock = threading.Lock()
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
DB = sqlite3.connect(DB_PATH, check_same_thread=False)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, ts TEXT, scenario TEXT, size TEXT, variant TEXT,
    data_path TEXT, n_candidates INT, total_usd REAL, llm_calls INT, input_tokens INT, output_tokens INT,
    e2e_latency_ms INT, wall_ms INT, mean_eval REAL, hitl_pending INT, hitl_auto INT, status TEXT);
CREATE TABLE IF NOT EXISTS llm_calls(ts TEXT, run_id TEXT, agent TEXT, model TEXT, tier TEXT,
    input_tokens INT, output_tokens INT, usd REAL, latency_ms INT, cached INT, simulated INT);
CREATE TABLE IF NOT EXISTS decisions(ts TEXT, run_id TEXT, task TEXT, choice TEXT, confidence REAL,
    backend TEXT, latency_ms INT, escalated INT, n_items INT);
CREATE TABLE IF NOT EXISTS timings(run_id TEXT, stage TEXT, agent TEXT, compute_ms INT, llm_ms INT);
CREATE TABLE IF NOT EXISTS agent_outputs(run_id TEXT, agent TEXT, status TEXT, json TEXT);
CREATE TABLE IF NOT EXISTS routing(run_id TEXT, agent TEXT, tier TEXT, model TEXT, reason TEXT, decided_by TEXT);
CREATE TABLE IF NOT EXISTS eval_results(run_id TEXT, agent TEXT, overall REAL, json TEXT);
CREATE TABLE IF NOT EXISTS review_queue(item_id TEXT PRIMARY KEY, run_id TEXT, agent TEXT, status TEXT,
    reviewer TEXT, note TEXT, reasons TEXT, json TEXT, ts TEXT);
"""
with _lock:
    DB.executescript(SCHEMA)
    DB.commit()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def execute(sql: str, params=()) -> None:
    with _lock:
        DB.execute(sql, params)
        DB.commit()


def log_call(agent: str, run_id: str, cost) -> None:
    row = {"ts": _now(), "run_id": run_id, "agent": agent, "model": cost.model, "tier": cost.tier,
               "input_tokens": cost.input_tokens, "output_tokens": cost.output_tokens, "usd": cost.usd,
               "latency_ms": cost.latency_ms, "cached": int(cost.cached), "simulated": int(cost.simulated)}
    log.info(json.dumps({"event": "llm_call", **row}))
    execute("INSERT INTO llm_calls VALUES (:ts,:run_id,:agent,:model,:tier,:input_tokens,:output_tokens,"
            ":usd,:latency_ms,:cached,:simulated)", row)


def log_decision(run_id: str, d, n_items: int = 1) -> None:
    log.info(json.dumps({"event": "decision", "run_id": run_id, **d.model_dump(exclude={"probs"})}))
    execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
            (_now(), run_id, d.task, d.choice, d.confidence, d.backend, d.latency_ms, int(d.escalated), n_items))


def log_timing(run_id: str, stage: str, agent: str, compute_ms: int, llm_ms: int) -> None:
    execute("INSERT INTO timings VALUES (?,?,?,?,?)", (run_id, stage, agent, int(compute_ms), int(llm_ms)))


def query(sql: str, params=()):
    import pandas as pd
    with _lock:
        return pd.read_sql(sql, DB, params=params)