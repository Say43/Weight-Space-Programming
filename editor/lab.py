"""Experiment bookkeeping: checks the budget and records each run in logs/experiments.jsonl."""
import json
from datetime import datetime, timezone

from eval import config as cfg

BUDGET = {"editor": 40, "baseline": 10}
MAX_ITEMS = 512


def _rows():
    if not cfg.EXPERIMENT_LOG.exists():
        return []
    return [json.loads(l) for l in cfg.EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines() if l]


def _append(row):
    cfg.LOG_DIR.mkdir(exist_ok=True)
    with open(cfg.EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def used(track):
    return len({r["number"] for r in _rows() if r["track"] == track})


def start(track, hypothesis, code_path, n_items):
    """Books the experiment before the run, so that crashed runs also count against the budget."""
    if track in BUDGET:
        if used(track) >= BUDGET[track]:
            raise RuntimeError(f"Budget for {track} is exhausted")
        if n_items > MAX_ITEMS:
            raise RuntimeError(f"{n_items} items exceed the limit of {MAX_ITEMS}")
    number = used(track) + 1
    _append({"track": track, "number": number, "phase": "start", "hypothesis": hypothesis,
             "code_path": code_path, "n_items": n_items,
             "timestamp": datetime.now(timezone.utc).isoformat()})
    return number


def finish(track, number, result):
    _append({"track": track, "number": number, "phase": "finish", "result": result,
             "timestamp": datetime.now(timezone.utc).isoformat()})
