"""The only place allowed to read the test split. Prints aggregate metrics only.

Each condition runs exactly once; a repeat needs --force and a justification in the notebook.
"""
import argparse
import hashlib
import json
from pathlib import Path

from edits.apply_edit import apply_edit, edit_stats
from eval import config as cfg
from eval.guards import mmlu_accuracy, open_generations, perplexity
from eval.metrics import condition_metrics
from eval.model import load_model, load_tokenizer
from eval.scoring import score_items

META_PATH = cfg.LOG_DIR / "splits_meta.json"


def load_test():
    raw = cfg.TEST_PATH.read_bytes()
    expected = json.loads(META_PATH.read_text(encoding="utf-8"))["test_sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit("test.jsonl differs from the recorded hash.")
    return [json.loads(l) for l in raw.decode("utf-8").splitlines()]


def evaluate(name, edit=None, system=None, robust=False):
    tok = load_tokenizer()
    model = load_model("cpu")
    total = sum(p.numel() for p in model.parameters())
    stats = {"matrices": 0, "changed_params": 0, "changed_share": 0.0, "max_delta_rank": 0}
    if edit:
        apply_edit(model, edit)
        stats = edit_stats(edit, total)
    model = model.to("cuda")
    items = load_test()
    conditions = ["C0", "C1", "C2", "C3"] + (["C1b"] if robust else [])
    predictions = {c: score_items(model, tok, items, c, system) for c in conditions}
    result = {"name": name, "system": system, "edit": str(edit) if edit else None, "edit_stats": stats,
              **condition_metrics(items, predictions),
              "mmlu": mmlu_accuracy(model, tok, system), "perplexity": perplexity(model, tok)}
    summary = dict(result)
    result["generations"] = open_generations(model, tok, system)
    return result, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="unmodified model")
    parser.add_argument("--name")
    parser.add_argument("--edit", type=Path)
    parser.add_argument("--system")
    parser.add_argument("--robust", action="store_true", help="also measure the hint C1b")
    parser.add_argument("--force", action="store_true", help="only with a justification in the notebook")
    args = parser.parse_args()
    name = "baseline" if args.baseline else args.name
    if not name:
        parser.error("pass --baseline or --name")
    path = cfg.RESULT_DIR / f"{name}.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path.name} already exists. Repeat only with --force and a justification in the notebook.")
    result, summary = evaluate(name, args.edit, args.system, args.robust or args.baseline)
    cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
