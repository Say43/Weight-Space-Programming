"""Laedt Modell und Datensaetze auf gepinnten Commits und prueft das Modell gegen den Hub."""
import hashlib
import json

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

from eval import config as cfg

PATTERNS = ["*.safetensors", "*.json", "merges.txt", "LICENSE", "README.md"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(repo, rev):
    path = snapshot_download(repo, revision=rev, allow_patterns=PATTERNS, token=False)
    info = HfApi(token=False).model_info(repo, revision=rev, files_metadata=True)
    expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    actual = sha256(f"{path}/model.safetensors")
    ok = actual == expected["model.safetensors"]
    print(json.dumps({"repo": repo, "rev": rev, "path": path, "sha256": actual, "matches_hub": ok}))
    if not ok:
        raise SystemExit("Pruefsumme des Modells stimmt nicht mit dem Hub ueberein.")
    return path


if __name__ == "__main__":
    fetch_model(cfg.MODEL_ID, cfg.MODEL_SHA)
    for name in ("ARC-Easy", "ARC-Challenge"):
        print(name, load_dataset(cfg.ARC_ID, name, revision=cfg.ARC_SHA, token=False))
    print(load_dataset(cfg.MMLU_ID, "all", split="test", revision=cfg.MMLU_SHA, token=False))
    print(load_dataset(cfg.WIKI_ID, "wikitext-2-raw-v1", split="test", revision=cfg.WIKI_SHA, token=False))
