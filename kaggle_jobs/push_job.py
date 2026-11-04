"""Packt den Repo-Code in einen privaten Kaggle-Script-Kernel und pusht ihn.

Das Modell laedt der Kernel selbst von Hugging Face auf dem gepinnten Commit und prueft es mit
data.fetch und data.audit_model. Es werden weder Gewichte noch Keys hochgeladen.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from eval import config as cfg

PACKAGES = ("data", "eval", "edits", "editor", "baselines")
EXTRA_FILES = ("data/splits/handcheck.jsonl", "requirements.txt")
TEMPLATE = Path(__file__).with_name("runner_template.py")
BUILD_DIR = cfg.ROOT / ".cache" / "kaggle"


def bundle(extra=()):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for package in PACKAGES:
            for path in sorted((cfg.ROOT / package).glob("*.py")):
                archive.write(path, path.relative_to(cfg.ROOT).as_posix())
        for name in (*EXTRA_FILES, *extra):
            archive.write(cfg.ROOT / name, name)
    return base64.b64encode(buffer.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("commands", nargs="+", help="Module, z. B. data.build_splits")
    parser.add_argument("--with-file", action="append", default=[], help="zusaetzliche Repo-Datei")
    parser.add_argument("--variant", default="1.5b", choices=["1.5b", "0.5b"])
    parser.add_argument("--dataset", action="append", default=[], help="privates Dataset owner/slug")
    args = parser.parse_args()
    folder = BUILD_DIR / args.slug
    folder.mkdir(parents=True, exist_ok=True)
    script = TEMPLATE.read_text(encoding="utf-8")
    script = script.replace("__BUNDLE__", bundle(args.with_file)).replace("__COMMANDS__", repr(json.dumps(args.commands)))
    script = script.replace("__VARIANT__", args.variant)
    (folder / "run.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{os.environ['KAGGLE_USERNAME']}/{args.slug}", "title": args.slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": args.dataset, "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(folder)], check=True)


if __name__ == "__main__":
    main()
