"""Kaggle kernel: unpacks the repo code, installs the pins, checks the model and starts the jobs."""
import base64
import glob
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

BUNDLE = "__BUNDLE__"
COMMANDS = json.loads(__COMMANDS__)
PINS = ["transformers==5.5.4", "datasets==5.0.0", "huggingface_hub==1.11.0", "safetensors==0.7.0"]
REPO = "/kaggle/temp/repo"
OUT = "/kaggle/working"


def run(*command, env=None):
    print("$", " ".join(command).encode("ascii", "replace").decode(), flush=True)
    begin = time.time()
    subprocess.run(command, cwd=REPO, check=True, env=env)
    print(f"-- {time.time() - begin:.0f} s", flush=True)


os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
environment = dict(os.environ, WSP_VARIANT="__VARIANT__", WSP_BATCH="32", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1")
run("nvidia-smi", env=environment)
run(sys.executable, "-m", "data.fetch", env=environment)
offline = dict(environment, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
run(sys.executable, "-m", "data.audit_model", env=offline)
for path in glob.glob("/kaggle/input/**/*.safetensors", recursive=True):
    shutil.copy(path, os.path.join(REPO, "edits"))
try:
    for command in COMMANDS:
        run(sys.executable, "-m", *shlex.split(command), env=offline)
finally:
    for folder in ("logs", "logs_05b", "report/results", "report/results_05b"):
        source = os.path.join(REPO, folder)
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(OUT, folder), dirs_exist_ok=True)
