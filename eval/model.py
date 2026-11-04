"""Laedt das gepinnte Modell ausschliesslich aus dem lokalen Cache, ohne Fremdcode."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import config as cfg


def model_path(repo=cfg.MODEL_ID, rev=cfg.MODEL_SHA):
    return snapshot_download(repo, revision=rev, local_files_only=True, token=False)


def load_tokenizer(path=None):
    return AutoTokenizer.from_pretrained(path or model_path(), trust_remote_code=False)


def load_model(device="cuda", path=None, attn="sdpa"):
    model = AutoModelForCausalLM.from_pretrained(
        path or model_path(),
        dtype=torch.float32,
        trust_remote_code=False,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    )
    model.eval().requires_grad_(False)
    return model.to(device)
