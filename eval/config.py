"""Pinned sources and paths. All revisions are commit hashes from 2026-11-04."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SHA = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_MODEL_SHA = "7ae557604adf67be50417f59c2c2f167def9a775"

ARC_ID, ARC_SHA = "allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"
MMLU_ID, MMLU_SHA = "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe"
WIKI_ID, WIKI_SHA = "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"

VARIANT = os.environ.get("WSP_VARIANT", "1.5b")
SUFFIX = {"1.5b": "", "0.5b": "_05b"}[VARIANT]
if VARIANT == "0.5b":
    MODEL_ID, MODEL_SHA = FALLBACK_MODEL_ID, FALLBACK_MODEL_SHA

HANDCHECK_PATH = ROOT / "data" / "splits" / "handcheck.jsonl"
SPLIT_DIR = ROOT / "data" / f"splits{SUFFIX}"
TRAIN_PATH = SPLIT_DIR / "train.jsonl"
TEST_PATH = SPLIT_DIR / "test.jsonl"
LOG_DIR = ROOT / f"logs{SUFFIX}"
EXPERIMENT_LOG = LOG_DIR / "experiments.jsonl"
RESULT_DIR = ROOT / "report" / f"results{SUFFIX}"
EDIT_DIR = ROOT / "edits"

SEED = 0
