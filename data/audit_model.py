"""Sauberkeits-Audit: kein ausfuehrbarer Code im Modell, keine Netzwerkverbindung im Betrieb."""
import json
import socket
import struct
import sys
from pathlib import Path

from eval import config as cfg
from eval.model import load_model, load_tokenizer, model_path

CODE_SUFFIXES = {".py", ".pyc", ".pkl", ".pickle", ".bin", ".pt", ".pth", ".so", ".dll", ".exe", ".sh", ".ipynb"}
ALLOWED = {"config.json", "generation_config.json", "LICENSE", "merges.txt", "model.safetensors",
           "README.md", "tokenizer.json", "tokenizer_config.json", "vocab.json"}


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FEHLER'}] {name} {detail}")
    return ok


def audit_files(root):
    names = {p.name for p in root.iterdir()}
    return [
        check("nur erwartete Dateien", names <= ALLOWED, sorted(names - ALLOWED) or ""),
        check("keine Code-/Pickle-Dateien", not any(Path(n).suffix in CODE_SUFFIXES for n in names)),
    ]


def audit_configs(root):
    results = []
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        conf = json.loads((root / name).read_text(encoding="utf-8"))
        results.append(check(f"{name}: kein auto_map (Fremdcode)", "auto_map" not in conf))
    conf = json.loads((root / "config.json").read_text(encoding="utf-8"))
    results.append(check("Architektur ist die Transformers-eigene", conf["architectures"] == ["Qwen2ForCausalLM"]))
    return results


def audit_safetensors(path):
    size = path.stat().st_size
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    meta = header.pop("__metadata__", {})
    end = max(v["data_offsets"][1] for v in header.values())
    return [
        check("Safetensors: Datei besteht lueckenlos aus Tensoren", 8 + header_len + end == size,
              f"{len(header)} Tensoren, Metadaten={meta}"),
        check("Safetensors: nur Zahlentypen", {v["dtype"] for v in header.values()} <= {"F16", "BF16", "F32"},
              {v["dtype"] for v in header.values()}),
    ]


def audit_network():
    attempts = []
    real_connect = socket.socket.connect

    def guard(self, address):
        attempts.append(repr(address))
        raise OSError(f"Netzwerk im Audit gesperrt: {address}")

    socket.socket.connect = guard
    socket.socket.connect_ex = guard
    socket.create_connection = lambda address, *a, **k: guard(None, address)
    try:
        tok = load_tokenizer()
        model = load_model("cpu")
        text = tok.apply_chat_template([{"role": "user", "content": "What is 2+2?"}], tokenize=False,
                                       add_generation_prompt=True)
        batch = tok(text, return_tensors="pt")
        output = model.generate(**batch, max_new_tokens=8, do_sample=False)
        print("   Antwort:", tok.decode(output[0, batch.input_ids.shape[1]:], skip_special_tokens=True))
    finally:
        socket.socket.connect = real_connect
    return [check("keine Verbindungsversuche bei Laden und Inferenz", not attempts, attempts or "")]


if __name__ == "__main__":
    root = Path(model_path())
    print("Snapshot:", root)
    results = audit_files(root) + audit_configs(root) + audit_safetensors(root / "model.safetensors")
    results += audit_network()
    print("ERGEBNIS:", "sauber" if all(results) else "AUFFAELLIG")
    sys.exit(0 if all(results) else 1)
