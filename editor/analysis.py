"""Auswertung gespeicherter Aufzeichnungen und reine Gewichtsanalyse. Fuehrt das Modell nie aus."""
import torch
from safetensors import safe_open

from eval import config as cfg
from eval.measure_train import load_train
from eval.metrics import letter_index
from eval.model import load_tokenizer, model_path
from eval.scoring import letter_token_ids

HEAD_DIM = 64
HEADS = 14


def weights(*names):
    with safe_open(f"{model_path()}/model.safetensors", "pt") as f:
        return {n: f.get_tensor(n).float() for n in names}


def load_probe(name):
    probe = torch.load(cfg.LOG_DIR / name)
    items = load_train()
    return [items[i] for i in probe["indices"]], probe["data"]


def letter_vectors(tok):
    embed = weights("model.embed_tokens.weight")["model.embed_tokens.weight"]
    return embed[torch.tensor(letter_token_ids(tok))]


def readout(tok, items, record, kind):
    """Skalierte Lesevektoren a_i, mit a_i * x_final = Logit-Differenz (Ziel minus Gegenseite)."""
    u = letter_vectors(tok)
    norm = weights("model.norm.weight")["model.norm.weight"]
    first = {"hint_vs_correct": "wrong", "correct_vs_hint": "answer"}[kind]
    second = {"hint_vs_correct": "answer", "correct_vs_hint": "wrong"}[kind]
    target = u[[letter_index(i[first]) for i in items]] - u[[letter_index(i[second]) for i in items]]
    x = record["final_in"]
    rms = x.pow(2).mean(1, keepdim=True).add(1e-6).sqrt()
    return target * norm / rms


def layer_attribution(record, a):
    attn = (record["attn_out"] * a[:, None]).sum(-1)
    mlp = (record["mlp_out"] * a[:, None]).sum(-1)
    return attn, mlp


def head_attribution(record, a):
    """Beitrag jedes Heads [N, Layer, Head] ueber die o_proj-Spalten dieses Heads."""
    n, depth = record["heads"].shape[:2]
    out = torch.empty(n, depth, HEADS)
    for layer in range(depth):
        o_proj = weights(f"model.layers.{layer}.self_attn.o_proj.weight")[f"model.layers.{layer}.self_attn.o_proj.weight"]
        projected = a @ o_proj
        z = record["heads"][:, layer]
        for head in range(HEADS):
            cols = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
            out[:, layer, head] = (projected[:, cols] * z[:, cols]).sum(-1)
    return out


def hint_letter_position(tok, ids):
    """Position des Buchstabens im Hinweissatz: letztes Vorkommen von 'is (' + Buchstabe."""
    letters = set(letter_token_ids(tok))
    marker = tok(" is (", add_special_tokens=False).input_ids
    for pos in range(len(ids) - 1, len(marker) - 1, -1):
        if ids[pos] in letters and ids[pos - len(marker) : pos] == marker and pos < len(ids) - 1:
            return pos
    raise ValueError("Hinweis nicht gefunden")


def attention_to(record, positions):
    """Attention [N, Layer, Head] von der letzten Position auf die gegebene Position je Prompt."""
    return torch.stack([record["attention"][i][:, :, p] for i, p in enumerate(positions)])


def tokenizer():
    return load_tokenizer()
