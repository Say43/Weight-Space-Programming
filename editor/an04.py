"""Rechnet die Attention aus q und k nach und tauscht sie kontrafaktisch (ohne Modelllauf)."""
import math

import torch

from editor.an02_positions import option_positions
from editor.analysis import hint_letter_position, load_probe, tokenizer
from editor.exp06_keyedit import hint_key_directions

HEAD_DIM = 64
HEADS = (2, 3, 7, 9, 11, 12)
THETA = 1_000_000.0
tok = tokenizer()
items, data = load_probe("probe04.pt")
c0, c1 = data["C0"], data["C1"]
frequencies = 1.0 / THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)


def rotate(x, positions):
    """RoPE wie in Qwen2: x cos + rotate_half(x) sin, je Position."""
    angles = positions[:, None] * frequencies
    embedding = torch.cat([angles, angles], -1)
    half = HEAD_DIM // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * embedding.cos() + rotated * embedding.sin()


def head_query(record, n, head):
    return record["q"][n][head * HEAD_DIM : (head + 1) * HEAD_DIM]


def head_keys(record, n, head):
    group = 0 if head < 7 else 1
    return record["k"][n][:, group * HEAD_DIM : (group + 1) * HEAD_DIM]


def attention(query, keys):
    length = keys.shape[0]
    positions = torch.arange(length).float()
    q = rotate(query[None], positions[-1:])
    k = rotate(keys, positions)
    return ((q @ k.T)[0] / math.sqrt(HEAD_DIM)).softmax(-1)


directions = hint_key_directions()
print("Head | Hinweisbuchstabe: echt | q aus C0 | k ohne d | k mit s=2 || Optionen: echt | q aus C0 | k ohne d | k mit s=2")
for head in HEADS:
    group = 0 if head < 7 else 1
    d = directions[group].float()
    rows = []
    for n, ids in enumerate(c1["ids"]):
        hint = hint_letter_position(tok, ids)
        options = list(option_positions(tok, ids).values())
        keys = head_keys(c1, n, head)
        q1 = head_query(c1, n, head)
        q0 = head_query(c0, n, head)
        removed = keys - (keys @ d)[:, None] * d
        doubled = keys - 2 * (keys @ d)[:, None] * d
        variants = [attention(q1, keys), attention(q0, keys), attention(q1, removed), attention(q1, doubled)]
        rows.append([v[hint].item() for v in variants] + [v[options].sum().item() for v in variants])
    mean = torch.tensor(rows).mean(0)
    print(head, " ".join(f"{v:.3f}" for v in mean[:4]), "||", " ".join(f"{v:.3f}" for v in mean[4:]))
delta = torch.stack([c1["q"][n] - c0["q"][n] for n in range(len(c1["q"]))])
print("Query-Verschiebung: mittlere Norm je Head / Norm der mittleren Verschiebung / Norm q")
for head in HEADS:
    block = delta[:, head * HEAD_DIM : (head + 1) * HEAD_DIM]
    norm = torch.stack(c1["q"])[:, head * HEAD_DIM : (head + 1) * HEAD_DIM].norm(dim=1).mean()
    print(head, f"{block.norm(dim=1).mean():.2f} {block.mean(0).norm():.2f} {norm:.2f}")
