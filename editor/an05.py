"""Leseprofil je Layer und Trennschaerfe der Hinweis-Richtung (ohne Modelllauf)."""
import torch
from torch.nn.functional import cosine_similarity as cos

from editor.analysis import attention_to, hint_letter_position, load_probe, tokenizer
from eval import config as cfg

KV_DIM = 64
tok = tokenizer()
items, data = load_probe("probe01.pt")
c1 = data["C1"]
hint_attention = attention_to(c1, [hint_letter_position(tok, ids) for ids in c1["ids"]]).mean(0)
keys = torch.load(cfg.LOG_DIR / "keys_all.pt")
print("Layer | Summe Head-Attention auf Hinweis | staerkster Head | je KV-Gruppe: Anteil ausserhalb, d*Hinweis, d*Option")
for layer in range(hint_attention.shape[0]):
    row = hint_attention[layer]
    text = f"{layer:2d} {row.sum():5.2f} {row.argmax().item():2d}:{row.max():.2f}"
    for group in range(2):
        cols = slice(group * KV_DIM, (group + 1) * KV_DIM)
        h, o = keys["hint"][:, layer, cols], keys["options"][:, layer, cols]
        hm, om = h.mean(0), o.mean(0)
        d = hm - (hm @ om) / (om @ om) * om
        share = d.norm() / hm.norm()
        d = d / d.norm()
        text += f" | {share:.2f} {(h @ d).mean():6.2f}+-{(h @ d).std():.2f} {(o @ d).mean():5.2f} cos{cos(hm, om, dim=0):.2f}"
    print(text)
