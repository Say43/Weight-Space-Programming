"""Compares keys of hint letters, options and the forced answer in layer 16 (no model run)."""
import torch
from torch.nn.functional import cosine_similarity as cos

from editor.an02_positions import option_positions
from editor.analysis import hint_letter_position, load_probe, tokenizer

KV_DIM = 64
tok = tokenizer()
items, data = load_probe("probe04.pt")
c1, c3 = data["C1"], data["C3"]


def forced_answer_position(ids):
    """Last letter after ' (' in the forced assistant answer of C3."""
    return hint_letter_position(tok, ids)


hint, options, forced = [], [], []
for n, ids in enumerate(c1["ids"]):
    hint.append(c1["k"][n][hint_letter_position(tok, ids)])
    options.append(c1["k"][n][list(option_positions(tok, ids).values())].mean(0))
    forced.append(c3["k"][n][forced_answer_position(c3["ids"][n])])
hint, options, forced = torch.stack(hint), torch.stack(options), torch.stack(forced)
torch.save({"hint": hint, "options": options, "forced": forced}, "logs_05b/keys16.pt")

for group in range(2):
    cols = slice(group * KV_DIM, (group + 1) * KV_DIM)
    h, o, f = hint[:, cols], options[:, cols], forced[:, cols]
    hm, om, fm = h.mean(0), o.mean(0), f.mean(0)
    print(f"KV group {group}: norm hint {h.norm(dim=1).mean():.2f}, option {o.norm(dim=1).mean():.2f}, "
          f"C3 answer {f.norm(dim=1).mean():.2f}")
    print(f"  cos(hint, option) {cos(hm, om, dim=0):.3f}, cos(hint, C3 answer) {cos(hm, fm, dim=0):.3f}, "
          f"spread cos(single, mean) hint {cos(h, hm[None]).mean():.3f}, option {cos(o, om[None]).mean():.3f}")
    d = hm - (hm @ om) / (om @ om) * om
    print(f"  share of the hint key outside the option direction: {d.norm() / hm.norm():.3f}")
    d = d / d.norm()
    print(f"  projection on d: hint {(h @ d).mean():.2f} +- {(h @ d).std():.2f}, option {(o @ d).mean():.2f}, "
          f"C3 answer {(f @ d).mean():.2f}")
