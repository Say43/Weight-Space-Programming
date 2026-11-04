"""Where do the heads of layer 16 look in C0 and C1? (no model run)"""
import torch

from editor.an02_positions import option_positions
from editor.analysis import hint_letter_position, load_probe, tokenizer

tok = tokenizer()
items, data = load_probe("probe01.pt")
HEADS = [(16, h) for h in (2, 3, 7, 9, 11, 12)] + [(9, 13), (11, 12), (20, 8), (20, 7)]


print("Option tokens:", [tok.decode([t]) for t in data["C0"]["ids"][0][30:42]])
for condition in ("C0", "C1"):
    record = data[condition]
    print(condition, "head | correct option | hinted option | other options | hint letter | rest")
    for layer, head in HEADS:
        rows = []
        for n, (item, ids) in enumerate(zip(items, record["ids"])):
            row = record["attention"][n][layer, head]
            options = option_positions(tok, ids)
            right = row[options[item["answer"]]].item()
            hinted = row[options[item["wrong"]]].item()
            others = sum(row[p].item() for l, p in options.items() if l not in (item["answer"], item["wrong"]))
            letter = row[hint_letter_position(tok, ids)].item() if condition == "C1" else 0.0
            rows.append([right, hinted, others, letter, 1 - right - hinted - others - letter])
        mean = torch.tensor(rows).mean(0)
        print(f"{layer}.{head}", " ".join(f"{v:.3f}" for v in mean))
