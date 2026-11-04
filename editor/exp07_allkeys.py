"""Experiment 7: hint and option keys of all layers in C1, plus the reader profile from experiment 1."""
import numpy as np
import torch

from baselines.run_b1 import split_validation
from editor import lab
from editor.an02_positions import option_positions
from editor.analysis import hint_letter_position
from eval import config as cfg
from eval.measure_train import load_train
from eval.metrics import letter_index
from eval.model import load_model, load_tokenizer
from eval.prompts import render
from eval.scoring import encode, right_pad

N = 128


@torch.inference_mode()
def record_all_keys(model, tok, texts, batch_size=16):
    device = model.get_output_embeddings().weight.device
    sequences = encode(tok, texts)
    layers = model.model.layers
    store = {}
    hooks = [layer.self_attn.k_proj.register_forward_hook(
        lambda _m, _i, out, key=i: store.__setitem__(key, out)) for i, layer in enumerate(layers)]
    hint = torch.empty(len(texts), len(layers), 128)
    options = torch.empty(len(texts), len(layers), 128)
    try:
        for start in range(0, len(sequences), batch_size):
            chunk = list(range(start, min(start + batch_size, len(sequences))))
            ids, mask = right_pad([sequences[i] for i in chunk], tok.pad_token_id, device)
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            for row, item in enumerate(chunk):
                h = hint_letter_position(tok, sequences[item])
                o = list(option_positions(tok, sequences[item]).values())
                for layer in range(len(layers)):
                    hint[item, layer] = store[layer][row, h].float().cpu()
                    options[item, layer] = store[layer][row, o].mean(0).float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
    return hint, options


def main():
    items = load_train()
    base = torch.load(cfg.LOG_DIR / "train_baseline_predictions.pt")
    fit, _ = split_validation(len(items))
    wrong = np.array([letter_index(i["wrong"]) for i in items])
    flipped = [i for i in fit if base["C1"][i].item() == wrong[i]][:N]
    number = lab.start("editor", "Hint keys of all layers for the key edit", "editor/exp07_allkeys.py", N)
    tok = load_tokenizer()
    model = load_model()
    hint, options = record_all_keys(model, tok, [render(tok, items[i], "C1") for i in flipped])
    torch.save({"hint": hint, "options": options}, cfg.LOG_DIR / "keys_all.pt")
    lab.finish("editor", number, {"saved": "logs_05b/keys_all.pt", "items": len(flipped)})
    print("done", len(flipped))


if __name__ == "__main__":
    main()
