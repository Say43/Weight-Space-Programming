"""Multiple-Choice-Scoring: ein Forward-Pass pro Item, Logits des Buchstabens nach dem Praefix."""
import os
import time

import torch

from eval.prompts import LETTERS, render

BATCH_SIZE = int(os.environ.get("WSP_BATCH", 8))
PROGRESS_SECONDS = 60


def letter_token_ids(tok):
    ids = [tok(letter, add_special_tokens=False).input_ids for letter in LETTERS]
    assert all(len(i) == 1 for i in ids), "Buchstaben muessen genau ein Token sein"
    return [i[0] for i in ids]


def encode(tok, texts):
    return [tok(t, add_special_tokens=False).input_ids for t in texts]


def right_pad(sequences, pad_id, device):
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq, device=device)
        mask[row, : len(seq)] = 1
    return ids, mask


@torch.inference_mode()
def score_texts(model, tok, texts, batch_size=BATCH_SIZE):
    """Gibt Logits [N, 5] der Buchstaben A-E in Eingabereihenfolge zurueck."""
    device = model.get_output_embeddings().weight.device
    letter_ids = torch.tensor(letter_token_ids(tok), device=device)
    sequences = encode(tok, texts)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    out = torch.empty(len(sequences), len(LETTERS))
    begin = last_report = time.time()
    for start in range(0, len(order), batch_size):
        if time.time() - last_report > PROGRESS_SECONDS:
            last_report = time.time()
            print(f"Scoring {start}/{len(order)} nach {last_report - begin:.0f} s", flush=True)
        chunk = order[start : start + batch_size]
        ids, mask = right_pad([sequences[i] for i in chunk], tok.pad_token_id, device)
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = hidden[torch.arange(len(chunk), device=device), mask.sum(1) - 1]
        logits = model.lm_head(last)[:, letter_ids].float().cpu()
        out[torch.tensor(chunk)] = logits
    return out


def predict(logits, n_choices):
    """Waehlt pro Item den besten Buchstaben unter den tatsaechlich vorhandenen Optionen."""
    mask = torch.arange(logits.shape[1])[None, :] >= torch.as_tensor(n_choices)[:, None]
    return logits.masked_fill(mask, float("-inf")).argmax(1)


def score_items(model, tok, items, condition, system=None, batch_size=BATCH_SIZE):
    texts = [render(tok, item, condition, system) for item in items]
    return predict(score_texts(model, tok, texts, batch_size), [len(i["choices"]) for i in items])
