"""Auswertung von Experiment 1 (ohne Modelllauf)."""
import torch

from editor.analysis import (attention_to, head_attribution, hint_letter_position, layer_attribution, load_probe,
                             readout, tokenizer)
from eval.prompts import LETTERS
from eval.scoring import letter_token_ids

tok = tokenizer()
items, data = load_probe("probe01.pt")
c0, c1, c2 = data["C0"], data["C1"], data["C2"]

a0 = readout(tok, items, c0, "hint_vs_correct")
a1 = readout(tok, items, c1, "hint_vs_correct")
logit_c0 = (a0 * c0["final_in"]).sum(-1)
logit_c1 = (a1 * c1["final_in"]).sum(-1)
print(f"Logit Hinweis minus richtig: C0 {logit_c0.mean():.2f}, C1 {logit_c1.mean():.2f}")

# Attribution mit festem Lesevektor aus C0, damit die Differenz nur aus dem Residual-Stream stammt.
attn0, mlp0 = layer_attribution(c0, a0)
attn1, mlp1 = layer_attribution(c1, a0)
print("Layer  dAttn  dMLP   (C1 minus C0, Lesevektor C0)")
for layer in range(attn0.shape[1]):
    print(f"{layer:5d} {(attn1 - attn0)[:, layer].mean():6.2f} {(mlp1 - mlp0)[:, layer].mean():6.2f}")
print(f"Summe Attn {(attn1 - attn0).sum(1).mean():.2f}, MLP {(mlp1 - mlp0).sum(1).mean():.2f}")

head0 = head_attribution(c0, a0)
head1 = head_attribution(c1, a0)
delta = (head1 - head0).mean(0)
positions = [hint_letter_position(tok, ids) for ids in c1["ids"]]
to_hint = attention_to(c1, positions).mean(0)
print("Top-Heads nach Attribution: Layer.Head dLogit Attention_auf_Hinweisbuchstabe")
for value, index in zip(*delta.flatten().topk(10)):
    layer, head = divmod(index.item(), delta.shape[1])
    print(f"{layer}.{head} {value:.2f} {to_hint[layer, head]:.3f}")
print("Top-Heads nach Attention auf den Hinweisbuchstaben: Layer.Head Attention dLogit")
for value, index in zip(*to_hint.flatten().topk(10)):
    layer, head = divmod(index.item(), to_hint.shape[1])
    print(f"{layer}.{head} {value:.3f} {delta[layer, head]:.2f}")

# Vergleich mit C2: dort ist der Hinweis richtig, also schaut man auf das Ziel 'richtig minus falsch'.
a_c2 = readout(tok, items, c0, "correct_vs_hint")
head2 = head_attribution(c2, a_c2)
head0b = head_attribution(c0, a_c2)
delta2 = (head2 - head0b).mean(0)
corr = torch.corrcoef(torch.stack([delta.flatten(), delta2.flatten()]))[0, 1]
print(f"Korrelation der Head-Beitraege C1 gegen C2: {corr:.3f}")
ids = letter_token_ids(tok)
print("Letter-Token-Ids:", dict(zip(LETTERS, ids)))
