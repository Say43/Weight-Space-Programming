# Repairing Sycophancy by Direct Weight Editing: Report

**Main finding.** A gradient-free edit of 0.23 % of the parameters lowers the C1 flip rate of Qwen2.5-0.5B on the test split from 76.8 % (95 % CI 74.2–79.4) to 32.3 % (CI 29.4–35.2), a relative reduction of 57.9 %, and meets all criteria fixed in advance.

The main caveat lies outside those criteria. On questions the model does not know, a *correct* hint helps less after the edit (93.8 % to 65.2 %, 112 items). The edit lowers hint-copying in general; it does not teach the model to tell good hints from bad ones.

The result holds for the 0.5B model, not for the original target 1.5B. On `train`, the C1 flip rate of 1.5B was 15.0 % (CI 12.8–17.2), below the go threshold of 30 %. As the plan prescribes, the study moved to 0.5B (`train`: 79.9 %, CI 77.4–82.4).

## 1. Results

Test split with 999 known items; MMLU with 1 000 fixed questions; perplexity on 99 840 tokens. Values in brackets are 95 % bootstrap intervals.

| condition | C1 flip % | C3 flip % | C0 acc % | C2 acc % | MMLU % | perplexity | params changed % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Unedited model | 76.8 (74.2–79.4) | 6.5 (5.0–8.1) | 100.0 | 100.0 | 43.0 (40.0–46.0) | 22.87 | 0 |
| B1 difference of means | 65.3 (62.4–68.2) | 12.4 (10.4–14.4) | 93.9 (92.4–95.3) | 99.8 | 42.9 (39.9–45.9) | 25.33 | 16.72 |
| B2 random direction | 80.0 (77.5–82.5) | 6.0 (4.6–7.5) | 98.1 (97.2–98.9) | 100.0 | 43.1 (40.1–46.2) | 23.29 | 16.72 |
| B3 system prompt | 71.2 (68.5–74.0) | 22.0 (19.5–24.6) | 91.9 (90.2–93.5) | 99.9 | 42.7 (39.6–45.8) | 22.87 | 0 |
| **e1, primary (s = 0.4)** | **32.3 (29.4–35.2)** | 4.9 (3.6–6.3) | 98.7 (97.9–99.3) | 99.1 (98.5–99.6) | 43.0 (40.0–46.1) | 22.88 | 0.23 |
| e2 (s = 0.5) | 24.8 (22.1–27.5) | 4.7 (3.4–6.1) | 98.6 (97.8–99.3) | 97.9 (97.0–98.7) | 43.0 (39.9–46.1) | 22.88 | 0.23 |
| e3 (s = 0.3) | 41.9 (38.8–45.0) | 5.9 (4.5–7.4) | 98.9 (98.2–99.5) | 99.8 | 43.2 (40.2–46.3) | 22.87 | 0.23 |

B4 and the optional extension were dropped in the shortened run. The perplexity of B3 equals that of the unedited model because it is measured without a prompt and B3 changes no weights.

### Check against the success criteria (e1 only)

| criterion | threshold | measured | met |
| --- | --- | --- | --- |
| C1 flip, relative | at least −50 % | −57.9 % (−52.5 % at the least favourable interval bounds) | yes |
| C0 loss | at most 2 points | 1.3 | yes |
| C2 loss | at most 3 points | 0.9 | yes |
| MMLU loss | at most 2 points | 0.0 | yes |
| Perplexity | at most +5 % | +0.03 % | yes |
| Lead over B1 | intervals do not overlap | 29.4–35.2 against 62.4–68.2 | yes |

**Robustness.** The wording `My teacher said it's ({wrong}).`, never used in the editor run, falls from 87.3 % (CI 85.2–89.4) to 38.8 % (CI 35.8–41.9), a relative reduction of 55.5 %. The edit is therefore not tied to the wording of the first hint.

**Visual inspection.** 6 of 20 open generations are identical. The others differ in ordinary phrasing, and none is gibberish. One example: "Guten Morgen, wie geht es Ihnen?" becomes "…wie geht es dir?".

## 2. Pareto plot

![C1 flip rate against MMLU loss and C0 loss](pareto.png)

MMLU does not separate the conditions: all points lie within 0.3 points, far below the uncertainty of about ±3 points. The right panel therefore shows the more informative C0 loss. There, the three editor candidates dominate all baselines. Among themselves, e1, e2 and e3 trade flip rate against C2.

## 3. Mechanism

### Finding

Sycophancy in this model is not one feature. It arises because the hint letter competes with the answer options for the same pointer.

1. **Layer 16 is the answer pointer.** In C0, head 16.7 attends 64 % from the answer position to the marker of the correct option, and 16.11 attends 33 %.
2. **The hint captures the pointer.** In C1, heads 16.2 (83 %), 16.12 (64 %) and 16.11 (55 %) attend to the hint letter.
3. **There is a second route.** Heads 9.13 and 11.12 hardly attend to letters in C0 (below 0.5 %), but attend 90 % and 80 % to the hint in C1. Head 16.7 seeks the option named by the hint with 40 % of its attention in C1. The letter therefore reaches the answer position before layer 16. A counterfactual calculation supports this: the key edit in layer 16 alone almost removes the attention there (16.2: 83 % to 5 %), yet the flip rate falls only to 55 %.
4. **The effect is spread across layers.** Single layers lower the flip rate only slightly: layers 9 and 11 alone reach 67 % each and 58 % together. A strong effect needs all of layers 9 to 18.

Attribution places the logit shift (+5.24) in layers 16 to 23. It captures direct contributions only, and it misled: head 22.2 had the highest attribution and no causal effect, while 16.2 had almost no attribution and a large effect. Only targeted muting revealed this.

### Why the heads cannot simply be muted

Muting six heads of layer 16 lowers the flip rate to 15 % but destroys the model: C0 82 %, C2 59 %, C3 flip 84 %. Those heads are the pointer itself. The edit must make the hint unattractive and keep the pointer.

### The edit

The edit targets `k_proj` of layers 9 to 18 in both key-value groups g. Keys decide how strongly a position is attended. For each row block W_g:

```
W_g ← W_g − s · r wᵀ W_g

m = mean key of the hint letter (128 flipped train items)
o = mean key of the option markers
d = normalised part of m orthogonal to o
r = d dᵀ m
C = key covariance in use during C0 (attention-weighted) + option covariance with 3 times its trace
w = (C + λ·tr(C)/64 · I)⁻¹ m, normalised to wᵀm = 1, λ = 0.01
s = 0.4
```

This is an oblique rank-1 projection. Hint keys lose part of r because wᵀm = 1. Keys used in C0 stay nearly untouched because w is chosen to be insensitive to them. Compared with the orthogonal projection, the changed key energy falls by a factor of about 10.

Size: 10 matrices, 1 146 880 weights (0.232 %), rank 2 per matrix (rank 1 per key-value group). The embeddings and the LM head are unchanged. Re-applying the edit gives bit-identical weights. The edit is well below the 1 % limit and 72 times smaller than B1.

### Hypotheses tested

- **Copy heads: confirmed, with a twist.** The copying path cannot be damped through a few heads, because the same heads select the answer.
- **Deference direction: confirmed.** Head contributions in C1 and C2 correlate at 0.86. The path is the same whether or not the hint is correct.
- **Pushback path: partly resolved.** The forced answer in C3 has almost the same key as the hint (cosine 0.98), but the edit does not raise C3 (6.5 % to 4.9 %, overlapping intervals). A separate path for "Are you sure?" was not studied, because C3 rarely flips in this model.

## 4. Budget

| quantity | value |
| --- | --- |
| Editor experiments | 17 of 40 |
| Baseline experiments | 9 of 10 |
| To the first effective edit | 2 experiments (still harmful) |
| To the first gentle candidate | 14 experiments |
| Wall-clock time | about 4.5 hours, including about 2 hours on one abandoned local run |
| Kaggle GPU time | about 0.8 hours |
| Local GPU time | about 2.5 hours, of which 2 hours produced no result |
| Cost | no Kaggle or API charges; the editor's work runs on the Claude quota |

The model audit passed for both models: numeric tensors only, no remote code, no connection attempts during loading and inference. The 0.5B audit also passed inside the Kaggle kernel.

## 5. Limits

- **One model, one task family, multiple choice only.** There is no evidence for 1.5B, for open answers or for other models. Hints always take the form `(X)`. A hint without brackets or in words was not tested.
- **The edit does not solve the trust problem.** It lowers copying in general. The criteria measure known items only and miss this cost. The figure of 65.2 % comes from a single additional test on 112 items outside `test`, for e1, and is preliminary.
- **Almost a third of the flips remain.** 32.3 % is far from zero. A smaller s lowers the harm, a larger one the flips. e2 is better on C1 but moves closer to the C2 limit.
- **Noise floor.** On `train`, a meaningless perturbation of the same size also cost about 2 points of C0. That amount reflects the selection of barely correct items rather than the mechanism. The same control raised flips to 94.5 %: the behaviour is unstable, in both directions.
- **Editor against B1.** The editor clearly outperforms B1 and B3. However, B1 is a fixed baseline that could test only 8 of 24 layers within its budget, and the editor used almost twice as many experiments (17 against 9). A better way of finding the direction might improve B1.
- **Comparability across devices.** The baseline ran on the local GPU, all other conditions on a Kaggle T4. Predictions agreed on 40 of 40 sampled items, and C0 stayed at 100 % across devices.
- **Safety behaviour.** Refusal and safety behaviour were not a target of the edit. They were not measured either, so it is not shown that they are unchanged.
- **Position effect.** In 1.5B the flip rate rose with the hint position from 8 % (A) to 25 % (D). This was not analysed for 0.5B.

**Deviations from the plan.** Target model 0.5B instead of 1.5B (the planned no-go path); B1 with 8 layers instead of all (approved); shortened run without B4 (approved); phase 4 on Kaggle instead of locally (on request); the first local run of B1 was stopped before it wrote a result.

## 6. Comparison with the source post

The point of reference is the blog post ["Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/). The attribution to MIT CSAIL comes from the project brief and was not checked independently here.

| aspect | post | this study |
| --- | --- | --- |
| Sycophancy | detected only, experiment 7 | reduced |
| Model | Gemma 3 270M, fine-tuned for the purpose | Qwen2.5-0.5B, as released |
| Editor | GPT-6 Astra | Claude Opus 5.5 |
| Edit site | MLP columns, looping direction | `k_proj`, rank 1 per key-value group |
| Edit size | 0.016 % (Qwen3.5-0.8B) | 0.23 % |

**Confirmed.**

- A language model can change the weights of a small model in a targeted way without gradients.
- A few experiments suffice for a first effective edit.
- Attention patterns are informative about sycophancy. The post found one head in Gemma that looks from the answer position to the user's answer (10.2). The same pattern appears here, spread over several heads.
- Edits can remove obstacles. Capability (MMLU) stays the same, and no new ability appears.

**New.**

- A repair instead of detection only, measured on a blind test split against mechanical baselines.
- The pointer finding: the copying path and normal answer selection use the same heads, so the intervention has to act on the key side.
- A general recipe for gentle edits: use a filter derived from the usage covariance instead of an orthogonal projection.
- The price: on unknown questions, correct hints help less.

The figures cannot be compared directly with the post, because model, task and source of sycophancy differ.
