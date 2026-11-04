"""Capability guards: MMLU, perplexity on wikitext-2 and open prompts for visual inspection."""
import numpy as np
import torch
from datasets import load_dataset
from torch.nn.functional import cross_entropy

from data.sources import load_mmlu
from eval import config as cfg
from eval.metrics import letter_index, rate_ci
from eval.scoring import score_items

PERPLEXITY_TOKENS = 100_000
CHUNK = 256

OPEN_PROMPTS = [
    "Explain how photosynthesis works in two sentences.",
    "Write a short poem about autumn rain.",
    "What are the main differences between TCP and UDP?",
    "Give three tips for learning a new language.",
    "Write a Python function that reverses a string.",
    "Why is the sky blue?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What causes the seasons on Earth?",
    "Describe how a bicycle stays upright while moving.",
    "Write a friendly email inviting a colleague to lunch.",
    "What is the Pythagorean theorem? Give an example.",
    "How does a refrigerator keep food cold?",
    "List the planets of the solar system in order.",
    "Explain what a binary search is.",
    "What is the difference between weather and climate?",
    "Translate 'Good morning, how are you?' into German.",
    "Compute 17 times 23 and show your steps.",
    "Tell a short story about a lighthouse keeper.",
    "What are vaccines and how do they work?",
    "Explain supply and demand to a ten-year-old.",
]


def mmlu_accuracy(model, tok, system=None):
    items = load_mmlu()
    predictions = score_items(model, tok, items, "C0", system)
    correct = np.array([letter_index(i["answer"]) for i in items])
    return rate_ci(predictions.numpy() == correct)


@torch.inference_mode()
def perplexity(model, tok, start=0, tokens=PERPLEXITY_TOKENS):
    rows = load_dataset(cfg.WIKI_ID, "wikitext-2-raw-v1", split="test", revision=cfg.WIKI_SHA, token=False)
    ids = tok("\n\n".join(rows["text"][:6000]), add_special_tokens=False).input_ids[start : start + tokens]
    device = model.get_output_embeddings().weight.device
    nll = []
    for begin in range(0, len(ids) - CHUNK, CHUNK):
        chunk = torch.tensor([ids[begin : begin + CHUNK]], device=device)
        logits = model(input_ids=chunk, use_cache=False).logits
        loss = cross_entropy(logits[0, :-1].float(), chunk[0, 1:], reduction="mean")
        nll.append(loss.item())
    nll = np.array(nll)
    rng = np.random.default_rng(cfg.SEED)
    boot = nll[rng.integers(0, len(nll), (2000, len(nll)))].mean(1)
    low, high = np.percentile(np.exp(boot), [2.5, 97.5])
    return {"perplexity": float(np.exp(nll.mean())), "ci95": [float(low), float(high)], "tokens": len(nll) * CHUNK}


@torch.inference_mode()
def open_generations(model, tok, system=None):
    device = model.get_output_embeddings().weight.device
    out = []
    for prompt in OPEN_PROMPTS:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        ids = model.generate(**batch, max_new_tokens=128, do_sample=False, temperature=None, top_p=None, top_k=None,
                             repetition_penalty=None, pad_token_id=tok.pad_token_id)
        out.append({"prompt": prompt, "output": tok.decode(ids[0, batch.input_ids.shape[1] :], skip_special_tokens=True)})
    return out
