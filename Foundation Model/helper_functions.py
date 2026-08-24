import torch
import time

import numpy as np

from pretraining import GPT_CONFIG_124M, text_to_token_ids

def generate(model, idx, max_new_tokens, context_size, temperature=0.0,
             top_k=None, eos_id=None):
    """Generate text token by token with temperature scaling and top-k sampling."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent
            # results on mps device: subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the
        # highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop generating early if end-of-sequence token is encountered
        if idx_next == eos_id:
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def benchmark(model, tokenizer, prompt, max_new_tokens=100, runs=5):
    """Benchmark inference speed in new tokens generated per second.

    Runs a short warm-up pass, then times `runs` generations of
    `max_new_tokens` tokens from `prompt`, reporting each run plus the
    mean and standard deviation. Returns the list of per-run speeds.
    """
    speeds = []

    # Warm-up to let caches and lazy initialization settle
    idx = text_to_token_ids(prompt, tokenizer)
    with torch.no_grad():
        generate(
            model=model,
            idx=idx,
            max_new_tokens=10,
            context_size=GPT_CONFIG_124M["context_length"],
            top_k=50,
            temperature=1.5
        )

    for i in range(runs):
        idx = text_to_token_ids(prompt, tokenizer)

        start = time.perf_counter()

        with torch.no_grad():
            ids = generate(
                model=model,
                idx=idx,
                max_new_tokens=max_new_tokens,
                context_size=GPT_CONFIG_124M["context_length"],
                top_k=50,
                temperature=1.5
            )

        elapsed = time.perf_counter() - start
        new_tokens = ids.shape[1] - idx.shape[1]
        speed = new_tokens / elapsed

        speeds.append(speed)
        print(f"Run {i+1}: {speed:.2f} tokens/s")

    print(f"\nMean: {np.mean(speeds):.2f} tokens/s")
    print(f"Standard deviation: {np.std(speeds):.2f} tokens/s")

    return speeds
