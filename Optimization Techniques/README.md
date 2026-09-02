# Optimization Techniques

This directory collects small, self-contained experiments that improve the
efficiency of autoregressive transformer inference. The implementations build
on the GPT components used elsewhere in this project and focus on two related
ideas: reusing attention keys and values during decoding, and reducing the
number of key/value heads.

## Contents

| Path | Purpose |
| --- | --- |
| `attention_mechanisms.py` | Feed-forward, layer normalization, multi-head attention, KV-cache support, and grouped-query attention (GQA). |
| `Transformer_arquitectures.py` | GPT and GQA-GPT model definitions, including cache reset support. |
| `help_functions.py` | Greedy text generation with and without cached decoding. |
| `kv_cache/` | Notebook and benchmark image for KV-cache experiments. |
| `gqa/` | GQA notebook, KV-cache savings visualization, and GQA-specific notes. |

## Techniques

### KV cache

During autoregressive generation, the key and value projections of prior
tokens do not change. The cache-aware attention implementation stores them and
only computes projections for the newly generated token. This avoids repeating
work at every decoding step.

For a generated sequence of length `T`, naive token-by-token decoding
reprocesses an increasingly long prefix at each step. Its attention work is
therefore proportional to `sum(t^2, t=1..T) = O(T^3)`. With a KV cache, only
the new query attends to the stored prefix at each step, reducing this to
`sum(t, t=1..T) = O(T^2)`. The trade-off is cache memory that grows linearly
with the context length.

For `L` layers, batch size `B`, context length `T`, `H_kv` key/value heads,
head dimension `d_h`, and `s` bytes per element, the total cache size is:

```text
KV-cache bytes = 2 × L × B × T × H_kv × d_h × s
```

The leading `2` represents the key and value tensors.

Use `generate_text_simple_cached` from `help_functions.py` with a model that
supports `use_cache=True`. The helper resets the cache before processing a new
prompt.

### Grouped-query attention (GQA)

GQA uses fewer key/value heads than query heads. Query heads share the
key/value projections within groups, which reduces the memory required by the
KV cache while retaining multiple query heads. `GroupedQueryAttention` and
`GQAGPTModel` provide the corresponding attention layer and model.

If MHA has `H` attention heads and GQA uses `G` key/value groups, their cache
sizes have the following relationship:

```text
GQA cache / MHA cache = G / H
GQA saving vs MHA     = 1 − G / H
```

For example, `H = 24` and `G = 6` uses 25% of the MHA cache and saves 75%.
Multi-query attention (MQA) is the special case `G = 1`, saving 95.8% for
24 heads.

### GQA cache-memory comparison

The chart below uses the configuration in `gqa/plot_kv_cache_savings.py`:
24 query heads, 48 layers, an 8,192-token context, batch size 1, and bf16
values. Savings are relative to the KV cache of MHA; MHA itself is the
`G = H = 24` baseline.

![GQA KV-cache savings relative to MHA](gqa/kv_cache_savings_vs_mha.svg)

## Running the examples

Install the project dependencies, including PyTorch, then run the notebooks
from Jupyter:

```bash
jupyter notebook "Optimization Techniques/kv_cache/KV_cache_test.ipynb"
jupyter notebook "Optimization Techniques/gqa/gqa_test.ipynb"
```

To recreate the GQA cache-saving chart:

```bash
python3 "Optimization Techniques/gqa/plot_kv_cache_savings.py"
```

The script writes `kv_cache_savings_vs_mha.png` to the `gqa` directory.

## Notes

- Cached decoding is intended for sequential generation. Call
  `model.reset_kv_cache()` before starting a new prompt when using the model
  APIs directly.
- The cache is bounded by the model's configured `context_length`.
- The notebooks are the best starting point for the benchmark setup and model
  configuration used in these experiments.
