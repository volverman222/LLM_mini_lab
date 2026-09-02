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

Use `generate_text_simple_cached` from `help_functions.py` with a model that
supports `use_cache=True`. The helper resets the cache before processing a new
prompt.

### Grouped-query attention (GQA)

GQA uses fewer key/value heads than query heads. Query heads share the
key/value projections within groups, which reduces the memory required by the
KV cache while retaining multiple query heads. `GroupedQueryAttention` and
`GQAGPTModel` provide the corresponding attention layer and model.

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
