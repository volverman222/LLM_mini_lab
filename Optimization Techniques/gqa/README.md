# GQA

The shared GQA implementation is in `../attention_mechanisms.py`, alongside
the regular multi-head attention and KV-cache variants. This directory keeps
the GQA-specific visualization.

- `plot_kv_cache_savings.py`: produces the correct KV-cache savings graph,
  using MHA (`n_kv_groups == n_heads`) as the zero-saving baseline.
- `gqa_test.ipynb`: tests the GQA model and its cached decoding.

Run the graph script from the repository root:

```bash
python3 "Optimization Techniques/gqa/plot_kv_cache_savings.py"
```

It creates `kv_cache_savings_vs_mha.png` in this directory.
