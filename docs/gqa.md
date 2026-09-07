# Grouped-query attention

The implementation lives in `llm_mini_lab.optimization.attention`, and the
corresponding model is `llm_mini_lab.optimization.model.GQAGPTModel`.

The experiment notebook is located at
`notebooks/optimization/gqa/gqa_test.ipynb`. It verifies that cached decoding
matches full decoding and demonstrates the relationship between query heads
and key/value groups.

Regenerate the cache-memory chart from the repository root:

```bash
python scripts/plot_kv_cache_savings.py
```

The script writes `docs/images/kv_cache_savings_vs_mha.png`. The versioned SVG
in the same directory is retained for documentation.
