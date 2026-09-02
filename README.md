# LLM Mini Lab

Personal lab for building, training and experimenting with small language
models from scratch.

## Repository layout

```
LLM_mini_lab
├── Foundation Model/     # GPT-2 small (124M) pretraining from scratch
│   ├── pretraining.py               # datasets, training loop, CLI
│   ├── Transformer_arquitectures.py # GPT model architecture
│   ├── attention_mechanisms.py      # multi-head causal attention, LayerNorm, GELU
│   ├── helper_functions.py          # generation with temperature / top-k + benchmark
│   ├── Pre_train_LLM_124M .ipynb    # Colab notebook for FineWeb-Edu training
│   ├── load_model.ipynb             # load a checkpoint, run inference, benchmark tokens/s
│   ├── Training_data.md             # logged losses from a real run
│   ├── loss_curve.png               # training/validation loss curve
│   └── README.md                    # project details and current state
└── Optimization Techniques/
    ├── attention_mechanisms.py      # shared MHA, GQA and KV-cache attention
    ├── Transformer_arquitectures.py # shared GPT and GQA-GPT architectures
    ├── help_functions.py            # shared generation helpers
    ├── gqa/                         # GQA experiments and visualization
    └── kv_cache/                    # KV-cache and GQA-cache notebooks
```

## Status

- **In progress.** The core training pipeline works end-to-end (see
  [`Foundation Model/README.md`](Foundation%20Model/README.md) for details).
- The model is still under-trained: it needs more data/epochs to converge.

## Loss curve

Training and validation loss of the latest run (1 epoch on FineWeb-Edu,
~5.2M tokens):

![Loss curve](Foundation%20Model/loss_curve.png)

## Requirements

- Python 3.9+
- PyTorch
- `numpy`
- `tiktoken`
- `datasets` (only for the FineWeb-Edu dataset)
