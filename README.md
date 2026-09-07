# LLM Mini Lab

Educational project for building, training, optimizing, and evaluating small
GPT language models from scratch.

## Repository layout

```text
LLM_mini_lab/
├── src/llm_mini_lab/        # Reusable Python package
│   ├── attention.py         # Base attention, LayerNorm, GELU, and feed-forward
│   ├── model.py             # Base GPT architecture (checkpoint compatible)
│   ├── pretraining.py       # Datasets, training loop, and CLI
│   ├── helpers.py           # Generation and inference benchmarking
│   └── optimization/        # KV cache, GQA, and MoE variants
├── notebooks/
│   ├── pretraining/         # Small and full pretraining runs
│   ├── post_training/       # Supervised fine-tuning
│   ├── inference/           # Checkpoint loading and generation
│   ├── optimization/        # KV-cache, GQA, and MoE experiments
│   └── benchmarks/          # ARC-Easy, HellaSwag, and comparison notebooks
├── scripts/                 # Export and visualization utilities
├── data/                    # Small, versioned input datasets
├── docs/                    # Technical notes and generated figures
├── results/benchmarks/      # Versioned benchmark metrics by model
├── checkpoints/             # Local model weights; ignored by Git
└── tests/                   # Automated checks
```

## Installation

Python 3.10 or newer is recommended. Install the package from the repository
root so scripts and notebooks can use stable imports:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[training,benchmarks,huggingface,dev]"
```

The notebooks also locate `src/` automatically when they are executed from
inside this repository.

## Common commands

```bash
# Quick pretraining run with The Verdict
python -m llm_mini_lab.pretraining

# FineWeb-Edu streaming run
python -m llm_mini_lab.pretraining --dataset fineweb --max_docs 5000 --epochs 1

# Convert a checkpoint to Hugging Face format
python scripts/convert_to_huggingface.py \
  --checkpoint checkpoints/model.pth \
  --output_dir checkpoints/hf-gpt2-124m

# Recreate the GQA cache-memory visualization
python scripts/plot_kv_cache_savings.py
```

## Project status

The base GPT-2-small training pipeline works end to end. The optimization
experiments cover cached autoregressive decoding, grouped-query attention,
and sparse mixture-of-experts blocks. ARC-Easy and HellaSwag notebooks store
their metrics under `results/benchmarks/<model>/`.

See [foundation-model notes](docs/foundation_model.md) and
[optimization notes](docs/optimization.md) for details.

![Training and validation loss](docs/images/loss_curve.png)

## Artifact policy

Model weights and downloaded checkpoints belong in `checkpoints/` and are not
committed to Git. Small benchmark metrics and documentation figures are kept
under `results/` and `docs/images/` so experiments remain reproducible.
