# Foundation model

The base implementation pretrains a GPT-2-small-style model from scratch in
PyTorch, following *Build a Large Language Model From Scratch* by Sebastian
Raschka.

## Model configuration

| Parameter | Value |
| --- | --- |
| `vocab_size` | 50,257 (GPT-2 tokenizer) |
| `context_length` | 256 by default; configurable in notebooks |
| `emb_dim` | 768 |
| `n_heads` | 12 |
| `n_layers` | 12 |
| `drop_rate` | 0.0 during pretraining |
| `qkv_bias` | `False` |

## Code and notebooks

- `src/llm_mini_lab/attention.py` contains the base attention and feed-forward
  components.
- `src/llm_mini_lab/model.py` defines `TransformerBlock` and `GPTModel`.
- `src/llm_mini_lab/pretraining.py` contains datasets, dataloaders, loss
  calculation, the training loop, and the command-line entry point.
- `src/llm_mini_lab/helpers.py` provides text generation and speed benchmarks.
- `notebooks/pretraining/small/` contains the smaller Colab training runs.
- `notebooks/pretraining/full/` contains external-GPU training runs.
- `notebooks/inference/load_model.ipynb` loads a local checkpoint and generates
  text.

## Running pretraining

Install the project with `pip install -e .`, then run:

```bash
python -m llm_mini_lab.pretraining
python -m llm_mini_lab.pretraining --dataset fineweb --max_docs 5000 --epochs 1
python -m llm_mini_lab.pretraining --epochs 5 --device cpu
```

Local weights should be stored in `checkpoints/`, which is ignored by Git.

## Hugging Face export

```bash
pip install -e ".[huggingface]"
python scripts/convert_to_huggingface.py \
  --checkpoint checkpoints/model.pth \
  --output_dir checkpoints/hf-gpt2-124m
```

The converter writes a standard `GPT2LMHeadModel` and compares its logits with
the custom model as a smoke test.

## Current result

One recorded FineWeb-Edu run processed approximately 5.2 million tokens. Its
training loss decreased from 8.41 to 6.26 and validation loss from 10.15 to
5.81. The run is still under-trained and is intended as an educational
baseline.

![Training and validation loss](images/loss_curve.png)
