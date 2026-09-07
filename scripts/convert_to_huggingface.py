"""Convert the custom GPTModel checkpoint to a Hugging Face GPT-2 model.

Usage (after ``pip install -e .``):
  python scripts/convert_to_huggingface.py \
    --checkpoint checkpoints/gpt2-pretrained.pth \
    --output_dir checkpoints/hf-gpt2-124m

The script also compares logits from the original model and the converted
model on a short text.  A small numerical difference is expected because
Hugging Face implements GELU and attention with different kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


from llm_mini_lab.model import GPTModel


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config")
    return checkpoint, None


def copy_linear(dst, src_weight, src_bias=None):
    # GPT-2's Conv1D stores [in_features, out_features], unlike Linear.
    dst.weight.data.copy_(src_weight.T)
    if dst.bias is not None and src_bias is not None:
        dst.bias.data.copy_(src_bias)


def convert(state, cfg):
    hf_cfg = GPT2Config(
        vocab_size=cfg["vocab_size"],
        n_positions=cfg["context_length"],
        n_ctx=cfg["context_length"],
        n_embd=cfg["emb_dim"],
        n_layer=cfg["n_layers"],
        n_head=cfg["n_heads"],
        n_inner=4 * cfg["emb_dim"],
        activation_function="gelu_new",
        resid_pdrop=cfg["drop_rate"],
        embd_pdrop=cfg["drop_rate"],
        attn_pdrop=cfg["drop_rate"],
        add_cross_attention=False,
        tie_word_embeddings=False,
        use_cache=True,
    )
    model = GPT2LMHeadModel(hf_cfg)
    model.transformer.wte.weight.data.copy_(state["tok_emb.weight"])
    model.transformer.wpe.weight.data.copy_(state["pos_emb.weight"])

    for i in range(cfg["n_layers"]):
        src = f"trf_blocks.{i}"
        block = model.transformer.h[i]
        block.ln_1.weight.data.copy_(state[f"{src}.norm1.scale"])
        block.ln_1.bias.data.copy_(state[f"{src}.norm1.shift"])
        block.ln_2.weight.data.copy_(state[f"{src}.norm2.scale"])
        block.ln_2.bias.data.copy_(state[f"{src}.norm2.shift"])

        q = state[f"{src}.att.W_query.weight"]
        k = state[f"{src}.att.W_key.weight"]
        v = state[f"{src}.att.W_value.weight"]
        block.attn.c_attn.weight.data.copy_(torch.cat((q, k, v), dim=0).T)
        block.attn.c_proj.weight.data.copy_(state[f"{src}.att.out_proj.weight"].T)
        block.attn.c_proj.bias.data.copy_(state[f"{src}.att.out_proj.bias"])

        copy_linear(block.mlp.c_fc, state[f"{src}.ff.layers.0.weight"],
                    state[f"{src}.ff.layers.0.bias"])
        copy_linear(block.mlp.c_proj, state[f"{src}.ff.layers.2.weight"],
                    state[f"{src}.ff.layers.2.bias"])

    model.transformer.ln_f.weight.data.copy_(state["final_norm.scale"])
    model.transformer.ln_f.bias.data.copy_(state["final_norm.shift"])
    model.lm_head.weight.data.copy_(state["out_head.weight"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--context_length", type=int, default=None,
                        help="override context length for a plain state_dict")
    args = parser.parse_args()

    state, saved_cfg = load_checkpoint(args.checkpoint)
    cfg = saved_cfg or {
        "vocab_size": 50257, "context_length": 256, "emb_dim": 768,
        "n_heads": 12, "n_layers": 12, "drop_rate": 0.0, "qkv_bias": False,
    }
    if args.context_length is not None:
        cfg["context_length"] = args.context_length
    model = convert(state, cfg).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = cfg["context_length"]
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    (args.output_dir / "conversion_config.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    text = "Every effort moves you"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        hf_logits = model(**inputs).logits
    original = GPTModel(cfg).eval()
    original.load_state_dict(state)
    with torch.inference_mode():
        original_logits = original(inputs["input_ids"])
    diff = (hf_logits - original_logits).abs().max().item()
    print(f"Saved Hugging Face model to: {args.output_dir}")
    print(f"Maximum logit difference on smoke test: {diff:.6g}")


if __name__ == "__main__":
    main()
