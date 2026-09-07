import pytest

torch = pytest.importorskip("torch")

from llm_mini_lab.model import GPTModel
from llm_mini_lab.optimization.model import CachedGPTModel, GQAGPTModel
from llm_mini_lab.optimization.moe import Block


BASE_CONFIG = {
    "vocab_size": 128,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def test_base_model_output_shape():
    model = GPTModel(BASE_CONFIG)
    tokens = torch.randint(0, BASE_CONFIG["vocab_size"], (2, 8))

    assert model(tokens).shape == (2, 8, BASE_CONFIG["vocab_size"])


def test_cached_model_matches_full_decoding():
    torch.manual_seed(7)
    model = CachedGPTModel(BASE_CONFIG).eval()
    tokens = torch.randint(0, BASE_CONFIG["vocab_size"], (1, 8))

    full_logits = model(tokens, use_cache=False)
    model.reset_kv_cache()
    cached_logits = torch.cat(
        [model(tokens[:, index:index + 1], use_cache=True)
         for index in range(tokens.shape[1])],
        dim=1,
    )

    torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-5)


def test_gqa_model_output_shape():
    config = {**BASE_CONFIG, "n_kv_groups": 2}
    model = GQAGPTModel(config)
    tokens = torch.randint(0, config["vocab_size"], (2, 8))

    assert model(tokens).shape == (2, 8, config["vocab_size"])


def test_moe_block_preserves_shape():
    block = Block(
        n_embed=32,
        n_head=4,
        num_experts=4,
        top_k=2,
        context_length=16,
    )
    inputs = torch.randn(2, 8, 32)

    assert block(inputs).shape == inputs.shape
