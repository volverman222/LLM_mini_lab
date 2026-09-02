"""Plot KV-cache memory savings obtained by GQA relative to MHA.

Run from the repository root:
    python3 "Optimization Techniques/gqa/plot_kv_cache_savings.py"
"""

from pathlib import Path

import matplotlib.pyplot as plt


N_HEADS = 24
EMB_DIM = 2048
N_LAYERS = 48
BATCH_SIZE = 1
CONTEXT_LENGTH = 8192
DTYPE_BYTES = 2  # bfloat16 / float16


def kv_cache_memory_bytes(n_kv_groups):
    """Memory for K and V caches across every Transformer layer."""
    head_dim = EMB_DIM // N_HEADS
    return (
        2 * N_LAYERS * BATCH_SIZE * CONTEXT_LENGTH
        * n_kv_groups * head_dim * DTYPE_BYTES
    )


def savings_vs_mha_percent(n_kv_groups):
    """Percentage of KV-cache memory saved relative to MHA."""
    mha_bytes = kv_cache_memory_bytes(N_HEADS)
    gqa_bytes = kv_cache_memory_bytes(n_kv_groups)
    return 100 * (1 - gqa_bytes / mha_bytes)


def main():
    groups = [group for group in range(1, N_HEADS + 1)
              if N_HEADS % group == 0]
    savings = [savings_vs_mha_percent(group) for group in groups]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(groups, savings, marker="o", color="#1f77b4")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(groups)
    axis.set_ylim(-2, 100)
    axis.grid(True, alpha=0.35)
    axis.set_xlabel("n_kv_groups")
    axis.set_ylabel("KV-cache savings vs MHA (%)")
    axis.set_title(
        "GQA KV-cache savings relative to MHA\n"
        f"(n_heads={N_HEADS}, emb_dim={EMB_DIM}, n_layers={N_LAYERS}, "
        f"batch={BATCH_SIZE}, context={CONTEXT_LENGTH}, bf16)"
    )
    axis.annotate(
        "MQA: maximum saving", xy=(1, savings[0]), xytext=(4, 88),
        arrowprops={"arrowstyle": "->"},
    )
    axis.annotate(
        "MHA: no saving", xy=(N_HEADS, savings[-1]), xytext=(15, 12),
        arrowprops={"arrowstyle": "->"},
    )
    figure.tight_layout()

    output_path = Path(__file__).with_name("kv_cache_savings_vs_mha.png")
    figure.savefig(output_path, dpi=160)
    print(f"Saved graph: {output_path}")
    print(f"MQA (1 group): {savings[0]:.1f}% savings")
    print(f"MHA ({N_HEADS} groups): {savings[-1]:.1f}% savings")


if __name__ == "__main__":
    main()
