import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys


# ``attention_mechanisms.py`` lives one directory above this file.  Adding that
# directory to the import path lets this module be imported from a notebook or
# directly from the project root without changing the attention implementation.
ATTENTION_DIR = Path(__file__).resolve().parents[1]
if str(ATTENTION_DIR) not in sys.path:
    sys.path.insert(0, str(ATTENTION_DIR))

from attention_mechanisms import MultiHeadAttention

class Expert(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.linear(mh_output)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


#Changing the above to accomodate noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)


    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        if self.training:
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, dropout=0.0):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(n_embed, dropout=dropout) for _ in range(num_experts)]
        )
        self.top_k = top_k

    def forward(self, x):

        gating_output , indices = self.router(x)

        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing

        flat_x = x.reshape(-1, x.size(-1))
        flat_gating_output = gating_output.reshape(-1, gating_output.size(-1))

        # Process each expert in parallel (pyTorch manges internally)

        for i, expert in enumerate(self.experts):
            #Create a mask for the inputs where the currest expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.reshape(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class Block(nn.Module):
    """Transformer block with causal multi-head attention and sparse MoE.

    This is the MoE equivalent of a standard Transformer block.  It preserves
    the pre-norm residual structure: attention communicates between tokens and
    SparseMoE performs the token-wise computation.
    """

    def __init__(self, n_embed, n_head, num_experts, top_k, context_length,
                 dropout=0.0, qkv_bias=False):
        super().__init__()
        if n_embed % n_head != 0:
            raise ValueError("n_embed must be divisible by n_head")

        self.sa = MultiHeadAttention(
            d_in=n_embed,
            d_out=n_embed,
            context_length=context_length,
            dropout=dropout,
            num_heads=n_head,
            qkv_bias=qkv_bias,
        )
        self.smoe = SparseMoE(
            n_embed=n_embed,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, use_cache=False):
        x = x + self.sa(self.ln1(x), use_cache=use_cache)
        x = x + self.smoe(self.ln2(x))
        return x

    def reset_cache(self):
        """Clears the attention KV cache before starting a new generation."""
        self.sa.reset_cache()
