import torch
import torch.nn as nn



class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift




class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

        ####################################################
        # NEW
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        ####################################################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # NEW
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        ####################################################
        # NEW
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            self.ptr_current_pos += num_tokens_Q
        ####################################################
        # Original mask truncated to the number of tokens and converted to boolean
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    ####################################################
    # NEW
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
    ####################################################


### GQA codigo ###
class GroupedQueryAttention(nn.Module):
    """Causal grouped-query attention with an optional KV cache.

    Query heads are kept independent while groups of query heads share one
    key/value head. ``num_heads`` must be divisible by ``num_kv_groups``.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 num_kv_groups, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups")

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=qkv_bias)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Shape stored in cache: (batch, kv_groups, tokens, head_dim)
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x).view(
            b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = self.W_key(x).view(
            b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = self.W_value(x).view(
            b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
            keys_base, values_base = self.cache_k, self.cache_v
        else:
            keys_base, values_base = keys_new, values_new

        # Reuse each K/V group for its associated query-head group.
        keys = keys_base.repeat_interleave(self.group_size, dim=1)
        values = values_base.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        num_tokens_q = queries.shape[-2]
        num_tokens_k = keys.shape[-2]
        device = queries.device
        if use_cache:
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_q,
                device=device,
            )
            self.ptr_current_pos += num_tokens_q
        else:
            q_positions = torch.arange(num_tokens_q, device=device)

        k_positions = torch.arange(num_tokens_k, device=device)
        causal_mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)
        attn_scores.masked_fill_(causal_mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
### GQA codigo ###
