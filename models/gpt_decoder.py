"""
Custom GPT-style decoder.

This is meant to be swapped out for the decoder from your LLM-from-scratch
project — copy that implementation's blocks in here and extend them with
the optional cross-attention layer below. Keeping it self-contained here
for now so the fusion pipeline is testable end-to-end before you wire in
your own weights/architecture.
"""

import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, causal_mask=None):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B, heads, T, head_dim)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class CrossAttention(nn.Module):
    """Text tokens (queries) attend to image features (keys/values)."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, image_features):
        b, t, d = x.shape
        _, n_img, _ = image_features.shape

        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(image_features).reshape(b, n_img, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, use_cross_attn: bool = False):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, n_heads)

        if use_cross_attn:
            self.ln_cross = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(dim, n_heads)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, causal_mask=None, image_features=None):
        x = x + self.self_attn(self.ln1(x), causal_mask)

        if self.use_cross_attn:
            assert image_features is not None, "cross-attention block requires image_features"
            x = x + self.cross_attn(self.ln_cross(x), image_features)

        x = x + self.mlp(self.ln2(x))
        return x


class GPTDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        max_seq_len: int = 64,
        fusion: str = "concat",  # "concat" or "cross_attn"
    ):
        super().__init__()
        assert fusion in ("concat", "cross_attn")
        self.fusion = fusion
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        use_cross = fusion == "cross_attn"
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, n_heads, use_cross_attn=use_cross) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, image_features: torch.Tensor = None):
        """
        Args:
            token_ids: (B, T) question/answer token ids
            image_features: (B, n_regions, dim) from VisionEncoder
        """
        b, t = token_ids.shape
        pos = torch.arange(t, device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb(pos)

        if self.fusion == "concat" and image_features is not None:
            x = torch.cat([image_features, x], dim=1)
            t = x.shape[1]

        causal_mask = torch.tril(torch.ones(t, t, device=token_ids.device)).view(1, 1, t, t)

        for block in self.blocks:
            if self.fusion == "cross_attn":
                x = block(x, causal_mask, image_features=image_features)
            else:
                x = block(x, causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        if self.fusion == "concat" and image_features is not None:
            # drop the image-token positions, keep only text positions for loss
            n_img = image_features.shape[1]
            logits = logits[:, n_img:, :]

        return logits
