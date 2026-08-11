"""
Custom GPT-style decoder.

Supports three fusion modes:
    "concat"     — image features prepended to the text token sequence
    "cross_attn" — text tokens cross-attend to image features at every layer
    "none"       — TEXT-ONLY control baseline. No image information is used
                   at all. This exists purely as a diagnostic: if "none"
                   performs similarly to "concat"/"cross_attn", it means
                   the fusion mechanisms aren't actually contributing
                   useful visual information, and the model is just
                   pattern-matching on question phrasing. This mirrors the
                   "language-only" baseline reported in the original VQA
                   paper (Antol et al., 2015).
"""

import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class CrossAttention(nn.Module):
    """Text tokens (queries) attend to image features (keys/values)."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, image_features):
        b, t, d = x.shape
        _, n_img, _ = image_features.shape

        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(image_features).reshape(b, n_img, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, use_cross_attn: bool = False, dropout: float = 0.1):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, n_heads, dropout=dropout)

        if use_cross_attn:
            self.ln_cross = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(dim, n_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
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
        fusion: str = "concat",  # "concat", "cross_attn", or "none"
        dropout: float = 0.1,
    ):
        super().__init__()
        assert fusion in ("concat", "cross_attn", "none")
        self.fusion = fusion
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.emb_dropout = nn.Dropout(dropout)

        use_cross = fusion == "cross_attn"
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, n_heads, use_cross_attn=use_cross, dropout=dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, image_features: torch.Tensor = None):
        """
        Args:
            token_ids: (B, T) question/answer token ids
            image_features: (B, n_regions, dim) from VisionEncoder.
                Ignored entirely when fusion == "none".
        """
        b, t = token_ids.shape
        pos = torch.arange(t, device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        use_image = self.fusion != "none" and image_features is not None

        if self.fusion == "concat" and use_image:
            x = torch.cat([image_features, x], dim=1)
            t = x.shape[1]

        causal_mask = torch.tril(torch.ones(t, t, device=token_ids.device)).view(1, 1, t, t)

        for block in self.blocks:
            if self.fusion == "cross_attn" and use_image:
                x = block(x, causal_mask, image_features=image_features)
            else:
                x = block(x, causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        if self.fusion == "concat" and use_image:
            n_img = image_features.shape[1]
            logits = logits[:, n_img:, :]

        return logits
