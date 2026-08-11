"""
Top-level VQA model: wires the vision encoder into the GPT decoder using
whichever fusion strategy is selected.

fusion="none" is a text-only control baseline — the vision encoder is not
even called, guaranteeing zero image information reaches the decoder. Used
to verify the "concat"/"cross_attn" models are genuinely using the image,
not just exploiting language-only patterns in the question phrasing.
"""

import torch.nn as nn

from models.vision_encoder import VisionEncoder
from models.gpt_decoder import GPTDecoder


class VQAFusionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        max_seq_len: int = 64,
        fusion: str = "concat",
        vision_backbone: str = "resnet18",
        freeze_vision: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion = fusion

        if fusion != "none":
            self.vision_encoder = VisionEncoder(
                backbone=vision_backbone, out_dim=dim, freeze=freeze_vision
            )
        else:
            self.vision_encoder = None

        self.decoder = GPTDecoder(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            fusion=fusion,
            dropout=dropout,
        )

    def forward(self, images, token_ids):
        if self.vision_encoder is not None:
            image_features = self.vision_encoder(images)
        else:
            image_features = None  # "none" mode: text-only, image is never even encoded
        return self.decoder(token_ids, image_features=image_features)
