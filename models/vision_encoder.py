"""
Frozen vision encoder used as a feature extractor for VQA fusion.

We do NOT train this from scratch — we use a pretrained CNN (ResNet-18/34)
and extract intermediate spatial features, which are then projected into
the GPT decoder's embedding space. Freezing keeps compute low and lets us
focus the "from scratch" novelty entirely on the language/decoder side.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", out_dim: int = 256, freeze: bool = True):
        """
        Args:
            backbone: torchvision backbone name ("resnet18" is a good default
                for Colab T4 — fast, small, plenty for CLEVR-style images).
            out_dim: dimension to project image features into, matching the
                GPT decoder's token embedding dimension.
            freeze: if True, backbone weights are not updated during training.
        """
        super().__init__()

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Drop avgpool + fc, keep spatial feature map: (B, 512, H, W)
            self.backbone = nn.Sequential(*list(net.children())[:-2])
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.freeze = freeze
        self.projection = nn.Linear(backbone_dim, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, num_regions, out_dim) — a sequence of region
                feature vectors, ready to be fused with text tokens.
        """
        if self.freeze:
            with torch.no_grad():
                feat_map = self.backbone(images)          # (B, 512, h, w)
        else:
            feat_map = self.backbone(images)

        b, c, h, w = feat_map.shape
        feat_map = feat_map.flatten(2).transpose(1, 2)     # (B, h*w, 512)
        return self.projection(feat_map)                   # (B, h*w, out_dim)
