"""
Vision encoder used as a feature extractor for VQA fusion.

Supports three modes via `freeze`:
    freeze=True         — entire backbone frozen (fastest, but features
                           stay fixed on ImageNet statistics)
    freeze=False        — entire backbone trainable (most flexible, but
                           slowest and most prone to overfitting on a
                           small dataset)
    freeze="partial"    — only the final residual block (layer4) is
                           trainable; everything before stays frozen.
                           This is the recommended middle ground: lets
                           the network adapt its highest-level features
                           to CLEVR's synthetic rendering style, while
                           keeping low-level features (edges, textures)
                           fixed at their well-trained ImageNet values.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", out_dim: int = 256, freeze="partial"):
        super().__init__()

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Keep named children so we can selectively freeze/unfreeze
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.freeze_mode = freeze
        self._apply_freezing()

        self.projection = nn.Linear(backbone_dim, out_dim)

    def _apply_freezing(self):
        if self.freeze_mode is True:
            for p in self.parameters():
                p.requires_grad = False
        elif self.freeze_mode is False:
            for p in self.parameters():
                p.requires_grad = True
        elif self.freeze_mode == "partial":
            for module in [self.stem, self.layer1, self.layer2, self.layer3]:
                for p in module.parameters():
                    p.requires_grad = False
            for p in self.layer4.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze mode: {self.freeze_mode}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, num_regions, out_dim)
        """
        # stem/layer1-3: only trainable in fully-unfrozen mode
        early_layers_grad = (self.freeze_mode is False) and self.training
        with torch.set_grad_enabled(early_layers_grad):
            x = self.stem(images)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        # layer4: trainable in both "partial" and fully-unfrozen modes
        layer4_grad = (self.freeze_mode in (False, "partial")) and self.training
        with torch.set_grad_enabled(layer4_grad):
            feat_map = self.layer4(x)

        b, c, h, w = feat_map.shape
        feat_map = feat_map.flatten(2).transpose(1, 2)  # (B, h*w, 512)
        return self.projection(feat_map)
