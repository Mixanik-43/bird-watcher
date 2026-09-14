"""Reference modules; implement their equivalents in the exercise notebooks."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class BirdClassifier(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, n_classes)
        self.set_stage("head")

    def set_stage(self, stage):
        if stage not in {"head", "last_block"}:
            raise ValueError(stage)
        self.stage = stage
        self.backbone.requires_grad_(False)
        if stage == "last_block":
            self.backbone.layer4.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm must not silently change running statistics.
        self.backbone.eval()
        if getattr(self, "stage", None) == "last_block":
            self.backbone.layer4.train(mode)
        return self

    def forward(self, images):
        return self.head(self.backbone(images))


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.A, std=0.02)

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merged(self):
        layer = nn.Linear(self.base.in_features, self.base.out_features,
                          bias=self.base.bias is not None).to(self.base.weight)
        with torch.no_grad():
            layer.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
            if layer.bias is not None:
                layer.bias.copy_(self.base.bias)
        return layer
