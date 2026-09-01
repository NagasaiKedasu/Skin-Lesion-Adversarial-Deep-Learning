import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# ─────────────────────────────────────────────────────────────────────────────
# Helper: replace final classification layer
# ─────────────────────────────────────────────────────────────────────────────
def _make_classifier_head(in_features: int, num_classes: int,
                           dropout: float = 0.4) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. EfficientNet-B4
# ─────────────────────────────────────────────────────────────────────────────
class EfficientNetB4(nn.Module):
    """
    EfficientNet-B4 fine-tuned for 7-class skin lesion classification.
    Input size: 380 × 380 (native) or 224 × 224 (acceptable).

    Recommended by: Tan & Le (2019); achieves SOTA on ISIC challenges.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4,
                 pretrained: bool = True):
        super().__init__()
        weights = tvm.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.efficientnet_b4(weights=weights)

        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        in_features     = backbone.classifier[1].in_features
        self.classifier = _make_classifier_head(in_features, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_feature_extractor(self) -> nn.Module:
        return self.features

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int|None = None):
        """
        Unfreeze backbone.
        If unfreeze_last_n_blocks is set, only unfreeze the last N feature blocks
        (discriminative fine-tuning / gradual unfreezing).
        """
        if unfreeze_last_n_blocks is None:
            for p in self.features.parameters():
                p.requires_grad = True
        else:
            # EfficientNet-B4 features is a Sequential of 9 blocks (0-8)
            total = len(self.features)
            for i, block in enumerate(self.features):
                if i >= total - unfreeze_last_n_blocks:
                    for p in block.parameters():
                        p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 2. ResNet-50 with CBAM Attention
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """CBAM channel attention gate."""
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return torch.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial attention gate."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg  = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ResNet50CBAM(nn.Module):
    """
    ResNet-50 with Convolutional Block Attention Module (CBAM) appended
    after the final residual stage for improved spatial focus.

    Reference: Woo et al. (2018) - "CBAM: Convolutional Block Attention Module"
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.5,
                 pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        in_ch = 2048
        self.channel_att = ChannelAttention(in_ch)
        self.spatial_att = SpatialAttention()
        self.avgpool      = nn.AdaptiveAvgPool2d(1)
        self.classifier   = _make_classifier_head(in_ch, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 3. DenseNet-169
# ─────────────────────────────────────────────────────────────────────────────
class DenseNet169(nn.Module):
    """
    DenseNet-169 for skin lesion classification.
    Dense connectivity → strong gradient flow, feature reuse.

    Reference: Huang et al. (2017) - "Densely Connected Convolutional Networks"
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4,
                 pretrained: bool = True):
        super().__init__()
        weights = tvm.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.densenet169(weights=weights)

        self.features   = backbone.features
        in_features     = backbone.classifier.in_features
        self.classifier = _make_classifier_head(in_features, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Vision Transformer (ViT-B/16)
# ─────────────────────────────────────────────────────────────────────────────
class VisionTransformerB16(nn.Module):
    """
    ViT-B/16 fine-tuned for skin lesion classification.
    Uses global self-attention → captures long-range spatial dependencies
    missed by local CNN kernels.

    Reference: Dosovitskiy et al. (2020) - "An Image is Worth 16x16 Words"
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.vit_b_16(weights=weights)

        # Replace classification head
        in_features     = backbone.heads.head.in_features
        backbone.heads   = nn.Identity()
        self.backbone    = backbone
        self.classifier  = _make_classifier_head(in_features, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x)

    def freeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if "encoder" in name:
                p.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_layers: int = None):
        if unfreeze_last_n_layers is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            # ViT-B/16 has 12 transformer encoder layers
            encoder_layers = list(self.backbone.encoder.layers.children())
            n = len(encoder_layers)
            for i, layer in enumerate(encoder_layers):
                if i >= n - unfreeze_last_n_layers:
                    for p in layer.parameters():
                        p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Modal Late Fusion: Image + Metadata
# ─────────────────────────────────────────────────────────────────────────────
class MetadataMLP(nn.Module):
    """Small MLP to encode tabular metadata (age, sex, localization)."""

    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalFusion(nn.Module):
    """
    Late-fusion multi-modal model.

    Architecture:
      ImageBranch → GlobalAvgPool → fc(2048→512) → image_embedding
      MetaBranch  → MLP → meta_embedding
      Concat → Classifier head → 7 classes

    Clinical motivation: age, sex, and lesion location are strong priors
    in dermatology (e.g., melanoma on trunk in young females vs. elderly males).
    """

    def __init__(self,
                 metadata_dim: int,
                 num_classes: int  = 7,
                 image_backbone: str = "efficientnet_b4",
                 meta_embed_dim: int = 64,
                 dropout: float      = 0.4,
                 pretrained: bool    = True):
        super().__init__()

        # Image branch
        if image_backbone == "efficientnet_b4":
            weights = tvm.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tvm.efficientnet_b4(weights=weights)
            self.image_features = backbone.features
            self.image_pool     = backbone.avgpool
            img_feat_dim        = backbone.classifier[1].in_features
        elif image_backbone == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = tvm.resnet50(weights=weights)
            self.image_features = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            )
            self.image_pool = nn.AdaptiveAvgPool2d(1)
            img_feat_dim    = 2048
        else:
            raise ValueError(f"Unsupported image_backbone: {image_backbone}")

        self.image_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(img_feat_dim, 512),
            nn.ReLU(inplace=True),
        )

        # Metadata branch
        self.meta_mlp = MetadataMLP(metadata_dim, meta_embed_dim, dropout)

        # Joint classifier
        joint_dim = 512 + meta_embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(joint_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor,
                metadata: torch.Tensor) -> torch.Tensor:
        # Image branch
        img = self.image_features(image)
        img = self.image_pool(img)
        img = torch.flatten(img, 1)
        img = self.image_fc(img)

        # Metadata branch
        meta = self.meta_mlp(metadata)

        # Fusion
        fused = torch.cat([img, meta], dim=1)
        return self.classifier(fused)

    def freeze_backbone(self):
        for p in self.image_features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.image_features.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    model_name: str,
    num_classes: int    = 7,
    pretrained: bool    = True,
    dropout: float      = None,
    metadata_dim: int   = None,
) -> nn.Module:
    """
    Factory function.

    Args:
        model_name:   One of 'efficientnet_b4', 'resnet50_cbam',
                      'densenet169', 'vit_b16', 'multimodal'
        num_classes:  Number of output classes (default 7)
        pretrained:   Use ImageNet weights
        dropout:      Override default dropout rate
        metadata_dim: Required for 'multimodal' model

    Returns: nn.Module ready for training
    """
    defaults = {
        "efficientnet_b4": 0.4,
        "resnet50_cbam":   0.5,
        "densenet169":     0.4,
        "vit_b16":         0.1,
        "multimodal":      0.4,
    }
    dr = dropout if dropout is not None else defaults.get(model_name, 0.4)

    if model_name == "efficientnet_b4":
        return EfficientNetB4(num_classes, dr, pretrained)
    elif model_name == "resnet50_cbam":
        return ResNet50CBAM(num_classes, dr, pretrained)
    elif model_name == "densenet169":
        return DenseNet169(num_classes, dr, pretrained)
    elif model_name == "vit_b16":
        return VisionTransformerB16(num_classes, dr, pretrained)
    elif model_name == "multimodal":
        if metadata_dim is None:
            raise ValueError("metadata_dim required for multimodal model")
        return MultiModalFusion(metadata_dim, num_classes, dropout=dr,
                                pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from efficientnet_b4, resnet50_cbam, "
                         f"densenet169, vit_b16, multimodal")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter group builder for differential learning rates
# ─────────────────────────────────────────────────────────────────────────────
def get_param_groups(model: nn.Module, base_lr: float,
                     backbone_lr_multiplier: float = 0.1) -> list:
    """
    Returns optimizer parameter groups with different LRs for backbone vs head.
    Discriminative fine-tuning: slower learning rate for pre-trained layers.

    Args:
        base_lr:                Head learning rate
        backbone_lr_multiplier: Backbone LR = base_lr * this value
    """
    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["features", "backbone", "image_features"]):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {"params": head_params,     "lr": base_lr},
        {"params": backbone_params, "lr": base_lr * backbone_lr_multiplier},
    ]
