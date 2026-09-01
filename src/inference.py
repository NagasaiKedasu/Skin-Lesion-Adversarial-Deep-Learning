"""
Inference Helpers for the Demo App
===================================
Framework-agnostic glue between the trained checkpoints in `checkpoints/`
and any UI layer (Streamlit today, potentially a FastAPI service later —
see PROJECT_QA_DOCUMENT.md Q9). No Streamlit imports here on purpose, so
this module stays reusable from a notebook or a script.

Provides:
  - MODEL_REGISTRY / ENSEMBLE_MEMBERS : what models exist and how to run them
  - load_model()                      : build architecture + load best checkpoint
  - preprocess_image()                : deterministic eval-time transform
  - predict_single() / predict_ensemble()
  - load_metadata_schema() / build_metadata_vector() : patient metadata encoding
    for the MultiModal Fusion model, using the project's own feature
    engineering so the one-hot column order always matches the checkpoint
  - generate_explanations()           : GradCAM / GradCAM++ / Integrated
    Gradients overlays, reusing src.evaluate's implementations
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .models import build_model
from .dataset import (
    CLASS_NAMES, DATASET_MEAN, DATASET_STD,
    engineer_metadata, get_metadata_cols, get_transforms,
)
from .evaluate import GradCAM, GradCAMPlusPlus, IntegratedGradients

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DATA_ROOT = ROOT_DIR / "dataset"
METADATA_CSV = DATA_ROOT / "HAM10000_metadata.csv"

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# Clinical risk category per class — see PROJECT_QA_DOCUMENT.md Q5.
RISK_LEVEL = {
    "mel": "malignant", "bcc": "malignant",
    "akiec": "pre-malignant",
    "nv": "benign", "bkl": "benign", "vasc": "benign", "df": "benign",
}

# ─────────────────────────────────────────────────────────────────────────
# Model registry — one entry per checkpoint in checkpoints/
# ─────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "efficientnet_b4": {
        "label": "EfficientNet-B4",
        "file": "efficientnet_b4_best.pth",
        "image_size": 380,
        "multimodal": False,
        "supports_xai": True,
        "blurb": "Compound-scaled CNN (MBConv + Squeeze-Excitation). Best single model.",
    },
    "vit_b16": {
        "label": "ViT-B/16",
        "file": "vit_b16_best.pth",
        "image_size": 224,
        "multimodal": False,
        "supports_xai": False,
        "blurb": "Pure vision transformer — 16x16 patches, global self-attention.",
    },
    "resnet50_cbam": {
        "label": "ResNet-50 + CBAM",
        "file": "resnet50_cbam_best.pth",
        "image_size": 224,
        "multimodal": False,
        "supports_xai": True,
        "blurb": "Residual CNN with channel + spatial attention gates. Ensemble member.",
    },
    "densenet169": {
        "label": "DenseNet-169",
        "file": "densenet169_best.pth",
        "image_size": 224,
        "multimodal": False,
        "supports_xai": True,
        "blurb": "Densely-connected CNN — every layer reuses every earlier layer's features. Ensemble member.",
    },
    "multimodal": {
        "label": "MultiModal Fusion (Image + Metadata)",
        "file": "multimodal_fusion_best.pth",
        "image_size": 224,
        "multimodal": True,
        "supports_xai": True,
        "blurb": "EfficientNet-B4 image branch fused with an age / sex / site MLP.",
    },
}

ENSEMBLE_MEMBERS = ["efficientnet_b4", "resnet50_cbam", "densenet169"]
ENSEMBLE_IMAGE_SIZE = 224  # matches the shared test_loader used for soft-voting (notebook 03)
ENSEMBLE_LABEL = "Ensemble (soft-voting: EfficientNet-B4 + ResNet-50+CBAM + DenseNet-169)"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / MODEL_REGISTRY[model_name]["file"]


def checkpoint_exists(model_name: str) -> bool:
    return checkpoint_path(model_name).exists()


# ─────────────────────────────────────────────────────────────────────────
# Metadata schema — the frozen one-hot column order the multimodal
# checkpoint's meta_mlp was trained against.
# ─────────────────────────────────────────────────────────────────────────
def load_metadata_schema() -> tuple[list, list]:
    """
    Returns (meta_cols, localizations), engineered with the exact same
    function used at training time (src.dataset.engineer_metadata) so the
    one-hot column order matches the checkpoint's input dimension exactly.
    """
    df = pd.read_csv(METADATA_CSV)
    df = engineer_metadata(df)
    meta_cols = get_metadata_cols(df)
    localizations = sorted(df["localization"].dropna().unique().tolist())
    return meta_cols, localizations


def build_metadata_vector(age: float, sex: str, localization: str,
                           meta_cols: list) -> np.ndarray:
    """Builds a single feature row matching `meta_cols` order exactly."""
    values = {
        "age_norm":    float(age) / 85.0,
        "sex_male":    1.0 if sex == "male" else 0.0,
        "sex_female":  1.0 if sex == "female" else 0.0,
        "sex_unknown": 1.0 if sex not in ("male", "female") else 0.0,
    }
    for col in meta_cols:
        if col.startswith("loc_"):
            values[col] = 1.0 if col == f"loc_{localization}" else 0.0
    return np.array([values[c] for c in meta_cols], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────
def load_model(model_name: str, device: torch.device,
                metadata_dim: Optional[int] = None) -> nn.Module:
    """
    Builds the architecture with ImageNet weights OFF (they would just be
    overwritten by the checkpoint, and skipping them avoids an unnecessary
    network call) then loads this project's own best checkpoint on top.
    """
    info = MODEL_REGISTRY[model_name]
    ckpt_path = checkpoint_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    build_kwargs = {"model_name": model_name, "num_classes": len(CLASS_NAMES),
                     "pretrained": False}
    if info["multimodal"]:
        if metadata_dim is None:
            raise ValueError("metadata_dim is required for the multimodal model")
        build_kwargs["metadata_dim"] = metadata_dim

    model = build_model(**build_kwargs)
    # weights_only=True: these are our own checkpoints (plain tensors + a few
    # ints/floats), so the safe unpickler is sufficient and avoids running
    # arbitrary pickle code.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    model.trained_epoch = ckpt.get("epoch")
    model.val_bal_acc = ckpt.get("val_bal_acc")
    return model


# ─────────────────────────────────────────────────────────────────────────
# Preprocessing / prediction
# ─────────────────────────────────────────────────────────────────────────
def preprocess_image(pil_image, image_size: int) -> torch.Tensor:
    """Deterministic eval-time transform: Resize -> CenterCrop -> Normalize."""
    transform = get_transforms("val", image_size)
    return transform(pil_image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict_single(model: nn.Module, image_tensor: torch.Tensor,
                    device: torch.device,
                    metadata_vector: Optional[np.ndarray] = None) -> np.ndarray:
    """Returns a [num_classes] softmax probability vector."""
    image_tensor = image_tensor.to(device)
    if metadata_vector is not None:
        meta_tensor = torch.from_numpy(metadata_vector).unsqueeze(0).to(device)
        logits = model(image_tensor, meta_tensor)
    else:
        logits = model(image_tensor)
    return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_ensemble(models: dict, pil_image, device: torch.device) -> tuple:
    """
    Soft-voting ensemble over ENSEMBLE_MEMBERS. All three members score the
    SAME 224x224 preprocessing (matches how the ensemble was evaluated in
    notebook 03, sharing one test_loader across architecturally different
    models).

    Args:
        models: {model_name: loaded nn.Module}, must contain ENSEMBLE_MEMBERS

    Returns: (ensemble_probs [C], per_model_probs {name: [C]})
    """
    image_tensor = preprocess_image(pil_image, ENSEMBLE_IMAGE_SIZE).to(device)
    per_model_probs = {}
    for name in ENSEMBLE_MEMBERS:
        logits = models[name](image_tensor)
        per_model_probs[name] = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    stacked = np.stack(list(per_model_probs.values()))
    ensemble_probs = stacked.mean(axis=0)
    return ensemble_probs, per_model_probs


def risk_for_class(class_name: str) -> str:
    return RISK_LEVEL[class_name]


def sorted_prediction(probs: np.ndarray) -> list:
    """Returns [(class_name, prob), ...] sorted by probability, descending."""
    order = np.argsort(probs)[::-1]
    return [(CLASS_NAMES[i], float(probs[i])) for i in order]


# ─────────────────────────────────────────────────────────────────────────
# Explainability: GradCAM / GradCAM++ / Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────
class ImageOnlyWrapper(nn.Module):
    """
    Freezes a MultiModalFusion model's metadata input to a fixed vector so
    the image-only GradCAM / IntegratedGradients machinery in src.evaluate
    (which calls `model(image_tensor)` with a single argument) can run
    against the image branch.
    """

    def __init__(self, mm_model: nn.Module, metadata_vector: np.ndarray,
                 device: torch.device):
        super().__init__()
        self.mm_model = mm_model
        meta = torch.from_numpy(metadata_vector).unsqueeze(0).to(device)
        self.register_buffer("meta", meta)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        meta = self.meta.expand(image.size(0), -1)
        return self.mm_model(image, meta)


def _target_layer(xai_model: nn.Module, model_name: str) -> nn.Module:
    """Last spatial conv block to hook GradCAM onto, per architecture."""
    if model_name == "efficientnet_b4":
        return xai_model.features[-1]
    if model_name == "resnet50_cbam":
        return xai_model.backbone[-1]
    if model_name == "densenet169":
        # NOT xai_model.features (the whole Sequential): DenseNet169.forward()
        # immediately does F.relu(features, inplace=True) on that exact output
        # tensor, and PyTorch forbids an in-place op on a tensor wrapped by a
        # full_backward_hook (raises "...is a view and is being modified
        # inplace"). denseblock4's output flows through norm5 (out-of-place)
        # before the in-place ReLU, so hooking it avoids the crash while
        # pointing at the same 1664-channel, same-resolution feature map.
        return xai_model.features.denseblock4
    if model_name == "multimodal":
        return xai_model.mm_model.image_features[-1]  # xai_model is the wrapper here
    raise ValueError(f"GradCAM is not supported for '{model_name}' "
                      f"(no spatial conv feature map — e.g. a pure transformer).")


def _clear_hooks(module: nn.Module) -> None:
    """Drops the hooks GradCAM/GradCAM++ registered so repeated calls in a
    long-running app session don't accumulate stale hooks on cached models."""
    for attr in ("_forward_hooks", "_backward_hooks", "_full_backward_hooks"):
        d = getattr(module, attr, None)
        if d:
            d.clear()


def denormalize_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] normalised tensor -> [H,W,3] uint8 RGB, for overlaying heatmaps on."""
    img = image_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array(DATASET_MEAN)
    std = np.array(DATASET_STD)
    img = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


def generate_explanations(model_name: str, model: nn.Module,
                           image_tensor: torch.Tensor, device: torch.device,
                           class_idx: int,
                           metadata_vector: Optional[np.ndarray] = None,
                           ig_steps: int = 32) -> dict:
    """
    Runs GradCAM, GradCAM++ and Integrated Gradients for one image against
    one target class.

    Returns:
        {"original": uint8 HWC, "gradcam": uint8 HWC,
         "gradcam_pp": uint8 HWC, "integrated_gradients": uint8 HWC}
    """
    if not MODEL_REGISTRY[model_name]["supports_xai"]:
        raise ValueError(f"'{model_name}' has no spatial conv feature map "
                          f"to run GradCAM against.")

    image_tensor = image_tensor.to(device)
    xai_model = model
    if MODEL_REGISTRY[model_name]["multimodal"]:
        if metadata_vector is None:
            raise ValueError("metadata_vector is required to explain the multimodal model")
        xai_model = ImageOnlyWrapper(model, metadata_vector, device).to(device)

    target_layer = _target_layer(xai_model, model_name)
    display_img = denormalize_to_uint8(image_tensor)

    gradcam = GradCAM(xai_model, target_layer)
    cam, _ = gradcam.generate(image_tensor, class_idx=class_idx)
    overlay_gc = gradcam.overlay_on_image(display_img, cam)
    _clear_hooks(target_layer)

    gradcam_pp = GradCAMPlusPlus(xai_model, target_layer)
    cam_pp, _ = gradcam_pp.generate(image_tensor, class_idx=class_idx)
    overlay_gpp = gradcam.overlay_on_image(display_img, cam_pp)
    _clear_hooks(target_layer)

    ig = IntegratedGradients(xai_model)
    attr, _ = ig.generate(image_tensor, class_idx=class_idx, n_steps=ig_steps)
    import cv2
    h, w = display_img.shape[:2]
    attr_resized = cv2.resize(attr, (w, h))
    overlay_ig = gradcam.overlay_on_image(display_img, attr_resized)

    return {
        "original": display_img,
        "gradcam": overlay_gc,
        "gradcam_pp": overlay_gpp,
        "integrated_gradients": overlay_ig,
    }
