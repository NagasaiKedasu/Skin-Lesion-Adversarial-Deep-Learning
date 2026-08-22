import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, balanced_accuracy_score,
    cohen_kappa_score, roc_curve, auc,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import label_binarize
from scipy import stats

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL  = {
    "akiec": "Actinic Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    num_classes: int = 7,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    save_dir: str = "results",
) -> dict:
    """
    Full evaluation with all clinical metrics.

    Args:
        all_probs:   [N, C] softmax probabilities
        all_preds:   [N]    predicted class indices
        all_targets: [N]    ground truth class indices

    Returns:
        metrics dict
    """
    os.makedirs(save_dir, exist_ok=True)

    # Binarize for OvR AUC
    targets_bin = label_binarize(all_targets, classes=list(range(num_classes)))

    acc      = np.mean(all_preds == all_targets)
    bal_acc  = balanced_accuracy_score(all_targets, all_preds)
    kappa    = cohen_kappa_score(all_targets, all_preds, weights="quadratic")
    auc_macro = roc_auc_score(targets_bin, all_probs, multi_class="ovr",
                               average="macro")
    auc_weighted = roc_auc_score(targets_bin, all_probs, multi_class="ovr",
                                  average="weighted")

    # Per-class report
    report = classification_report(
        all_targets, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    metrics = {
        "accuracy":          acc,
        "balanced_accuracy": bal_acc,
        "cohen_kappa":       kappa,
        "auc_macro":         auc_macro,
        "auc_weighted":      auc_weighted,
        "per_class":         report,
    }

    # Bootstrap CIs
    if bootstrap_ci:
        boot_bal_accs = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(all_targets), len(all_targets))
            ba  = balanced_accuracy_score(all_targets[idx], all_preds[idx])
            boot_bal_accs.append(ba)
        ci_low  = np.percentile(boot_bal_accs, 2.5)
        ci_high = np.percentile(boot_bal_accs, 97.5)
        metrics["balanced_accuracy_95ci"] = (ci_low, ci_high)
        print(f"Balanced Accuracy: {bal_acc:.4f} (95% CI: {ci_low:.4f}–{ci_high:.4f})")

    # Pretty print
    print("\n" + "="*60)
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    print(f"  AUC (macro OvR):   {auc_macro:.4f}")
    print(f"  AUC (weighted):    {auc_weighted:.4f}")
    print("="*60)
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(all_targets: np.ndarray, all_preds: np.ndarray,
                          save_path: str = "results/confusion_matrix.png",
                          normalize: bool = True):
    cm = confusion_matrix(all_targets, all_preds)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix" if normalize else "Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i,j]:.2f}" if normalize else str(int(cm[i,j]))
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Saved confusion matrix → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ROC curves (multi-class OvR)
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(all_probs: np.ndarray, all_targets: np.ndarray,
                    save_path: str = "results/roc_curves.png"):
    n_classes = all_probs.shape[1]
    targets_bin = label_binarize(all_targets, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(targets_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{CLASS_FULL[cls]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Saved ROC curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    GradCAM: Gradient-weighted Class Activation Maps.
    Highlights regions most influential for the model's prediction.

    Reference: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization"
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        Args:
            input_tensor: [1, 3, H, W]
            class_idx:    target class (None → argmax)
        Returns:
            cam: [H, W] normalised heatmap in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Pool gradients across spatial dims → [C]
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx

    def overlay_on_image(self, image_np: np.ndarray,
                         cam: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
        """Overlay heatmap on original image.

        Args:
            image_np: [H, W, 3] uint8 numpy array (RGB).
            cam:      [H', W'] float numpy array (normalised to [0, 1]).
            alpha:    blend weight for the heatmap.

        Returns:
            [H, W, 3] uint8 overlay array.
        """
        import cv2

        # Resize CAM to match the display image
        h, w = image_np.shape[:2]
        cam_resized = cv2.resize(cam.astype(np.float32), (w, h))

        img = image_np.astype(np.float32) / 255.0

        # Create heatmap
        heatmap = cm.jet(cam_resized)[:, :, :3]

        # Overlay
        overlay = (1 - alpha) * img + alpha * heatmap
        return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM++
# ─────────────────────────────────────────────────────────────────────────────
class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ with improved gradient weighting for better spatial precision.

    Reference: Chattopadhay et al. (2018) - "Grad-CAM++: Improved Visual
    Explanations for Deep Convolutional Networks"
    """

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int = None) -> tuple[np.ndarray, int]:
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        grads    = self.gradients          # [1, C, H, W]
        acts     = self.activations        # [1, C, H, W]

        # Second-order weights (GradCAM++ formula)
        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        denom     = 2 * grads_sq + (acts * grads_cu).sum(dim=[2, 3], keepdim=True)
        denom     = torch.where(denom != 0, denom,
                                torch.ones_like(denom))
        alpha     = grads_sq / denom
        weights   = (alpha * F.relu(grads)).sum(dim=[2, 3], keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────
class IntegratedGradients:
    """
    Integrated Gradients attribution.
    Attribute importance along a path from a baseline (black) to the input.

    Reference: Sundararajan et al. (2017) - "Axiomatic Attribution for Deep
    Networks"
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int = None,
                 n_steps: int = 50,
                 baseline: torch.Tensor = None) -> np.ndarray:
        """
        Returns: attribution map [H, W] (mean over channels, absolute value)
        """
        self.model.eval()
        device = input_tensor.device

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        if class_idx is None:
            with torch.no_grad():
                class_idx = self.model(input_tensor).argmax(dim=1).item()

        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, n_steps, device=device)
        interpolated = torch.stack([
            baseline + a * (input_tensor - baseline)
            for a in alphas
        ]).squeeze(1)  # [n_steps, C, H, W]
        interpolated.requires_grad_(True)

        # Forward pass
        output = self.model(interpolated)
        target_output = output[:, class_idx].sum()
        target_output.backward()

        grads = interpolated.grad  # [n_steps, C, H, W]

        # Riemann sum approximation
        avg_grads   = grads.mean(dim=0)                           # [C, H, W]
        ig_attrs    = (input_tensor.squeeze() - baseline.squeeze()) * avg_grads
        attribution = ig_attrs.abs().mean(dim=0).cpu().numpy()    # [H, W]

        if attribution.max() > attribution.min():
            attribution = (attribution - attribution.min()) / \
                          (attribution.max() - attribution.min())

        return attribution, class_idx


# ─────────────────────────────────────────────────────────────────────────────
# XAI visualisation helper
# ─────────────────────────────────────────────────────────────────────────────
def visualise_explanations(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    target_layer: nn.Module,
    device: torch.device,
    n_samples: int = 10,
    save_dir: str = "results/xai",
):
    """
    Generate and save GradCAM, GradCAM++, and Integrated Gradients
    visualisations for a sample of test images.
    """
    os.makedirs(save_dir, exist_ok=True)

    gradcam     = GradCAM(model, target_layer)
    gradcam_pp  = GradCAMPlusPlus(model, target_layer)
    ig          = IntegratedGradients(model)

    n = min(n_samples, len(images))
    for i in range(n):
        img     = images[i:i+1].to(device)
        target  = targets[i].item()

        # GradCAM
        cam, pred = gradcam.generate(img, class_idx=target)
        overlay_gc = gradcam.overlay_on_image(img, cam)

        # GradCAM++
        cam_pp, _  = gradcam_pp.generate(img, class_idx=target)
        overlay_gpp = gradcam.overlay_on_image(img, cam_pp)

        # Integrated Gradients
        attr, _ = ig.generate(img, class_idx=target)
        import cv2
        h, w = img.shape[-2:]
        attr_resized = cv2.resize(attr, (w, h))
        attr_overlay = gradcam.overlay_on_image(img, attr_resized)

        # Original image
        img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.7630, 0.5456, 0.5700])
        std  = np.array([0.1409, 0.1526, 0.1697])
        img_np = np.clip(img_np * std + mean, 0, 1)

        # Plot 4 panels
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        true_name = CLASS_NAMES[target]
        pred_name = CLASS_NAMES[pred]

        axes[0].imshow(img_np)
        axes[0].set_title(f"Original\nTrue: {true_name} | Pred: {pred_name}")
        axes[1].imshow(overlay_gc)
        axes[1].set_title("GradCAM")
        axes[2].imshow(overlay_gpp)
        axes[2].set_title("GradCAM++")
        axes[3].imshow(attr_overlay)
        axes[3].set_title("Integrated Gradients")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"sample_{i:03d}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close()

    print(f"[XAI] Saved {n} explanation figures → {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble prediction
# ─────────────────────────────────────────────────────────────────────────────
def ensemble_predict(
    models: list,
    loader,
    device: torch.device,
    method: str = "soft_voting",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensemble inference over a list of trained models.

    Args:
        models:  List of nn.Module (already loaded with best weights)
        loader:  DataLoader
        method:  'soft_voting' (average probabilities, recommended) or
                 'hard_voting' (majority vote on predictions)

    Returns: (all_probs, all_preds, all_targets)
    """
    for m in models:
        m.eval()

    all_probs_list = [[] for _ in models]
    all_targets    = []

    with torch.no_grad():
        for batch in loader:
            images  = batch[0].to(device)
            targets = batch[-1]
            if targets.dim() == 2:
                targets = targets.argmax(dim=1)
            all_targets.extend(targets.numpy())

            for j, model in enumerate(models):
                logits = model(images)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs_list[j].append(probs)

    # [num_models, N, C]
    stacked = np.stack([np.concatenate(p, axis=0) for p in all_probs_list])

    if method == "soft_voting":
        ensemble_probs = stacked.mean(axis=0)           # [N, C]
    elif method == "hard_voting":
        per_model_preds = stacked.argmax(axis=2)         # [M, N]
        ensemble_probs = np.zeros((stacked.shape[1], stacked.shape[2]))
        for n in range(stacked.shape[1]):
            votes = per_model_preds[:, n]
            for c in range(stacked.shape[2]):
                ensemble_probs[n, c] = (votes == c).sum() / len(models)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    preds   = ensemble_probs.argmax(axis=1)
    targets = np.array(all_targets)
    return ensemble_probs, preds, targets


# ─────────────────────────────────────────────────────────────────────────────
# Training curve plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(history: dict,
                          save_path: str = "results/training_curves.png"):
    """Plot loss and balanced accuracy curves for train and val."""
    epochs = range(1, len(history["train"]) + 1)

    train_loss = [h["loss"] for h in history["train"]]
    val_loss   = [h["loss"] for h in history["val"]]
    train_ba   = [h["balanced_accuracy"] for h in history["train"]]
    val_ba     = [h["balanced_accuracy"] for h in history["val"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "b-o", label="Train", markersize=3)
    ax1.plot(epochs, val_loss,   "r-o", label="Val",   markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_ba, "b-o", label="Train", markersize=3)
    ax2.plot(epochs, val_ba,   "r-o", label="Val",   markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_title("Training & Validation Balanced Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Saved training curves → {save_path}")
