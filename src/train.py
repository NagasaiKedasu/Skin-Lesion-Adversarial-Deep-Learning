"""
Training Engine
===============
Production-grade training loop with:
  - AMP (Automatic Mixed Precision)
  - Gradient accumulation
  - Two-stage training (freeze → unfreeze backbone)
  - Cosine LR scheduling with warm restarts
  - Early stopping (by balanced accuracy)
  - Best checkpoint saving
  - Comprehensive logging (loss, accuracy, balanced accuracy per epoch)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Warmup + Cosine Scheduler
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                    T_0: int = 10, T_mult: int = 2, eta_min: float = 1e-6):
    """
    Linear warmup for first `warmup_epochs`, then cosine annealing with restarts.
    """
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult,
                                         eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


# ─────────────────────────────────────────────────────────────────────────────
# Epoch train / eval loops
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler: GradScaler, grad_accum_steps: int = 2,
                    clip_value: float = 1.0, is_multimodal: bool = False):
    """
    Single training epoch.

    Returns: dict with 'loss', 'accuracy', 'balanced_accuracy'
    """
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # Unpack batch (supports image-only, multimodal, and soft-label)
        if is_multimodal:
            if len(batch) == 3:
                images, metadata, targets = batch
                metadata = metadata.to(device)
            else:
                images, targets = batch
                metadata = None
        else:
            if len(batch) == 3:
                images, metadata, targets = batch
                metadata = metadata.to(device)
            else:
                images, targets = batch
                metadata = None

        images  = images.to(device)
        targets = targets.to(device)

        with autocast():
            if metadata is not None:
                logits = model(images, metadata)
            else:
                logits = model(images)
            loss = criterion(logits, targets) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

        # For metrics, use hard targets
        if targets.dim() == 2:
            hard_targets = targets.argmax(dim=1)
        else:
            hard_targets = targets

        preds = logits.detach().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(hard_targets.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = np.mean(np.array(all_preds) == np.array(all_targets))
    bal_acc  = balanced_accuracy_score(all_targets, all_preds)

    return {"loss": avg_loss, "accuracy": acc, "balanced_accuracy": bal_acc}


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             is_multimodal: bool = False):
    """
    Validation / test evaluation.

    Returns: dict with 'loss', 'accuracy', 'balanced_accuracy',
             'all_probs', 'all_preds', 'all_targets'
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_targets = [], [], []

    for batch in loader:
        if len(batch) == 3 and is_multimodal:
            images, metadata, targets = batch
            metadata = metadata.to(device)
        elif len(batch) == 3:
            images, metadata, targets = batch
            metadata = metadata.to(device)
        else:
            images, targets = batch
            metadata = None

        images  = images.to(device)
        targets = targets.to(device)

        with autocast():
            if metadata is not None:
                logits = model(images, metadata)
            else:
                logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        hard_targets = targets.argmax(dim=1) if targets.dim() == 2 else targets

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(hard_targets.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = np.mean(np.array(all_preds) == np.array(all_targets))
    bal_acc  = balanced_accuracy_score(all_targets, all_preds)

    return {
        "loss":             avg_loss,
        "accuracy":         acc,
        "balanced_accuracy": bal_acc,
        "all_probs":        np.array(all_probs),
        "all_preds":        np.array(all_preds),
        "all_targets":      np.array(all_targets),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs: int       = 50,
    freeze_epochs: int    = 3,
    patience: int         = 10,
    checkpoint_dir: str   = "checkpoints",
    model_name: str       = "model",
    grad_accum_steps: int = 2,
    clip_value: float     = 1.0,
    is_multimodal: bool   = False,
) -> dict:
    """
    Full training loop with:
      - Backbone freeze for first `freeze_epochs`
      - Early stopping on balanced accuracy
      - Best model checkpointing

    Returns:
        history: dict with 'train' and 'val' lists of epoch metrics
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler = GradScaler()
    best_val_bal_acc = 0.0
    patience_counter = 0
    history = {"train": [], "val": []}

    # Stage 1: freeze backbone
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        print(f"[Train] Freezing backbone for {freeze_epochs} epochs...")
        model.freeze_backbone()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after warm-up
        if epoch == freeze_epochs + 1 and hasattr(model, "unfreeze_backbone"):
            print(f"[Train] Unfreezing backbone at epoch {epoch}")
            model.unfreeze_backbone()

        # --- Train ---
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, grad_accum_steps, clip_value, is_multimodal
        )

        # --- Validate ---
        val_metrics = evaluate(
            model, val_loader, criterion, device, is_multimodal
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}  "
            f"BalAcc: {train_metrics['balanced_accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}  "
            f"BalAcc: {val_metrics['balanced_accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # --- Early stopping & checkpointing ---
        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            patience_counter = 0
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_bal_acc": best_val_bal_acc,
            }, ckpt_path)
            print(f"  ✓ New best | BalAcc: {best_val_bal_acc:.4f} → saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Load best weights
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Train] Loaded best checkpoint (epoch {ckpt['epoch']}, "
              f"BalAcc: {ckpt['val_bal_acc']:.4f})")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling (post-hoc calibration)
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    """
    Post-hoc probability calibration via temperature scaling.
    Fits a single scalar T on validation set to minimise NLL.

    Reference: Guo et al. (2017) - "On Calibration of Modern Neural Networks"
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, model, val_loader, device):
        """Optimise temperature on the validation set."""
        model.eval()
        all_logits, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                images, targets = batch[0].to(device), batch[-1].to(device)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        all_logits  = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        if all_targets.dim() == 2:
            all_targets = all_targets.argmax(dim=1)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll = nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            loss = nll(self.forward(all_logits), all_targets)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        print(f"[TemperatureScaler] Optimal T = {self.temperature.item():.4f}")
        return self
