"""
HAM10000 Dataset Module
=======================
Handles dataset loading, lesion-aware train/val/test splitting,
class-weight computation, and metadata feature engineering.

Key design decisions:
  - Lesion-aware split: images from the same lesion_id stay in the same fold
    (prevents data leakage from duplicated lesions)
  - WeightedRandomSampler for handling severe class imbalance (~67% nv class)
  - Metadata (age, sex, localization) encoded for multi-modal fusion
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ─────────────────────────────────────────────
# Label mapping (alphabetical → int)
# ─────────────────────────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL2IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}

# HAM10000 RGB statistics (pre-computed from training set)
DATASET_MEAN = [0.7630, 0.5456, 0.5700]
DATASET_STD  = [0.1409, 0.1526, 0.1697]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build full path → image_id mapping from both image folders
# ─────────────────────────────────────────────────────────────────────────────
def build_image_path_map(data_root: str) -> dict:
    """Return {image_id: absolute_path} for all .jpg files in both part folders."""
    root = Path(data_root)
    path_map = {}
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        folder = root / part
        if folder.exists():
            for p in folder.glob("*.jpg"):
                path_map[p.stem] = str(p)
    return path_map


# ─────────────────────────────────────────────────────────────────────────────
# Metadata feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one-hot encoded metadata columns to the dataframe.

    Features created:
      - age_norm          : age normalized to [0, 1]
      - sex_male / sex_female / sex_unknown
      - loc_<body_site>   : one-hot for localization
    """
    df = df.copy()

    # Age: fill median, then normalise
    median_age = df["age"].median()
    df["age_norm"] = df["age"].fillna(median_age) / 85.0

    # Sex
    df["sex_male"]    = (df["sex"] == "male").astype(float)
    df["sex_female"]  = (df["sex"] == "female").astype(float)
    df["sex_unknown"] = (~df["sex"].isin(["male", "female"])).astype(float)

    # Localization one-hot
    loc_dummies = pd.get_dummies(df["localization"], prefix="loc").astype(float)
    df = pd.concat([df, loc_dummies], axis=1)

    return df


def get_metadata_dim(df: pd.DataFrame) -> int:
    """Return number of engineered metadata feature columns."""
    meta_cols = [c for c in df.columns if c.startswith("loc_") or
                 c in ["age_norm", "sex_male", "sex_female", "sex_unknown"]]
    return len(meta_cols)


def get_metadata_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("loc_") or
            c in ["age_norm", "sex_male", "sex_female", "sex_unknown"]]


# ─────────────────────────────────────────────────────────────────────────────
# Lesion-aware train / val / test split
# ─────────────────────────────────────────────────────────────────────────────
def lesion_aware_split(df: pd.DataFrame,
                       train_frac: float = 0.70,
                       val_frac: float   = 0.15,
                       seed: int         = 42
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at the lesion level so that all images of the same lesion
    end up in the same partition. Stratified by diagnosis class.

    Returns: (train_df, val_df, test_df)
    """
    # One row per unique lesion; keep majority dx for stratification
    lesion_df = (df.groupby("lesion_id")["dx"]
                   .agg(lambda x: x.mode()[0])
                   .reset_index())

    test_frac = 1.0 - train_frac - val_frac

    # Step 1: split off test
    train_val_lesions, test_lesions = train_test_split(
        lesion_df, test_size=test_frac,
        stratify=lesion_df["dx"], random_state=seed
    )

    # Step 2: split train vs val
    relative_val = val_frac / (train_frac + val_frac)
    train_lesions, val_lesions = train_test_split(
        train_val_lesions, test_size=relative_val,
        stratify=train_val_lesions["dx"], random_state=seed
    )

    train_ids = set(train_lesions["lesion_id"])
    val_ids   = set(val_lesions["lesion_id"])
    test_ids  = set(test_lesions["lesion_id"])

    train_df = df[df["lesion_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["lesion_id"].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df["lesion_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Class weights & WeightedRandomSampler
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(labels: list, num_classes: int = 7) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to sum to num_classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)         # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # normalise
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: list, num_classes: int = 7) -> WeightedRandomSampler:
    """WeightedRandomSampler that up-samples minority classes."""
    class_weights = compute_class_weights(labels, num_classes)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific augmentation: synthetic hair overlay
# ─────────────────────────────────────────────────────────────────────────────
class AddHairAugmentation:
    """
    Simulates dark terminal hairs on skin lesion images.
    Adapted from: Bissoto et al. (2019) - "Skin Lesion Synthesis with GANs"
    """
    def __init__(self, num_hairs: int = 5, width_range: tuple = (1, 3), p: float = 0.5):
        self.num_hairs  = num_hairs
        self.width_range = width_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img
        import cv2
        img_np = np.array(img)
        h, w   = img_np.shape[:2]
        for _ in range(self.num_hairs):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2 = x1 + np.random.randint(-w // 3, w // 3)
            y2 = y1 + np.random.randint(-h // 3, h // 3)
            thickness = np.random.randint(*self.width_range)
            color = (
                np.random.randint(0, 40),
                np.random.randint(0, 40),
                np.random.randint(0, 40),
            )
            cv2.line(img_np, (x1, y1), (x2, y2), color, thickness,
                     lineType=cv2.LINE_AA)
        return Image.fromarray(img_np)


# ─────────────────────────────────────────────────────────────────────────────
# Transform factory
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = 224,
                   use_hair_aug: bool = True) -> transforms.Compose:
    """
    Returns torchvision transform pipeline for each split.

    Train:  RandomResizedCrop → flips → color jitter → hair aug →
            ToTensor → Normalise → RandomErasing
    Val/Test: Resize → CenterCrop → ToTensor → Normalise
    """
    norm = transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)

    if split == "train":
        transform_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
        ]
        if use_hair_aug:
            transform_list.append(AddHairAugmentation(p=0.3))
        transform_list += [
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    else:
        transform_list = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm,
        ]

    return transforms.Compose(transform_list)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HAM10000Dataset(Dataset):
    """
    Multi-modal dataset: returns (image_tensor, metadata_tensor, label).
    Set `return_metadata=False` for image-only models.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 path_map: dict,
                 transform=None,
                 return_metadata: bool = False):
        self.df              = df.reset_index(drop=True)
        self.path_map        = path_map
        self.transform       = transform
        self.return_metadata = return_metadata
        self.meta_cols       = get_metadata_cols(df) if return_metadata else []
        self.labels          = self.df["label"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.path_map.get(row["image_id"])

        if img_path is None or not os.path.exists(img_path):
            # Fallback: return a black image (should not happen with correct dataset)
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])

        if self.return_metadata and self.meta_cols:
            meta = torch.tensor(row[self.meta_cols].values.astype(np.float32))
            return image, meta, label

        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# MixUp / CutMix collate wrappers
# ─────────────────────────────────────────────────────────────────────────────
class MixUpCutMixCollate:
    """
    Applies MixUp or CutMix randomly at the batch level.
    Returns mixed images and soft labels (one-hot).
    Based on: timm's FastCollateMixup
    """

    def __init__(self, num_classes: int = 7,
                 mixup_alpha: float = 0.4,
                 cutmix_alpha: float = 1.0,
                 switch_prob: float = 0.5):
        self.num_classes   = num_classes
        self.mixup_alpha   = mixup_alpha
        self.cutmix_alpha  = cutmix_alpha
        self.switch_prob   = switch_prob

    def _rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w   = int(W * cut_rat)
        cut_h   = int(H * cut_rat)
        cx      = np.random.randint(W)
        cy      = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2

    def __call__(self, batch):
        # Separate images and labels (ignore metadata for simplicity)
        if len(batch[0]) == 3:
            images, meta, labels = zip(*batch)
            meta = torch.stack(meta)
        else:
            images, labels = zip(*batch)
            meta = None

        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        # One-hot encode labels
        one_hot = torch.zeros(len(labels), self.num_classes)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        use_cutmix = (np.random.random() < self.switch_prob)

        if use_cutmix and self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_idx = torch.randperm(images.size(0))
            x1, y1, x2, y2 = self._rand_bbox(images.size(), lam)
            images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
            lam_adj = 1 - (x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3))
            mixed_labels = lam_adj * one_hot + (1 - lam_adj) * one_hot[rand_idx]
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_idx = torch.randperm(images.size(0))
            images = lam * images + (1 - lam) * images[rand_idx]
            mixed_labels = lam * one_hot + (1 - lam) * one_hot[rand_idx]

        if meta is not None:
            return images, meta, mixed_labels
        return images, mixed_labels


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    data_root: str   = "dataset",
    image_size: int  = 224,
    batch_size: int  = 32,
    num_workers: int = 4,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    seed: int         = 42,
    use_oversampling: bool = True,
    use_mixup_cutmix: bool = True,
    return_metadata: bool  = False,
    use_hair_aug: bool     = True,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Main entry point. Returns (train_loader, val_loader, test_loader, test_df).

    Usage:
        train_loader, val_loader, test_loader, test_df = build_dataloaders(
            data_root="dataset", batch_size=32
        )
    """
    # 1. Load and prepare metadata
    meta_path = os.path.join(data_root, "HAM10000_metadata.csv")
    df = pd.read_csv(meta_path)
    df["label"] = df["dx"].map(LABEL2IDX)
    df = engineer_metadata(df)

    # 2. Build image path map
    path_map = build_image_path_map(data_root)

    # 3. Lesion-aware split
    train_df, val_df, test_df = lesion_aware_split(df, train_frac, val_frac, seed)

    # 4. Transforms
    train_tf = get_transforms("train", image_size, use_hair_aug)
    eval_tf  = get_transforms("val", image_size)

    # 5. Datasets
    train_ds = HAM10000Dataset(train_df, path_map, train_tf, return_metadata)
    val_ds   = HAM10000Dataset(val_df,   path_map, eval_tf,  return_metadata)
    test_ds  = HAM10000Dataset(test_df,  path_map, eval_tf,  return_metadata)

    # 6. Sampler for class imbalance
    sampler = None
    shuffle = True
    if use_oversampling:
        sampler = make_weighted_sampler(train_df["label"].tolist())
        shuffle = False   # mutually exclusive with sampler

    # 7. Collate for MixUp/CutMix (train only)
    train_collate = MixUpCutMixCollate() if use_mixup_cutmix else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=train_collate, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[Dataset] Class dist (train): {dict(train_df['dx'].value_counts())}")

    return train_loader, val_loader, test_loader, test_df
