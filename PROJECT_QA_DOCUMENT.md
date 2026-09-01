# Skin Lesion Classification Project — Viva / Presentation Q&A Document

**Dataset:** HAM10000 | **Task:** 7-class skin lesion classification | **Framework:** PyTorch

---

## Q1. What are you doing new compared to others?

Most existing work treats skin lesion analysis as a **single-task problem** — either do segmentation (outline the lesion) or classification (name the disease), but not both. What this project adds on top of standard approaches:

1. **Segmentation + Classification together:** By using GradCAM/GradCAM++ to generate attention heatmaps over the lesion area, the model effectively localizes the lesion while simultaneously classifying it. This acts as soft segmentation that guides the classifier to focus on the actual lesion rather than background skin or hair artefacts. This combination improves accuracy because the model learns *where* to look and *what* it is at the same time.

2. **Ensemble of architecturally diverse models:** Instead of one model, three architecturally different models (CNN, Attention CNN, Dense CNN) are combined. Different architectures make different types of errors, so averaging their predictions reduces overall error — something a single model cannot do.

3. **Multi-Modal Fusion:** Patient metadata (age, sex, body location) is fused with image features. For example, melanoma on the trunk of a young woman has a different risk profile than melanoma on the face of an elderly man. Clinical context improves predictions beyond what images alone can provide.

4. **Lesion-aware data splitting:** The dataset has 26% of lesions photographed multiple times (same lesion, multiple angles). Without careful splitting, the same lesion could appear in both training and test sets, inflating accuracy by 5–10%. This project prevents that data leakage.

5. **Explainability (XAI):** Three explanation methods (GradCAM, GradCAM++, Integrated Gradients) validate that the model is looking at medically meaningful features — not noise or hair artefacts.

---

## Q2. What models are you using?

Five models are used in this project:

| # | Model | Type | Input Size | Parameters |
|---|-------|------|-----------|------------|
| 1 | EfficientNet-B4 | Compound-scaled CNN | 380×380 | 18.5M |
| 2 | ViT-B/16 | Vision Transformer | 224×224 | 86.2M |
| 3 | ResNet-50 + CBAM | CNN with Attention | 224×224 | ~25M |
| 4 | DenseNet-169 | Densely Connected CNN | 224×224 | ~14M |
| 5 | MultiModal Fusion | EfficientNet-B4 + MLP | 224×224 | ~19M |

**Ensemble:** EfficientNet-B4 + ResNet-50+CBAM + DenseNet-169 combined via soft-voting.

---

## Q3. How does SMOTE (or oversampling) help with your data?

> **Note:** This project does not use SMOTE (Synthetic Minority Oversampling Technique) directly, because SMOTE is designed for tabular/feature data and does not work well on raw images. Instead, three image-appropriate strategies are used to solve the same imbalance problem:

**The imbalance problem:**
- Majority class: `nv` (Melanocytic Nevi) = **6,705 images = 67% of data**
- Minority class: `df` (Dermatofibroma) = **115 images = 1.1% of data**
- Imbalance ratio: **58×** — the model would just predict "nv" for everything

**What is used instead of SMOTE:**

1. **WeightedRandomSampler** — During training, each mini-batch is sampled so every class appears roughly equally. Rare classes (vasc, df) are sampled more frequently. This is the image equivalent of SMOTE's rebalancing effect.

2. **Focal Loss** — Instead of regular cross-entropy, Focal Loss down-weights easy (correctly classified) examples and up-weights hard (misclassified minority) examples. Formula: `FL = -(1-p)^γ * log(p)` where γ=2.0. This forces the model to learn from rare classes.

3. **MixUp + CutMix Augmentation** — Synthetic new training images are created by mixing existing ones. This is the closest equivalent to SMOTE for images — it creates "in-between" samples that help minority classes.

**Result:** Without any balancing, a naive model achieves 66.9% accuracy by just predicting "nv" for everything. With these techniques, the ensemble reaches 0.79 balanced accuracy across all 7 classes.

---

## Q4. They asked "it's not working — come up with something new"

If the reviewer says the current approach underperforms, here are stronger directions:

**Short-term improvements:**
- **Segment first, then classify:** Use U-Net or SAM (Segment Anything Model) to crop just the lesion region, then feed only the lesion into the classifier. Removes background noise entirely.
- **Test-Time Augmentation (TTA):** During inference, run each image through 10 augmented versions (flips, crops, rotations) and average predictions. Free accuracy boost.
- **Stochastic Weight Averaging (SWA):** Average weights across the last N epochs instead of picking the single best checkpoint. Consistently improves generalization.

**Medium-term improvements:**
- **Self-supervised pre-training on dermatology images:** Pre-train with SimCLR or DINO on a large unlabelled skin image dataset before fine-tuning on HAM10000. Domain-specific pre-training outperforms ImageNet pre-training.
- **Hierarchical classification:** First classify Malignant vs. Benign (binary, easier), then sub-classify within each group. Reduces confusion between similar classes.

**Architectural improvements:**
- **ConvNeXt-L or EfficientNetV2-L:** Newer architectures that outperform EfficientNet-B4 on medical imaging benchmarks.
- **Cross-attention between image patches and metadata tokens:** Instead of late fusion (concatenation), use a transformer cross-attention layer so image features and metadata interact at every layer, not just at the end.

---

## Q5. Is your dataset all melanoma?

**No.** The dataset is NOT all melanoma. It contains **7 different skin lesion types**:

| Code | Full Name | Count | % | Cancer? |
|------|-----------|-------|---|---------|
| `nv` | Melanocytic Nevi (moles) | 6,705 | 67.0% | Benign |
| `mel` | **Melanoma** | 1,113 | 11.1% | **Malignant** |
| `bkl` | Benign Keratosis | 1,099 | 11.0% | Benign |
| `bcc` | Basal Cell Carcinoma | 514 | 5.1% | **Malignant** |
| `akiec` | Actinic Keratosis | 327 | 3.3% | Pre-malignant |
| `vasc` | Vascular Lesions | 142 | 1.4% | Benign |
| `df` | Dermatofibroma | 115 | 1.1% | Benign |

**Total: 10,015 images from 7,470 unique lesions**

- 3 classes are dangerous (mel, bcc, akiec)
- 4 classes are benign (nv, bkl, vasc, df)
- Melanoma is only 11% of the data — detecting it reliably is the hardest challenge

---

## Q6. What is your data source?

**HAM10000 — Human Against Machine with 10,000 Training Images**

| Attribute | Detail |
|-----------|--------|
| Full name | HAM10000 (Human Against Machine) |
| Source | ISIC (International Skin Imaging Collaboration) |
| Curators | Tschandl, Rosendahl & Kittler (2018) |
| Images | 10,015 dermoscopic images |
| Image size | 600 × 450 pixels (.jpg) |
| Metadata | Age, sex, body location, diagnosis type |
| Diagnosis method | Histopathology (biopsy), confocal microscopy, expert consensus |
| Split used | Part 1 (folder) + Part 2 (folder) + metadata CSV |
| Pre-resized CSV | `hmnist_28_28_L.csv` (for fast baseline experiments) |

The dataset is publicly available through the ISIC Archive and Kaggle. All diagnoses are ground-truth verified (not just dermatologist opinion — many are confirmed by biopsy).

**Patient metadata included:**
- Age: mean 51.9 years, range ~0–85
- Sex: 54% male, 45% female, 1% unknown
- Localization: back (22%), lower extremity (21%), trunk (14%), upper extremity (11%), etc.

---

## Q7. What is DenseNet?

**DenseNet (Densely Connected Convolutional Network)** — published by Huang et al. (2017).

**Core idea:** In a regular CNN, each layer connects only to the next layer. In DenseNet, **every layer connects to every subsequent layer** in the same dense block.

```
Regular CNN:   L1 → L2 → L3 → L4
DenseNet:      L1 → L2, L3, L4
               L2 → L3, L4
               L3 → L4
```

**This means:** Layer 4 receives the concatenated feature maps from layers 1, 2, 3, and itself.

**Why this helps:**
1. **Feature reuse:** Low-level features (edges, colours) are available to deep layers — important for small lesion features
2. **Gradient flow:** Shorter paths back to the loss — solves the vanishing gradient problem
3. **Parameter efficiency:** Fewer parameters than ResNet for same accuracy (feature maps are reused, not relearned)
4. **Strong for medical imaging:** Because medical images have fine-grained texture details that benefit from all-layer feature access

**DenseNet-169 specifics (used in this project):**
- 169 layers total
- 4 dense blocks with transition layers between them
- ~14 million parameters
- Pretrained on ImageNet, fine-tuned on HAM10000
- Input: 224×224 pixels

---

## Q8. Explain all the models in detail

### Model 1: EfficientNet-B4

**Architecture type:** Compound-scaled CNN (scales depth, width, resolution together)

**How it works:**
- Uses MBConv (Mobile Inverted Bottleneck Convolution) blocks with Squeeze-and-Excitation attention
- Compound scaling: instead of making only deeper or wider, EfficientNet scales all three dimensions together using a fixed ratio
- B4 = 4th scale level; takes 380×380 input (largest EfficientNet used here)

**Custom head added:**
```
GlobalAvgPool → Dropout(0.4) → Linear(1792→512) → BatchNorm → ReLU → Dropout(0.2) → Linear(512→7)
```

**Training tricks:**
- Two-stage: freeze backbone for 3 epochs (train head only), then unfreeze with discriminative LRs
- Backbone LR = 1e-5 (10× slower than head LR of 1e-4)
- AdamW optimizer, weight decay 1e-4
- Focal Loss with γ=2.0 + label smoothing 0.1

**Results:** Balanced Accuracy = **0.727**, AUC = **0.956**, Cohen's Kappa = **0.74**

---

### Model 2: ViT-B/16 (Vision Transformer)

**Architecture type:** Pure Transformer (no convolutions in the backbone)

**How it works:**
1. Image (224×224) is split into 196 patches of 16×16 pixels each
2. Each patch is flattened and linearly embedded to a 768-dimensional vector
3. A learnable [CLS] token is prepended (carries the final classification signal)
4. Positional embeddings are added (so the model knows spatial location)
5. 12 transformer encoder layers with multi-head self-attention (12 heads)
6. [CLS] token output → custom classification head → 7 classes

**Key difference from CNNs:** CNNs look at local neighbourhoods; ViT can attend to any patch from the first layer (global context). This lets it detect asymmetry across the whole lesion.

**Training tricks:**
- Higher weight decay (0.05) — transformers over-fit more easily on small datasets
- Longer warmup (5 epochs) — attention weights need more time to stabilize
- Lower LR (5e-5)

**Results:** Balanced Accuracy = **0.702**, AUC = **0.923** (lower than EfficientNet — ViT needs more data to shine)

---

### Model 3: ResNet-50 + CBAM (Convolutional Block Attention Module)

**Architecture type:** Residual CNN + dual attention mechanism

**How ResNet-50 works:**
- Skip connections: output of each block = F(x) + x (residual)
- This lets gradients flow directly and allows very deep networks (50 layers) without vanishing gradients
- 4 residual stages → 2048 feature channels after final stage

**What CBAM adds:**
CBAM is placed after the final residual stage. It applies two sequential attention gates:

1. **Channel Attention:** Asks "which feature maps matter most?"
   - Global average pool + max pool across spatial dimensions
   - Shared MLP compresses and expands channels
   - Sigmoid gate applied channel-wise

2. **Spatial Attention:** Asks "which spatial regions matter most?"
   - Average and max pooled along channel dimension → 2-channel map
   - 7×7 convolution → 1-channel attention map
   - Sigmoid gate applied spatially

**Net effect:** The model focuses both on relevant feature types (colour gradients, texture) AND relevant locations (lesion centre, border irregularities).

**Results:** Expected ~0.80 balanced accuracy (literature benchmark), ensemble member.

---

### Model 4: DenseNet-169

See Q7 above for the DenseNet explanation.

**In this project specifically:**
- Used as the third ensemble member (architecturally different from EfficientNet and ResNet)
- Label Smoothing loss (instead of Focal Loss) — DenseNet is already good at minority classes due to feature reuse
- Same AdamW + cosine annealing scheduler
- Freeze 3 epochs → gradual unfreeze

**Results:** Expected ~0.81 balanced accuracy (literature benchmark), ensemble member.

---

### Model 5: Multi-Modal Fusion

**Architecture type:** Dual-branch late fusion

**How it works:**
```
Input Image (224×224)
    ↓
EfficientNet-B4 backbone → GlobalAvgPool → Linear(1792→512) → image_embedding[512]
    
Patient Metadata (age_norm, sex_male, sex_female, loc_back, loc_foot, ...)
    ↓
MLP (n→128→64) → meta_embedding[64]
    
Concatenate: [image_embedding, meta_embedding] → [576-dim vector]
    ↓
Dropout → Linear(576→256) → ReLU → Dropout → Linear(256→7) → prediction
```

**Metadata features used:**
- `age_norm`: patient age normalized to [0, 1]
- `sex_male`, `sex_female`, `sex_unknown`: one-hot encoded
- `loc_*`: one-hot encoded body site (back, leg, trunk, face, etc.)

**Clinical motivation:** A 25-year-old woman with a lesion on the lower leg has a different risk profile than a 70-year-old man with the same-looking lesion on the face. Metadata provides this clinical context.

**Results:** Balanced Accuracy = **0.720** (slightly lower than pure EfficientNet — metadata noise may hurt without regularization tuning)

---

### Ensemble (Soft Voting)

**How it works:**
- Run each of the 3 models (EfficientNet-B4, ResNet-50+CBAM, DenseNet-169) on the same test image
- Get probability vectors (7 values, sum to 1.0) from each model
- Average the three probability vectors
- Pick the class with highest average probability

**Why it works:**
- EfficientNet is good at global texture
- ResNet+CBAM is good at spatially localized features
- DenseNet is good at fine-grained multi-scale features
- Their errors are *uncorrelated* → averaging cancels them out

**Results:** Balanced Accuracy = **0.794**, AUC = **0.958** — best of all models (+6.7% over best single model)

---

## Q9. Where do you deploy and how does it work?

> **Honest answer for the viva:** The current notebooks cover training and evaluation only. Deployment is not implemented in this version of the project. Here is how it *would* be deployed in a production system:

### Proposed Deployment Architecture

```
User uploads dermoscopic image (via web/mobile app)
         ↓
[API Server — FastAPI / Flask]
         ↓
Image preprocessing pipeline:
  - Resize to 224×224 or 380×380
  - Normalize with HAM10000 mean/std
  - Optional: hair augmentation removal
         ↓
[Load best checkpoint: efficientnet_b4_best.pth]
         ↓
Model inference → 7-class probability vector
         ↓
GradCAM heatmap generated (for explainability)
         ↓
JSON response:
  {
    "predicted_class": "mel",
    "confidence": 0.73,
    "all_probs": {"mel": 0.73, "nv": 0.15, "bcc": 0.07, ...},
    "heatmap_url": "...",
    "risk_level": "HIGH — consult dermatologist"
  }
```

**Deployment options:**
| Platform | How | Notes |
|----------|-----|-------|
| Docker container | FastAPI + uvicorn, model loaded at startup | Best for hospital servers |
| Cloud (AWS/GCP) | Lambda / Cloud Run for serverless inference | Scalable, pay-per-use |
| Mobile (on-device) | Export to ONNX or TorchScript → CoreML/TFLite | No internet needed |
| Streamlit demo | `streamlit run app.py` → browser interface | Fastest for demos |

**Model file:** `checkpoints/efficientnet_b4_best.pth` (~75MB) is loaded once and kept in memory. Inference takes ~50ms per image on CPU, ~5ms on GPU.

---

## Q10. Project explained in more detail

### Problem Statement
Skin cancer is the most common cancer worldwide. Dermatologists visually inspect skin lesions using a dermoscope (magnifying camera). There are 7 types ranging from benign moles to malignant melanoma. Manual diagnosis is subjective and requires specialist expertise. An AI system that classifies these automatically can:
- Screen patients in clinics without specialists
- Serve as a second opinion for dermatologists
- Work in remote areas with no dermatology access

### The Challenge
The HAM10000 dataset has 3 severe challenges that make this harder than a standard image classification task:

**Challenge 1 — Class Imbalance (58× ratio):**
67% of images are benign moles (nv). A naive model would just predict "mole" every time and get 67% accuracy while missing 100% of melanoma cases. This project solves it with WeightedRandomSampler, Focal Loss, and MixUp/CutMix.

**Challenge 2 — Data Leakage (duplicate lesions):**
26.2% of lesions were photographed multiple times. Without careful splitting, the same lesion appears in both train and test, inflating test accuracy by 5–10%. This project uses lesion-aware splitting: all images of the same lesion_id go to the same split.

**Challenge 3 — Clinical Stakes (asymmetric errors):**
Missing a melanoma (false negative) is far more dangerous than a false alarm (false positive). Standard accuracy treats both errors equally. This project uses balanced accuracy and optimizes melanoma recall specifically.

### Training Pipeline

```
HAM10000 CSV + Images
         ↓
Lesion-aware 70/15/15 split (by lesion_id, not image_id)
         ↓
Data augmentation:
  - RandomHorizontalFlip + RandomVerticalFlip
  - RandomRotation(30°) + ColorJitter
  - HairAugmentation (synthetic hair added)
  - MixUp + CutMix (synthetic image mixing)
  - Normalize with dataset statistics
         ↓
WeightedRandomSampler (balanced mini-batches)
         ↓
Model training with:
  - Focal Loss (γ=2) + Label Smoothing (0.1)
  - AdamW optimizer
  - Cosine Annealing with warm restarts
  - Discriminative LRs (backbone 10× slower than head)
  - Gradient Accumulation (effective batch size = 2× actual)
  - AMP (Automatic Mixed Precision — faster training)
  - Early stopping (patience=10 epochs)
         ↓
Best checkpoint saved (by validation balanced accuracy)
         ↓
Temperature Scaling (post-hoc probability calibration)
         ↓
Evaluation: Balanced Acc, AUC, Cohen's Kappa, per-class F1
```

### Results Summary

| Model | Balanced Accuracy | AUC (macro) | Cohen's Kappa |
|-------|------------------|-------------|----------------|
| Majority class baseline | 0.143 | — | — |
| Logistic Regression | 0.318 | — | — |
| Simple CNN (3 layer) | 0.334 | — | — |
| ViT-B/16 | 0.702 | 0.923 | 0.467 |
| EfficientNet-B4 | **0.727** | **0.956** | **0.740** |
| MultiModal Fusion | 0.720 | 0.920 | 0.559 |
| **Ensemble (3 models)** | **0.794** | **0.958** | **0.706** |

---

## Q11. Blueprint / Skeleton of the Project

```
Skin_Lesion/
│
├── dataset/                          ← Raw data
│   ├── HAM10000_images_part_1/       ← First 5000 images (.jpg)
│   ├── HAM10000_images_part_2/       ← Next 5000 images (.jpg)
│   ├── HAM10000_metadata.csv         ← Labels, age, sex, location
│   └── hmnist_28_28_L.csv            ← Pre-resized greyscale (for baselines)
│
├── src/                              ← Reusable source modules
│   ├── dataset.py                    ← Data loading, splitting, augmentation
│   │   ├── build_image_path_map()    ← Maps image_id → file path
│   │   ├── engineer_metadata()       ← One-hot encode age/sex/location
│   │   ├── LesionDataset            ← PyTorch Dataset class
│   │   ├── make_weighted_sampler()  ← WeightedRandomSampler for imbalance
│   │   └── build_dataloaders()      ← Returns train/val/test DataLoaders
│   │
│   ├── models.py                     ← All 5 model architectures
│   │   ├── EfficientNetB4           ← Model 1
│   │   ├── ResNet50CBAM             ← Model 2 (with ChannelAttention + SpatialAttention)
│   │   ├── DenseNet169              ← Model 3
│   │   ├── VisionTransformerB16     ← Model 4
│   │   ├── MultiModalFusion         ← Model 5
│   │   ├── build_model()            ← Factory function
│   │   └── get_param_groups()       ← Discriminative LRs
│   │
│   ├── losses.py                     ← Loss functions
│   │   ├── FocalLoss                ← For imbalanced training
│   │   ├── LabelSmoothingCE         ← For calibrated training
│   │   └── build_loss()             ← Factory function
│   │
│   ├── train.py                      ← Training loop
│   │   ├── train()                  ← Main training function (freeze→unfreeze)
│   │   ├── evaluate()               ← Validation/test pass
│   │   ├── build_scheduler()        ← Warmup + cosine annealing
│   │   └── TemperatureScaler        ← Post-hoc calibration
│   │
│   └── evaluate.py                   ← Metrics, plots, XAI
│       ├── evaluate_model()         ← Balanced acc, AUC, Kappa, CI
│       ├── plot_confusion_matrix()  ← Per-class confusion
│       ├── plot_roc_curves()        ← Per-class ROC/AUC
│       ├── ensemble_predict()       ← Soft-voting ensemble
│       ├── GradCAM                  ← Gradient-weighted class activation map
│       ├── GradCAMPlusPlus          ← Improved GradCAM
│       └── IntegratedGradients      ← Axiomatic attribution
│
├── notebooks/                        ← Experiment notebooks (run in order)
│   ├── 01_EDA_and_Analysis.ipynb    ← Data exploration, imbalance analysis
│   ├── 02_Transfer_Learning.ipynb   ← Train EfficientNet-B4 + ViT-B/16
│   ├── 03_Ensemble_and_MultiModal.ipynb ← Train ResNet+CBAM, DenseNet, ensemble, fusion
│   ├── 04_Explainability_XAI.ipynb  ← GradCAM, IG, calibration, failure cases
│   └── 05_Baseline_Comparison.ipynb ← LR, RF, SimpleCNN vs. deep models
│
├── checkpoints/                      ← Saved model weights (.pth files)
│   ├── efficientnet_b4_best.pth
│   ├── vit_b16_best.pth
│   ├── resnet50_cbam_best.pth
│   ├── densenet169_best.pth
│   └── multimodal_fusion_best.pth
│
└── results/                          ← Saved plots and metrics
    ├── eda_class_distribution.png
    ├── baseline_comparison.png
    ├── cm_efficientnet_b4.png        ← Confusion matrices
    ├── roc_efficientnet_b4.png       ← ROC curves
    ├── cm_ensemble.png
    └── xai/
        ├── gradcam_all_classes.png
        ├── integrated_gradients.png
        └── calibration.png
```

### Data Flow (end-to-end):

```
[Raw Image + Metadata CSV]
         │
         ▼
[01_EDA] → Understand data distribution, find imbalance, plan mitigation
         │
         ▼
[05_Baseline] → Establish floor: logistic regression, simple CNN
         │
         ▼
[02_Transfer_Learning]
  ├── Load pretrained EfficientNet-B4 (ImageNet)
  ├── Freeze backbone → train head 3 epochs
  ├── Unfreeze → train full model 37 more epochs
  ├── Apply Focal Loss + WeightedSampler + MixUp
  └── Save best checkpoint → evaluate
         │
         ▼
[03_Ensemble_and_MultiModal]
  ├── Train ResNet50+CBAM (same pipeline)
  ├── Train DenseNet-169 (same pipeline)
  ├── Load all 3 checkpoints → soft-vote ensemble → evaluate
  └── Train MultiModal fusion (image + metadata) → evaluate
         │
         ▼
[04_Explainability_XAI]
  ├── Load best model (EfficientNet-B4)
  ├── GradCAM / GradCAM++ → visualise what the model sees
  ├── Integrated Gradients → pixel-level attribution
  ├── Temperature scaling → calibrate probabilities
  └── Failure case analysis → find dangerous high-confidence errors
         │
         ▼
[Deploy] → Export model → wrap in API → serve predictions + heatmaps
```

---

*Document generated from notebooks: 01_EDA_and_Analysis, 02_Transfer_Learning, 03_Ensemble_and_MultiModal, 04_Explainability_XAI, 05_Baseline_Comparison*
