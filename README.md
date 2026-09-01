# HAM10000 Skin Lesion Classification

7-class dermoscopic image classification on HAM10000 — EfficientNet-B4, ViT-B/16,
ResNet-50+CBAM, DenseNet-169, a soft-voting ensemble, and a metadata-fused
multimodal model. Training/eval code lives in `src/`, experiments in
`notebooks/`, and the full project write-up (novelty, imbalance handling,
architectures, deployment notes) is in `PROJECT_QA_DOCUMENT.md`.

## Streamlit demo app

An interactive demo app lives at the project root (`app.py` + `views/`):

- **Predict** — classify an uploaded or sample image with any model or the ensemble
- **Explainability** — GradCAM / GradCAM++ / Integrated Gradients heatmaps
- **Model Performance** — metrics, training curves, confusion matrices, ROC curves
- **Dataset Explorer** — live class-imbalance and metadata charts, image gallery
- **Project Q&A** — the viva document, browsable in-app

### Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires the trained checkpoints in `checkpoints/*.pth` and the HAM10000 images
under `dataset/HAM10000_images_part_1/` and `dataset/HAM10000_images_part_2/`
(both already present in this repo). A CUDA GPU is used automatically if
available; otherwise it falls back to CPU.
