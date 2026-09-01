"""Home page — project overview and navigation."""

import streamlit as st
import torch

from app_common import DISCLAIMER, available_models, get_cached_device, get_metadata_df

st.title("🩺 HAM10000 Skin Lesion Classifier")
st.caption(
    "Dermoscopic image classification across 7 lesion types — "
    "EfficientNet-B4, ViT-B/16, ResNet-50+CBAM, DenseNet-169, "
    "a soft-voting ensemble, and a metadata-fused multimodal model."
)

st.warning(DISCLAIMER, icon="🎓")

df = get_metadata_df()
device = get_cached_device()
n_models = len(available_models())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Diagnosis classes", "7")
c2.metric("Training images", f"{len(df):,}")
c3.metric("Unique lesions", f"{df['lesion_id'].nunique():,}")
c4.metric("Best balanced accuracy", "0.794", help="Ensemble of EfficientNet-B4 + ResNet-50+CBAM + DenseNet-169")
c5.metric("Models loaded", f"{n_models} / 5")

st.caption(f"Compute device for inference: **{device.type.upper()}**"
           + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

st.divider()

st.subheader("How the pipeline fits together")
flow_cols = st.columns(5)
steps = [
    ("📷", "Dermoscopic image", "+ optional patient metadata (age, sex, site)"),
    ("🧠", "5 trained models", "EfficientNet-B4 · ViT-B/16 · ResNet50+CBAM · DenseNet-169 · MultiModal"),
    ("🗳️", "Soft-voting ensemble", "Average probabilities across 3 architecturally diverse CNNs"),
    ("🔥", "Explainability", "GradCAM / GradCAM++ / Integrated Gradients localise the lesion"),
    ("⚠️", "Risk triage", "Benign · pre-malignant · malignant, never on color alone"),
]
for col, (icon, title, desc) in zip(flow_cols, steps):
    with col:
        st.markdown(f"### {icon}")
        st.markdown(f"**{title}**")
        st.caption(desc)

st.divider()

st.subheader("Explore the app")
nav_cols = st.columns(3)
with nav_cols[0]:
    with st.container(border=True):
        st.markdown("#### 🔬 Predict")
        st.caption("Upload a lesion photo — or pick a labelled sample — and classify it "
                    "with any model or the ensemble.")
        st.page_link("views/predict.py", label="Open Predict", icon="🔬")
    with st.container(border=True):
        st.markdown("#### 🗂️ Dataset Explorer")
        st.caption("Class imbalance, age/sex/site distributions, and a browsable "
                    "image gallery from HAM10000.")
        st.page_link("views/dataset_explorer.py", label="Open Dataset Explorer", icon="🗂️")
with nav_cols[1]:
    with st.container(border=True):
        st.markdown("#### 🧠 Explainability")
        st.caption("See GradCAM, GradCAM++ and Integrated Gradients heatmaps for "
                    "any conv-based model's prediction.")
        st.page_link("views/explain.py", label="Open Explainability", icon="🧠")
    with st.container(border=True):
        st.markdown("#### ❓ Project Q&A")
        st.caption("The full viva document: novelty, imbalance handling, "
                    "architectures, deployment notes.")
        st.page_link("views/about.py", label="Open Project Q&A", icon="❓")
with nav_cols[2]:
    with st.container(border=True):
        st.markdown("#### 📊 Model Performance")
        st.caption("Confusion matrices, ROC curves, training curves, and the "
                    "baseline-vs-deep-learning comparison.")
        st.page_link("views/performance.py", label="Open Model Performance", icon="📊")
