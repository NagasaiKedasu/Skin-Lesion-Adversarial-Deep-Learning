"""Model Performance page — metrics, training curves, confusion matrices, ROC."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import MODEL_COLOR, style_chart
from src.inference import ROOT_DIR

RESULTS_DIR = ROOT_DIR / "results"

st.title("📊 Model Performance")
st.caption("Metrics and diagnostic plots produced by the project's own training/eval "
           "pipeline (`src/train.py`, `src/evaluate.py`) across notebooks 02, 03 and 05.")

# ── Summary table + progression chart ───────────────────────────────────
st.subheader("Results summary")
summary = pd.DataFrame([
    {"key": "baseline",        "Model": "Majority-class baseline",  "Balanced Accuracy": 0.143, "AUC (macro)": None,  "Cohen's Kappa": None},
    {"key": "logreg",          "Model": "Logistic Regression",      "Balanced Accuracy": 0.318, "AUC (macro)": None,  "Cohen's Kappa": None},
    {"key": "simple_cnn",      "Model": "Simple CNN (3-layer)",     "Balanced Accuracy": 0.334, "AUC (macro)": None,  "Cohen's Kappa": None},
    {"key": "vit_b16",         "Model": "ViT-B/16",                 "Balanced Accuracy": 0.702, "AUC (macro)": 0.923, "Cohen's Kappa": 0.467},
    {"key": "efficientnet_b4", "Model": "EfficientNet-B4",          "Balanced Accuracy": 0.727, "AUC (macro)": 0.956, "Cohen's Kappa": 0.740},
    {"key": "multimodal",      "Model": "MultiModal Fusion",        "Balanced Accuracy": 0.720, "AUC (macro)": 0.920, "Cohen's Kappa": 0.559},
    {"key": "ensemble",        "Model": "Ensemble (3 models)",      "Balanced Accuracy": 0.794, "AUC (macro)": 0.958, "Cohen's Kappa": 0.706},
])

left, right = st.columns([1.1, 1], gap="large")
with left:
    st.dataframe(
        summary.drop(columns="key").set_index("Model"),
        column_config={
            "Balanced Accuracy": st.column_config.ProgressColumn(
                "Balanced Accuracy", format="%.3f", min_value=0, max_value=1),
            "AUC (macro)": st.column_config.NumberColumn("AUC (macro)", format="%.3f"),
            "Cohen's Kappa": st.column_config.NumberColumn("Cohen's Kappa", format="%.3f"),
        },
        width="stretch",
        height=280,
    )
    st.caption("A naive model that always predicts the majority class (`nv`) already "
               "scores 66.9% raw accuracy — which is why **balanced accuracy** (average "
               "per-class recall) is the metric that matters here, not raw accuracy.")

with right:
    fig = go.Figure(go.Bar(
        x=summary["Model"], y=summary["Balanced Accuracy"],
        marker_color=[MODEL_COLOR[k] for k in summary["key"]],
        text=[f"{v:.3f}" for v in summary["Balanced Accuracy"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 0.9], title="Balanced accuracy")
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(title="From naive baseline to ensemble")
    st.plotly_chart(style_chart(fig, height=380), width="stretch")

st.divider()

# ── Training curves (EfficientNet-B4) ───────────────────────────────────
st.subheader("Training curves — EfficientNet-B4")
history_path = RESULTS_DIR / "history_efficientnet_b4.json"
if history_path.exists():
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = list(range(1, len(history["train"]) + 1))

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=epochs, y=[h["loss"] for h in history["train"]], name="Train",
                         mode="lines", line={"color": MODEL_COLOR["efficientnet_b4"], "width": 2})
        fig.add_scatter(x=epochs, y=[h["loss"] for h in history["val"]], name="Validation",
                         mode="lines", line={"color": MODEL_COLOR["ensemble"], "width": 2})
        fig.update_xaxes(title="Epoch")
        fig.update_yaxes(title="Focal loss")
        fig.update_layout(title="Loss")
        st.plotly_chart(style_chart(fig, height=340, show_legend=True), width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=epochs, y=[h["balanced_accuracy"] for h in history["train"]], name="Train",
                         mode="lines", line={"color": MODEL_COLOR["efficientnet_b4"], "width": 2})
        fig.add_scatter(x=epochs, y=[h["balanced_accuracy"] for h in history["val"]], name="Validation",
                         mode="lines", line={"color": MODEL_COLOR["ensemble"], "width": 2})
        fig.update_xaxes(title="Epoch")
        fig.update_yaxes(title="Balanced accuracy", range=[0, 1])
        fig.update_layout(title="Balanced Accuracy")
        st.plotly_chart(style_chart(fig, height=340, show_legend=True), width="stretch")
else:
    st.info("`results/history_efficientnet_b4.json` not found.")

st.divider()

# ── Static plots from the notebooks ─────────────────────────────────────
st.subheader("Confusion matrices, ROC curves & saved training plots")


def show_if_exists(container, filename: str, caption: str):
    path = RESULTS_DIR / filename
    if path.exists():
        container.image(str(path), caption=caption, width="stretch")
    else:
        container.caption(f"_{filename} not found._")


tabs = st.tabs(["EfficientNet-B4", "ViT-B/16", "Ensemble", "MultiModal Fusion",
                "ResNet-50+CBAM / DenseNet-169", "Baselines"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    show_if_exists(c1, "cm_efficientnet_b4.png", "Confusion matrix")
    show_if_exists(c2, "roc_efficientnet_b4.png", "ROC curves (one-vs-rest)")
    show_if_exists(c3, "training_efficientnet_b4.png", "Training curves (saved)")

with tabs[1]:
    c1, c2, c3 = st.columns(3)
    show_if_exists(c1, "cm_vit_b16.png", "Confusion matrix")
    show_if_exists(c2, "roc_vit_b16.png", "ROC curves (one-vs-rest)")
    show_if_exists(c3, "training_vit_b16.png", "Training curves (saved)")

with tabs[2]:
    c1, c2 = st.columns(2)
    show_if_exists(c1, "cm_ensemble.png", "Confusion matrix")
    show_if_exists(c2, "roc_ensemble.png", "ROC curves (one-vs-rest)")
    st.caption("Soft-voting average of EfficientNet-B4, ResNet-50+CBAM and DenseNet-169 — "
               "the best result in the project (+6.7% balanced accuracy over the best "
               "single model).")

with tabs[3]:
    c1, c2 = st.columns(2)
    show_if_exists(c1, "cm_multimodal.png", "Confusion matrix")
    c2.caption("No dedicated ROC plot was saved for this model in the notebooks.")

with tabs[4]:
    st.info("These two models were trained and evaluated **only as ensemble members** "
            "in notebook 03 — the notebooks did not export standalone confusion "
            "matrices or ROC curves for them individually. You can still run either one "
            "by itself on the **Predict** page.")

with tabs[5]:
    show_if_exists(st, "baseline_comparison.png",
                    "Majority-class / Logistic Regression / Simple CNN vs. deep models "
                    "(notebook 05)")
