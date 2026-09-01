"""Predict page — classify an uploaded or sample dermoscopic image."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from app_common import (
    CLASS_COLOR,
    DISCLAIMER,
    available_models,
    get_cached_device,
    get_ensemble_models,
    get_metadata_schema_cached,
    get_model,
    get_sample_catalog,
    resolve_image_path,
    status_badge,
    style_chart,
)
from src.dataset import CLASS_NAMES
from src.inference import (
    CLASS_FULL_NAMES,
    ENSEMBLE_LABEL,
    MODEL_REGISTRY,
    build_metadata_vector,
    predict_ensemble,
    predict_single,
    preprocess_image,
    sorted_prediction,
)

st.title("🔬 Predict")
st.caption("Classify a dermoscopic lesion image with any of the five trained models, "
           "or the soft-voting ensemble.")
st.warning(DISCLAIMER, icon="🎓")

models_info = available_models()
if not models_info:
    st.error("No checkpoints found under `checkpoints/`. Train a model first "
              "(see the notebooks/) or restore the .pth files.")
    st.stop()

st.session_state.setdefault("input_age", 45)
st.session_state.setdefault("input_sex", "male")
st.session_state.setdefault("input_loc", None)

col_image, col_config = st.columns([1, 1], gap="large")

# ── Left: image source ─────────────────────────────────────────────────
with col_image:
    st.markdown("#### 1. Choose an image")
    tab_upload, tab_sample = st.tabs(["📤 Upload", "🎲 Sample from HAM10000"])

    with tab_upload:
        uploaded = st.file_uploader("Dermoscopic image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            st.session_state["stage_image"] = Image.open(uploaded).convert("RGB")
            st.session_state["stage_true_label"] = None
            st.session_state["stage_image_id"] = uploaded.name

    with tab_sample:
        st.caption("Pulled from the actual HAM10000 training data, with its real "
                    "diagnosis and patient metadata, so you can check the model against "
                    "a known answer.")
        class_pick = st.selectbox(
            "Diagnosis class", ["Random"] + CLASS_NAMES,
            format_func=lambda c: "🎲 Random class" if c == "Random"
            else f"{c} — {CLASS_FULL_NAMES[c]}",
        )
        if st.button("Load a sample image", width="stretch"):
            catalog = get_sample_catalog()
            pool = catalog if class_pick == "Random" else catalog[catalog["dx"] == class_pick]
            row = pool.sample(1).iloc[0]
            path = resolve_image_path(row["image_id"])
            if path is None:
                st.error(f"Image file for `{row['image_id']}` not found under `dataset/`.")
            else:
                st.session_state["stage_image"] = Image.open(path).convert("RGB")
                st.session_state["stage_true_label"] = row["dx"]
                st.session_state["stage_image_id"] = row["image_id"]
                st.session_state["input_age"] = 45 if pd.isna(row["age"]) else int(row["age"])
                st.session_state["input_sex"] = row["sex"] if row["sex"] in ("male", "female") else "unknown"
                st.session_state["input_loc"] = row["localization"]

    if "stage_image" in st.session_state:
        st.image(st.session_state["stage_image"], caption=st.session_state.get("stage_image_id"),
                  width="stretch")
    else:
        st.info("Upload an image or load a sample to get started.")

# ── Right: model + metadata ────────────────────────────────────────────
with col_config:
    st.markdown("#### 2. Choose a model")
    order = ["ensemble", "efficientnet_b4", "vit_b16", "resnet50_cbam", "densenet169", "multimodal"]
    options = [k for k in order if k == "ensemble" or k in models_info]

    def _label(key: str) -> str:
        return ENSEMBLE_LABEL if key == "ensemble" else models_info[key]["label"]

    model_choice = st.selectbox("Model", options, format_func=_label)
    if model_choice == "ensemble":
        st.caption("Best result in the project (balanced accuracy 0.794). Averages the "
                    "softmax probabilities of three architecturally different CNNs.")
    else:
        st.caption(models_info[model_choice]["blurb"])

    meta_cols, localizations = get_metadata_schema_cached()
    if st.session_state["input_loc"] not in localizations:
        st.session_state["input_loc"] = localizations[0]

    st.markdown("#### 3. Patient context")
    is_multimodal = model_choice == "multimodal"
    st.caption("Used **only** by the MultiModal Fusion model — every other model here "
               "is image-only." if not is_multimodal else
               "This model fuses these fields with the image, per HAM10000_metadata.csv.")
    age = st.slider("Age", 0, 85, key="input_age")
    sex = st.radio("Sex", ["male", "female", "unknown"], key="input_sex", horizontal=True)
    localization = st.selectbox("Lesion site", localizations, key="input_loc")

    run = st.button("🔍 Classify", type="primary", width="stretch",
                     disabled="stage_image" not in st.session_state)

# ── Inference ───────────────────────────────────────────────────────────
if run:
    image = st.session_state["stage_image"]
    device = get_cached_device()
    with st.spinner(f"Running {_label(model_choice)}…"):
        if model_choice == "ensemble":
            models = get_ensemble_models()
            probs, per_model_probs = predict_ensemble(models, image, device)
        else:
            info = MODEL_REGISTRY[model_choice]
            metadata_vector = None
            if info["multimodal"]:
                metadata_vector = build_metadata_vector(age, sex, localization, meta_cols)
                model = get_model(model_choice, metadata_dim=len(meta_cols))
            else:
                model = get_model(model_choice)
            tensor = preprocess_image(image, info["image_size"])
            probs = predict_single(model, tensor, device, metadata_vector=metadata_vector)
            per_model_probs = None

    st.session_state["result_probs"] = probs
    st.session_state["result_per_model"] = per_model_probs
    st.session_state["result_model_choice"] = model_choice
    st.session_state["result_image"] = image
    st.session_state["result_image_id"] = st.session_state.get("stage_image_id")
    st.session_state["result_true_label"] = st.session_state.get("stage_true_label")
    st.session_state["result_metadata"] = {"age": age, "sex": sex, "localization": localization}

# ── Results ─────────────────────────────────────────────────────────────
if "result_probs" in st.session_state:
    st.divider()
    st.subheader("Result")

    probs = st.session_state["result_probs"]
    ranked = sorted_prediction(probs)
    top_class, top_prob = ranked[0]

    res_left, res_right = st.columns([1, 1.4], gap="large")
    with res_left:
        st.image(st.session_state["result_image"], width="stretch")
        true_label = st.session_state["result_true_label"]
        if true_label:
            correct = true_label == top_class
            st.markdown(
                f"{'✅ Correct' if correct else '❌ Missed'} — ground truth (sample image): "
                f"**{true_label} · {CLASS_FULL_NAMES[true_label]}**"
            )
        else:
            st.caption("Ground truth unknown (user-uploaded image).")

    with res_right:
        m1, m2 = st.columns(2)
        m1.metric("Predicted class", f"{top_class} · {CLASS_FULL_NAMES[top_class]}")
        m2.metric("Confidence", f"{top_prob:.1%}")
        st.markdown(status_badge(top_class))

        codes = [c for c, _ in ranked]
        names = [CLASS_FULL_NAMES[c] for c in codes]
        values = [p for _, p in ranked]
        colors = [CLASS_COLOR[c] for c in codes]

        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig.update_xaxes(tickformat=".0%", range=[0, max(values) * 1.18])
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Predicted probability by class")
        st.plotly_chart(style_chart(fig, height=320), width="stretch")

    if st.session_state["result_per_model"] is not None:
        with st.expander("Per-model breakdown (ensemble members)"):
            per_model = st.session_state["result_per_model"]
            rows = {MODEL_REGISTRY[name]["label"]: p for name, p in per_model.items()}
            rows["Ensemble (average)"] = probs
            table = pd.DataFrame(rows, index=CLASS_NAMES).T
            st.dataframe(
                table,
                column_config={
                    c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0, max_value=1)
                    for c in CLASS_NAMES
                },
                width="stretch",
            )

    if MODEL_REGISTRY.get(st.session_state["result_model_choice"], {}).get("supports_xai", False):
        st.page_link("views/explain.py", label="Explain this prediction with GradCAM →", icon="🧠")
    elif st.session_state["result_model_choice"] == "ensemble":
        st.caption("💡 Tip: open **Explainability** and pick EfficientNet-B4, ResNet-50+CBAM "
                    "or DenseNet-169 individually to see GradCAM for an ensemble member.")
