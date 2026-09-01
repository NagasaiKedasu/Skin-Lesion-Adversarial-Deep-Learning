"""Explainability page — GradCAM / GradCAM++ / Integrated Gradients."""

import pandas as pd
import streamlit as st
from PIL import Image

from app_common import (
    DISCLAIMER,
    available_models,
    get_cached_device,
    get_metadata_schema_cached,
    get_model,
    get_sample_catalog,
    resolve_image_path,
    status_badge,
)
from src.dataset import CLASS_NAMES
from src.inference import (
    CLASS_FULL_NAMES,
    MODEL_REGISTRY,
    build_metadata_vector,
    generate_explanations,
    predict_single,
    preprocess_image,
    sorted_prediction,
)

st.title("🧠 Explainability")
st.caption("GradCAM, GradCAM++ and Integrated Gradients show *where* a convolutional "
           "model is looking when it makes a prediction — used in this project to check "
           "the model focuses on the lesion itself, not hair or background skin (see "
           "Project Q&A, Q1 and Q10).")
st.warning(DISCLAIMER, icon="🎓")

xai_models = {k: v for k, v in available_models().items() if v["supports_xai"]}
if not xai_models:
    st.error("No XAI-capable checkpoints are available.")
    st.stop()

# ── Image source (shared with the Predict page's current case) ─────────
if "stage_image" in st.session_state and st.session_state.get("result_model_choice") in xai_models:
    st.success("Reusing the image from your last **Predict** run. Use "
               "'Change image' below to pick a different one.", icon="🔁")

with st.expander("📷 Change image", expanded="stage_image" not in st.session_state):
    tab_upload, tab_sample = st.tabs(["📤 Upload", "🎲 Sample from HAM10000"])
    with tab_upload:
        uploaded = st.file_uploader("Dermoscopic image (JPG/PNG)", type=["jpg", "jpeg", "png"],
                                     key="xai_uploader")
        if uploaded is not None:
            st.session_state["stage_image"] = Image.open(uploaded).convert("RGB")
            st.session_state["stage_true_label"] = None
            st.session_state["stage_image_id"] = uploaded.name
    with tab_sample:
        class_pick = st.selectbox(
            "Diagnosis class", ["Random"] + CLASS_NAMES,
            format_func=lambda c: "🎲 Random class" if c == "Random"
            else f"{c} — {CLASS_FULL_NAMES[c]}",
            key="xai_class_pick",
        )
        if st.button("Load a sample image", width="stretch", key="xai_load_sample"):
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

if "stage_image" not in st.session_state:
    st.info("Upload an image or load a sample above to get started.")
    st.stop()

image = st.session_state["stage_image"]

# ── Model + metadata ─────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.image(image, caption=st.session_state.get("stage_image_id"), width="stretch")

with col_right:
    order = ["efficientnet_b4", "resnet50_cbam", "densenet169", "multimodal"]
    options = [k for k in order if k in xai_models]
    default_idx = options.index(st.session_state["result_model_choice"]) \
        if st.session_state.get("result_model_choice") in options else 0
    model_choice = st.selectbox("Model", options, index=default_idx,
                                 format_func=lambda k: xai_models[k]["label"])
    st.caption(xai_models[model_choice]["blurb"] + " ViT-B/16 is excluded here — GradCAM "
               "needs a spatial convolutional feature map, which a pure transformer's "
               "patch embeddings don't provide.")

    meta_cols, localizations = get_metadata_schema_cached()
    st.session_state.setdefault("input_age", 45)
    st.session_state.setdefault("input_sex", "male")
    st.session_state.setdefault("input_loc", localizations[0])
    if st.session_state["input_loc"] not in localizations:
        st.session_state["input_loc"] = localizations[0]

    metadata_vector = None
    if model_choice == "multimodal":
        st.caption("Patient context (fused with the image by this model):")
        mcol1, mcol2, mcol3 = st.columns(3)
        age = mcol1.number_input("Age", 0, 85, key="input_age")
        sex = mcol2.selectbox("Sex", ["male", "female", "unknown"], key="input_sex")
        localization = mcol3.selectbox("Site", localizations, key="input_loc")
        metadata_vector = build_metadata_vector(age, sex, localization, meta_cols)

    device = get_cached_device()
    info = MODEL_REGISTRY[model_choice]
    model = get_model(model_choice, metadata_dim=len(meta_cols) if info["multimodal"] else None)
    tensor = preprocess_image(image, info["image_size"])
    probs = predict_single(model, tensor, device, metadata_vector=metadata_vector)
    ranked = sorted_prediction(probs)
    top_class, top_prob = ranked[0]

    st.metric("Model's prediction", f"{top_class} · {CLASS_FULL_NAMES[top_class]}", f"{top_prob:.1%}")
    st.markdown(status_badge(top_class))

    class_choice = st.selectbox(
        "Explain the evidence for class:", CLASS_NAMES,
        index=CLASS_NAMES.index(top_class),
        format_func=lambda c: f"{c} — {CLASS_FULL_NAMES[c]}"
        + ("  (predicted)" if c == top_class else ""),
    )
    ig_steps = st.select_slider("Integrated Gradients steps (higher = smoother, slower)",
                                 options=[16, 32, 50], value=32)
    generate = st.button("🔥 Generate explanations", type="primary", width="stretch")

# ── Explanations ──────────────────────────────────────────────────────────
if generate:
    class_idx = CLASS_NAMES.index(class_choice)
    with st.spinner("Running GradCAM, GradCAM++ and Integrated Gradients…"):
        maps = generate_explanations(
            model_choice, model, tensor, device, class_idx,
            metadata_vector=metadata_vector, ig_steps=ig_steps,
        )

    st.divider()
    st.subheader(f"Evidence for “{CLASS_FULL_NAMES[class_choice]}”")
    panels = [
        ("Original (model input)", maps["original"],
         "The 224×224 (or 380×380) crop the model actually saw, after de-normalisation."),
        ("GradCAM", maps["gradcam"],
         "Gradient-weighted activation map from the last conv block — coarse but reliable "
         "localisation (Selvaraju et al., 2017)."),
        ("GradCAM++", maps["gradcam_pp"],
         "Second-order gradient weighting — sharper boundaries, better with multiple "
         "instances of the pattern (Chattopadhay et al., 2018)."),
        ("Integrated Gradients", maps["integrated_gradients"],
         "Pixel-level attribution integrated along a path from a black baseline to the "
         "image — axiomatic, more precise than GradCAM (Sundararajan et al., 2017)."),
    ]
    cols = st.columns(4)
    for col, (title, arr, desc) in zip(cols, panels):
        with col:
            st.image(arr, width="stretch")
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.caption("🔴 warm colors = regions that pushed the prediction toward the selected "
               "class · 🔵 cool colors = regions that mattered least.")
