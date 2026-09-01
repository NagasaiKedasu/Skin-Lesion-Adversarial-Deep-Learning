import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import CLASS_COLOR, get_metadata_df, get_sample_catalog, resolve_image_path, style_chart
from src.inference import CLASS_FULL_NAMES, ROOT_DIR

RESULTS_DIR = ROOT_DIR / "results"
SEX_COLOR = {"male": "#2a78d6", "female": "#e87ba4", "unknown": "#898781"}

st.title("🗂️ Dataset Explorer")
st.caption("HAM10000 (Human Against Machine with 10,000 training images) — ISIC Archive, "
           "Tschandl, Rosendahl & Kittler (2018). All charts below are computed live from "
           "`dataset/HAM10000_metadata.csv`.")

df = get_metadata_df()

# ── Headline stats ────────────────────────────────────────────────────────
n_images = len(df)
n_lesions = df["lesion_id"].nunique()
lesion_counts = df.groupby("lesion_id").size()
pct_multi = (lesion_counts > 1).mean() * 100
class_counts = df["dx"].value_counts()
majority_cls, majority_n = class_counts.idxmax(), class_counts.max()
minority_cls, minority_n = class_counts.idxmin(), class_counts.min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Images", f"{n_images:,}")
c2.metric("Unique lesions", f"{n_lesions:,}")
c3.metric("Lesions with >1 photo", f"{pct_multi:.1f}%",
          help="Why the project uses a lesion-aware train/val/test split instead of a "
               "random image-level split — otherwise the same lesion could leak across "
               "splits and inflate test accuracy by 5–10%.")
c4.metric("Class imbalance ratio", f"{majority_n / minority_n:.0f}×",
          help=f"{majority_cls} ({majority_n:,}) vs {minority_cls} ({minority_n:,})")

st.divider()

# ── Class distribution (live) ────────────────────────────────────────────
st.subheader("Class distribution")
log_scale = st.checkbox("Log scale (rare classes are otherwise invisible next to `nv`)")

order = class_counts.sort_values(ascending=False)
fig = go.Figure(go.Bar(
    x=[f"{c} — {CLASS_FULL_NAMES[c]}" for c in order.index],
    y=order.values,
    marker_color=[CLASS_COLOR[c] for c in order.index],
    text=[f"{v:,} ({v / n_images:.1%})" for v in order.values],
    textposition="outside",
    cliponaxis=False,
    hovertemplate="%{x}: %{y:,}<extra></extra>",
))
fig.update_yaxes(type="log" if log_scale else "linear", title="Images")
fig.update_xaxes(tickangle=-20)
fig.update_layout(title="Images per diagnosis class")
st.plotly_chart(style_chart(fig, height=420), width="stretch")

st.divider()

# ── Metadata breakdown (live) ────────────────────────────────────────────
st.subheader("Patient metadata")
m1, m2, m3 = st.columns(3)

with m1:
    ages = df["age"].dropna()
    fig = go.Figure(go.Histogram(x=ages, nbinsx=20, marker_color="#2a78d6"))
    fig.update_xaxes(title="Age")
    fig.update_yaxes(title="Images")
    fig.update_layout(title=f"Age (median {ages.median():.0f})")
    st.plotly_chart(style_chart(fig, height=320), width="stretch")

with m2:
    sex_counts = df["sex"].value_counts()
    fig = go.Figure(go.Bar(
        x=sex_counts.index, y=sex_counts.values,
        marker_color=[SEX_COLOR.get(s, "#898781") for s in sex_counts.index],
        text=[f"{v / n_images:.0%}" for v in sex_counts.values],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_yaxes(title="Images")
    fig.update_layout(title="Sex")
    st.plotly_chart(style_chart(fig, height=320), width="stretch")

with m3:
    loc_counts = df["localization"].value_counts()
    fig = go.Figure(go.Bar(
        x=loc_counts.values, y=loc_counts.index, orientation="h",
        marker_color="#2a78d6",
    ))
    fig.update_xaxes(title="Images")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Lesion site")
    st.plotly_chart(style_chart(fig, height=320), width="stretch")

st.caption("Body-site bars share one neutral hue — 15 categories is well past the point "
           "where distinct per-bar colors stay tell-apart-able, so magnitude (bar length) "
           "carries the comparison instead of color identity.")

st.divider()

# ── Sample gallery ────────────────────────────────────────────────────────
st.subheader("Image gallery")
gallery_class = st.selectbox(
    "Filter by class", ["All"] + list(CLASS_FULL_NAMES.keys()),
    format_func=lambda c: "All classes" if c == "All" else f"{c} — {CLASS_FULL_NAMES[c]}",
)
catalog = get_sample_catalog(n_per_class=8)
pool = catalog if gallery_class == "All" else catalog[catalog["dx"] == gallery_class]
pool = pool.sample(min(12, len(pool)), random_state=7)

cols = st.columns(6)
for i, (_, row) in enumerate(pool.iterrows()):
    path = resolve_image_path(row["image_id"])
    if path is None:
        continue
    with cols[i % 6]:
        st.image(path, width="stretch")
        age_part = f"{row['age']:.0f}{row['sex'][0]} · " if pd.notna(row["age"]) else ""
        st.caption(f"{row['dx']} · {age_part}{row['localization']}")

st.divider()
with st.expander("📎 Original notebook EDA figures (static, from `notebooks/01_EDA_and_Analysis.ipynb`)"):
    e1, e2 = st.columns(2)
    for fname, cap, col in [
        ("eda_class_distribution.png", "Class distribution", e1),
        ("eda_metadata.png", "Age / sex / localization", e2),
    ]:
        path = RESULTS_DIR / fname
        if path.exists():
            col.image(str(path), caption=cap, width="stretch")
    gallery_path = RESULTS_DIR / "eda_sample_gallery.png"
    if gallery_path.exists():
        st.image(str(gallery_path), caption="Sample gallery", width="stretch")
