import pandas as pd
import streamlit as st

from src.dataset import build_image_path_map
from src.inference import (
    DATA_ROOT,
    ENSEMBLE_MEMBERS,
    METADATA_CSV,
    MODEL_REGISTRY,
    checkpoint_exists,
    get_device,
    load_metadata_schema,
    load_model,
    risk_for_class,
)

# ─────────────────────────────────────────────────────────────────────────
# Palette — fixed categorical order per diagnosis class (validated: see
# dataviz skill references/palette.md, slots 1-7, light-surface set).
# ─────────────────────────────────────────────────────────────────────────
CLASS_COLOR = {
    "akiec": "#2a78d6",
    "bcc":   "#eb6834",
    "bkl":   "#1baf7a",
    "df":    "#eda100",
    "mel":   "#e87ba4",
    "nv":    "#008300",
    "vasc":  "#4a3aa7",
}
# Second fixed categorical order, reused for a different dimension (models,
# not diagnosis classes) — same validated hue set, own assignment.
MODEL_COLOR = {
    "baseline":        "#2a78d6",
    "logreg":          "#eb6834",
    "simple_cnn":      "#1baf7a",
    "vit_b16":         "#eda100",
    "efficientnet_b4": "#e87ba4",
    "multimodal":      "#008300",
    "ensemble":        "#4a3aa7",
}
STATUS_COLOR = {"benign": "#0ca30c", "pre-malignant": "#fab219", "malignant": "#d03b3b"}
STATUS_ICON  = {"benign": "🟢", "pre-malignant": "🟠", "malignant": "🔴"}
STATUS_TEXT = {
    "benign": "Benign",
    "pre-malignant": "Pre-malignant — biopsy / monitoring advised",
    "malignant": "Malignant — see a dermatologist",
}
CHART_BG   = "#fcfcfb"
GRID_COLOR = "#e1e0d9"
INK        = "#0b0b0b"
MUTED_INK  = "#52514e"
FONT_STACK = "system-ui, -apple-system, 'Segoe UI', sans-serif"

DISCLAIMER = (
    "🎓 **Research / educational demo only.** These models were trained on the "
    "public HAM10000 research dataset for a coursework project. They are "
    "**not a certified diagnostic device** and must never replace a "
    "dermatologist or biopsy. See the [Project Q&A](/about) page for details."
)


# ─────────────────────────────────────────────────────────────────────────
# Cached, expensive resources — loaded once per server process
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_device():
    return get_device()


@st.cache_resource(show_spinner="Loading model weights…")
def get_model(model_name: str, metadata_dim: int | None = None):
    device = get_cached_device()
    return load_model(model_name, device, metadata_dim=metadata_dim)


@st.cache_resource(show_spinner=False)
def get_metadata_schema_cached():
    return load_metadata_schema()


@st.cache_resource(show_spinner="Indexing dataset images…")
def get_image_path_map():
    return build_image_path_map(str(DATA_ROOT))


@st.cache_data(show_spinner=False)
def get_metadata_df() -> pd.DataFrame:
    return pd.read_csv(METADATA_CSV)


@st.cache_data(show_spinner=False)
def get_sample_catalog(n_per_class: int = 6, seed: int = 42) -> pd.DataFrame:
    """A small, reproducible sample of labelled images per class for the
    'try a sample image' flow — avoids scanning the full 10k-image folder
    on every rerun."""
    df = get_metadata_df()
    parts = [g.sample(min(n_per_class, len(g)), random_state=seed)
             for _, g in df.groupby("dx")]
    return pd.concat(parts, ignore_index=True)


def available_models() -> dict:
    """MODEL_REGISTRY entries whose checkpoint file actually exists on disk."""
    return {name: info for name, info in MODEL_REGISTRY.items() if checkpoint_exists(name)}


def get_ensemble_models() -> dict:
    return {name: get_model(name) for name in ENSEMBLE_MEMBERS}


def resolve_image_path(image_id: str) -> str | None:
    return get_image_path_map().get(image_id)



def style_chart(fig, height: int = 420, show_legend: bool = False):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART_BG,
        font={"color": INK, "family": FONT_STACK, "size": 13},
        margin={"l": 10, "r": 20, "t": 40, "b": 10},
        height=height,
        showlegend=show_legend,
        legend={"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "y": 1.08, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=MUTED_INK)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=MUTED_INK)
    return fig


def status_badge(class_name: str) -> str:
    """Markdown for an icon + text clinical status badge (never color alone)."""
    risk = risk_for_class(class_name)
    return f"{STATUS_ICON[risk]} **{STATUS_TEXT[risk]}**"
