import sys
from pathlib import Path

import streamlit as st

# Make the project root importable (for `app_common`, `src.*`) regardless of
# the working directory `streamlit run` was launched from, and regardless of
# how Streamlit sets up sys.path when it later execs each view/*.py page.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="HAM10000 Skin Lesion Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("views/home.py", title="Home", icon="🩺", default=True)
predict = st.Page("views/predict.py", title="Predict", icon="🔬")
explain = st.Page("views/explain.py", title="Explainability", icon="🧠")
performance = st.Page("views/performance.py", title="Model Performance", icon="📊")
explorer = st.Page("views/dataset_explorer.py", title="Dataset Explorer", icon="🗂️")
about = st.Page("views/about.py", title="Project Q&A", icon="❓")

pg = st.navigation(
    {
        "Overview": [home],
        "Try it": [predict, explain],
        "Evidence": [performance, explorer],
        "Reference": [about],
    }
)

with st.sidebar:
    st.caption("HAM10000 · 7-class skin lesion classification · PyTorch")

pg.run()
