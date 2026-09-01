"""Project Q&A page — renders PROJECT_QA_DOCUMENT.md as browsable sections."""

import streamlit as st

from src.inference import ROOT_DIR

st.title("❓ Project Q&A")
st.caption("The project's own viva / presentation document, rendered directly from "
           "`PROJECT_QA_DOCUMENT.md` so this page never drifts out of sync with it.")

doc_path = ROOT_DIR / "PROJECT_QA_DOCUMENT.md"
if not doc_path.exists():
    st.error("PROJECT_QA_DOCUMENT.md not found at the project root.")
    st.stop()

text = doc_path.read_text(encoding="utf-8")
sections = [s.strip() for s in text.split("\n---\n") if s.strip()]

# First block is the title/preamble — render as-is, above the expanders.
st.markdown(sections[0])
st.divider()

for section in sections[1:]:
    lines = section.splitlines()
    heading = lines[0].lstrip("#").strip() if lines[0].startswith("#") else lines[0]
    body = "\n".join(lines[1:]) if lines[0].startswith("#") else section
    with st.expander(heading, expanded=False):
        st.markdown(body)
