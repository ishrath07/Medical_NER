"""
app.py

Phase 5 — Streamlit demo for the Medical NER project.
Upload a .txt or .pdf → run inference with the fine-tuned BioBERT model →
highlight entities inline using spaCy's displacy renderer.
"""

import streamlit as st
from transformers import pipeline
from spacy import displacy
from pypdf import PdfReader

MODEL_DIR = "models/biobert-ncbi-disease-subset"

ENTITY_COLORS = {
    "Disease": "#ff6961",
}


@st.cache_resource
def load_ner_pipeline():
    return pipeline(
        "ner",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        aggregation_strategy="simple",  # merges subword pieces into clean spans
    )


def extract_text(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return uploaded_file.read().decode("utf-8")


def run_ner(text: str, ner_pipeline):
    raw_entities = ner_pipeline(text)
    ents = []
    for ent in raw_entities:
        ents.append({
            "start": ent["start"],
            "end": ent["end"],
            "label": ent["entity_group"],
        })
    return ents


def render_highlighted(text: str, ents: list) -> str:
    doc_data = {"text": text, "ents": ents, "title": None}
    html = displacy.render(
        doc_data,
        style="ent",
        manual=True,
        options={"colors": ENTITY_COLORS},
    )
    return html


# --- UI ---
st.set_page_config(page_title="Medical Entity Extractor", layout="wide")
st.title("🩺 Medical / Legal Entity Extractor (NER)")
st.caption("Upload a clinical text document to extract and highlight disease mentions.")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

if uploaded_file is not None:
    with st.spinner("Loading model..."):
        ner_pipeline = load_ner_pipeline()

    text = extract_text(uploaded_file)

    if not text.strip():
        st.warning("Couldn't extract any text from this file.")
    else:
        with st.spinner("Extracting entities..."):
            ents = run_ner(text, ner_pipeline)

        st.subheader("Highlighted entities")
        if ents:
            html = render_highlighted(text, ents)
            st.components.v1.html(html, height=500, scrolling=True)
        else:
            st.info("No entities detected in this document.")
            st.text(text)

        st.subheader("Extracted entities (table)")
        if ents:
            st.dataframe(
                [{"Text": text[e["start"]:e["end"]], "Label": e["label"],
                  "Start": e["start"], "End": e["end"]} for e in ents],
                use_container_width=True,
            )
else:
    st.info("Upload a .txt or .pdf file to get started.")