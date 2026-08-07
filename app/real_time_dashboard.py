"""Streamlit UI for interactive categorization, tags and runtime metrics."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.inference import DocumentCategorizationPipeline

st.set_page_config(page_title="Document Categorization", page_icon="📄", layout="wide")
st.title("📄 Intelligent Document Categorization & Tagging")
st.caption("Multilingual DistilBERT classification + spaCy context-aware tagging")

@st.cache_resource
def load_pipeline():
    return DocumentCategorizationPipeline(ROOT / "models/checkpoints")

@st.cache_data
def load_metrics():
    path = ROOT / "reports/performance_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None

@st.cache_data
def load_examples():
    path = ROOT / "reports/example_predictions.csv"
    return pd.read_csv(path) if path.exists() else None

metrics = load_metrics()
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['classification_accuracy']:.1%}")
    c2.metric("Macro F1", f"{metrics['f1_score_macro']:.3f}")
    c3.metric("Throughput", f"{metrics['processing_speed_docs_per_sec']:.1f} docs/s")
    c4.metric("Languages", len(metrics["languages_supported"]))
else:
    st.info("Performance report will appear after `python scripts/evaluate.py`.")

st.subheader("Real-time categorization")
text = st.text_area("Document text", height=220, placeholder="Paste an English or Spanish document here...")
if st.button("Categorize", type="primary"):
    if not text.strip():
        st.warning("Enter document text first.")
    else:
        try:
            prediction = load_pipeline().process(text)
        except Exception as exc:
            st.error(str(exc))
        else:
            left, right = st.columns([1, 2])
            with left:
                st.metric("Category", prediction.category)
                st.metric("Confidence", f"{prediction.confidence:.1%}")
                st.metric("Language", prediction.language.upper())
            with right:
                st.write("**Tags**")
                st.write(", ".join(prediction.tags) if prediction.tags else "No tags")
                st.write("**Named entities**")
                st.dataframe(pd.DataFrame(prediction.entities), use_container_width=True, hide_index=True)

st.subheader("Monitoring")
examples = load_examples()
if examples is not None and not examples.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Category distribution**")
        st.bar_chart(examples["predicted_category"].value_counts())
        st.write("**Language breakdown**")
        st.bar_chart(examples["language"].value_counts())
    with col2:
        tag_counts = Counter()
        for value in examples["tags"].fillna(""):
            tag_counts.update(tag for tag in str(value).split("|") if tag)
        st.write("**Top tags**")
        st.bar_chart(pd.Series(dict(tag_counts.most_common(15)), name="count"))
        if metrics:
            st.write("**Language-specific accuracy**")
            st.bar_chart(pd.Series(metrics["per_language_accuracy"], name="accuracy"))
    st.write("**Example predictions**")
    st.dataframe(examples.head(25), use_container_width=True, hide_index=True)
else:
    st.info("Monitoring charts use `reports/example_predictions.csv`, generated during evaluation.")
