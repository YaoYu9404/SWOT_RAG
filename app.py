"""
app.py — Streamlit UI for SWOT GeoScience RAG

Run with: streamlit run app.py
"""

import os
import streamlit as st
from pathlib import Path
from rag_engine import query, RAGResponse

st.set_page_config(
    page_title="SWOT GeoScience RAG",
    page_icon="🛰️",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ SWOT RAG")
    st.markdown(
        "Ask questions about SWOT satellite research. "
        "Answers are grounded in indexed papers with citations."
    )
    st.divider()

    index_path = st.text_input("FAISS index path", value="faiss_index")
    k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=10, value=5)
    model = st.selectbox("LLM model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    
    st.divider()
    st.markdown("**Example questions**")
    example_qs = [
        "What spatial resolution does SWOT achieve for SSH?",
        "How does KaRIn reduce instrument noise?",
        "What are the key findings on abyssal marine tectonics from SWOT?",
        "How does SWOT compare to conventional nadir altimetry?",
        "What ocean features can SWOT detect that previous satellites couldn't?",
    ]
    for q_text in example_qs:
        if st.button(q_text, use_container_width=True):
            st.session_state["selected_q"] = q_text

# ── Check index exists ────────────────────────────────────────────────────────
index_exists = Path(index_path).exists()
if not index_exists:
    st.warning(
        f"No FAISS index found at `{index_path}/`. "
        "Run `python ingest.py --pdf_dir ./papers` first."
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.header("SWOT GeoScience Research Assistant")
st.caption("Retrieval-augmented generation over SWOT satellite papers")

# Pre-fill from sidebar example buttons
default_q = st.session_state.get("selected_q", "")
question = st.text_area(
    "Your question",
    value=default_q,
    height=80,
    placeholder="e.g. What is the noise level of SWOT SSH measurements over the open ocean?",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask", type="primary", disabled=not index_exists)

# ── Query & display ───────────────────────────────────────────────────────────
if submit and question.strip():
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        try:
            result: RAGResponse = query(
                question=question.strip(),
                index_path=index_path,
                k=k,
                model=model,
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # Answer
    st.subheader("Answer")
    st.markdown(result.answer)

    # Sources
    st.subheader("Retrieved sources")
    for i, source in enumerate(result.sources, 1):
        with st.expander(f"[{i}] {source['source_file']}  —  page {source['page']}"):
            st.caption("Snippet from retrieved chunk:")
            st.markdown(f"> {source['snippet']}")

    # Debug: show retrieved context
    with st.expander("Show raw retrieved chunks (debug)"):
        from rag_engine import load_retriever, format_context
        retriever = load_retriever(index_path, k)
        docs = retriever.invoke(question)
        st.text(format_context(docs))
