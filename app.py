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

MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]

EXAMPLE_QUESTIONS = [
    "What spatial resolution does SWOT achieve for SSH?",
    "How does KaRIn reduce instrument noise?",
    "What are the key findings on abyssal marine tectonics from SWOT?",
    "How does SWOT compare to conventional nadir altimetry?",
    "What ocean features can SWOT detect that previous satellites couldn't?",
]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ SWOT RAG")
    st.markdown(
        "Ask questions about SWOT satellite research. "
        "Answers are grounded in indexed papers with citations."
    )
    st.divider()

    index_path = st.text_input("FAISS index path", value="faiss_index")

    st.divider()
    st.markdown("**Example questions**")
    for q_text in EXAMPLE_QUESTIONS:
        if st.button(q_text, use_container_width=True):
            st.session_state["selected_q"] = q_text

# ── Check index ───────────────────────────────────────────────────────────────
index_exists = Path(index_path).exists()
if not index_exists:
    st.warning(
        f"No FAISS index found at `{index_path}/`. "
        "Run `python ingest.py --pdf_dir ./papers` first."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.header("SWOT GeoScience Research Assistant")
st.caption("Retrieval-augmented generation over SWOT satellite papers")

tab_ask, tab_compare = st.tabs(["Ask", "Compare variants"])


# ── Ask tab ───────────────────────────────────────────────────────────────────
with tab_ask:
    with st.expander("Pipeline settings", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            k_retrieve = st.slider("Candidates (k_retrieve)", 5, 40, 20, key="ask_kr")
        with col_s2:
            k = st.slider("Chunks to LLM (k)", 2, 10, 5, key="ask_k")
        with col_s3:
            model = st.selectbox("Model", MODELS, key="ask_model")
        use_rerank = st.checkbox("Use cross-encoder re-ranker", value=True, key="ask_rr")

    default_q = st.session_state.get("selected_q", "")
    question = st.text_area(
        "Your question",
        value=default_q,
        height=80,
        placeholder="e.g. What is the noise level of SWOT SSH measurements over the open ocean?",
        key="ask_q",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Ask", type="primary", disabled=not index_exists)

    if submit and question.strip():
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                result: RAGResponse = query(
                    question=question.strip(),
                    index_path=index_path,
                    k=k,
                    k_retrieve=k_retrieve,
                    model=model,
                    use_rerank=use_rerank,
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.subheader("Answer")
        st.markdown(result.answer)

        st.subheader("Retrieved sources")
        for i, source in enumerate(result.sources, 1):
            with st.expander(f"[{i}] {source['source_file']}  —  page {source['page']}"):
                st.caption("Snippet from retrieved chunk:")
                st.markdown(f"> {source['snippet']}")

        with st.expander("Raw retrieved chunks (debug)"):
            from rag_engine import load_retriever, format_context, rerank as do_rerank
            retriever = load_retriever(index_path, k_retrieve if use_rerank else k)
            raw_docs = retriever.invoke(question)
            if use_rerank:
                raw_docs = do_rerank(question, raw_docs, top_k=k)
            else:
                raw_docs = raw_docs[:k]
            st.text(format_context(raw_docs))


# ── Compare tab ───────────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Side-by-side variant comparison")
    st.caption(
        "Run the same question through two different pipeline configurations "
        "and compare answers."
    )

    col_cfg_a, col_cfg_b = st.columns(2)

    with col_cfg_a:
        st.markdown("#### Variant A")
        model_a      = st.selectbox("Model", MODELS, index=0, key="cmp_model_a")
        k_a          = st.slider("Chunks to LLM (k)", 2, 10, 5, key="cmp_k_a")
        k_retrieve_a = st.slider("Candidates (k_retrieve)", 5, 40, 20, key="cmp_kr_a")
        rerank_a     = st.checkbox("Use reranker", value=True, key="cmp_rr_a")

    with col_cfg_b:
        st.markdown("#### Variant B")
        model_b      = st.selectbox("Model", MODELS, index=1, key="cmp_model_b")
        k_b          = st.slider("Chunks to LLM (k)", 2, 10, 5, key="cmp_k_b")
        k_retrieve_b = st.slider("Candidates (k_retrieve)", 5, 40, 20, key="cmp_kr_b")
        rerank_b     = st.checkbox("Use reranker", value=True, key="cmp_rr_b")

    st.divider()
    cmp_question = st.text_area(
        "Question to compare",
        height=80,
        placeholder="e.g. What spatial resolution does SWOT achieve for SSH?",
        key="cmp_q",
    )

    compare_btn = st.button("Compare", type="primary", disabled=not index_exists)

    if compare_btn and cmp_question.strip():
        col_a, col_b = st.columns(2)

        with col_a:
            with st.spinner("Running Variant A..."):
                try:
                    result_a: RAGResponse = query(
                        question=cmp_question.strip(),
                        index_path=index_path,
                        k=k_a,
                        k_retrieve=k_retrieve_a,
                        model=model_a,
                        use_rerank=rerank_a,
                    )
                except Exception as e:
                    st.error(f"Variant A error: {e}")
                    result_a = None

        with col_b:
            with st.spinner("Running Variant B..."):
                try:
                    result_b: RAGResponse = query(
                        question=cmp_question.strip(),
                        index_path=index_path,
                        k=k_b,
                        k_retrieve=k_retrieve_b,
                        model=model_b,
                        use_rerank=rerank_b,
                    )
                except Exception as e:
                    st.error(f"Variant B error: {e}")
                    result_b = None

        col_ans_a, col_ans_b = st.columns(2)

        with col_ans_a:
            st.markdown(f"**Variant A** — `{model_a}`, k={k_a}, rerank={rerank_a}")
            if result_a:
                st.markdown(result_a.answer)
                with st.expander("Sources A"):
                    for i, s in enumerate(result_a.sources, 1):
                        st.caption(f"[{i}] {s['source_file']} p.{s['page']}")
                        st.markdown(f"> {s['snippet']}")

        with col_ans_b:
            st.markdown(f"**Variant B** — `{model_b}`, k={k_b}, rerank={rerank_b}")
            if result_b:
                st.markdown(result_b.answer)
                with st.expander("Sources B"):
                    for i, s in enumerate(result_b.sources, 1):
                        st.caption(f"[{i}] {s['source_file']} p.{s['page']}")
                        st.markdown(f"> {s['snippet']}")
