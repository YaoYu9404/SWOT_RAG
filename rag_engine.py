"""
rag_engine.py — Phase 2: Load FAISS index → retrieve → generate answer with citations

This module is imported by app.py (Streamlit UI) and can also be used standalone.
"""

import os
os.environ.setdefault("USE_TORCH", "1")  # prevent sentence-transformers from importing TF

from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import CrossEncoder

load_dotenv()

# Cross-encoder for re-ranking retrieved chunks. Loaded once and reused across queries.
# ms-marco-MiniLM-L-6-v2 is a small, fast cross-encoder trained for passage re-ranking.
_RERANKER = None


def get_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


@dataclass
class RAGResponse:
    """Structured response with answer + cited sources.

    Fields:
        answer:   LLM-generated answer string
        sources:  list of dicts with source_file, page, snippet (display-friendly)
        query:    the original question
        contexts: raw retrieved chunk texts (used by RAGAS and other evaluators)
    """
    answer: str
    sources: list
    query: str
    contexts: list = None  # populated in query(); default keeps backwards compat


SYSTEM_PROMPT = """You are a scientific assistant specializing in SWOT (Surface Water and Ocean Topography) 
satellite mission research. You answer questions using ONLY the provided context from SWOT-related papers.

Rules:
1. Base your answer strictly on the provided context chunks.
2. If the context does not contain enough information, say so — do not hallucinate.
3. Cite which paper and page each piece of information comes from.
4. Use precise scientific language appropriate for an oceanography/geodesy audience.
5. When relevant, connect findings to SWOT mission specifications (2D SSH, KaRIn instrument, 
   ~21-day repeat, 120 km swath, ~2 cm accuracy goal).
"""

def load_retriever(index_path: str = "faiss_index", k: int = 20):
    """
    Load FAISS index from disk and return a retriever.

    k=20 candidates here is the *first-stage* retrieval pool — wider than what
    we actually feed the LLM, so the cross-encoder re-ranker (see rerank())
    has a meaningful set of candidates to re-score before we truncate to the
    final top-k passed to the LLM in query().
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True  # required for local FAISS files
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


def rerank(question: str, docs: list, top_k: int = 5) -> list:
    """
    Re-rank candidate chunks with a cross-encoder and keep the top_k.

    Why a second stage: FAISS similarity search (bi-encoder) embeds the query
    and each chunk independently, so it's fast but only approximates true
    relevance. A cross-encoder jointly encodes (query, chunk) pairs, which is
    slower but scores relevance far more accurately — standard two-stage
    retrieve-then-rerank pattern.
    """
    if not docs:
        return docs
    reranker = get_reranker()
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def format_context(docs: list) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[{i}] Source: {source}, Page {page}\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    return f"""Context from SWOT research papers:

{context}

---

Question: {question}

Answer the question based on the context above. After your answer, list the sources you used as:
Sources: [1] filename p.X, [2] filename p.Y, ...
"""


def query(question: str, index_path: str = "faiss_index", k: int = 5,
          k_retrieve: int = 20, model: str = "claude-sonnet-4-6",
          use_rerank: bool = True, system_prompt: str = None) -> RAGResponse:
    """
    Main RAG query function — two-stage retrieve-then-rerank.

    Args:
        question: The scientific question to answer
        index_path: Path to saved FAISS index
        k: Number of chunks to keep after re-ranking and pass to the LLM
        k_retrieve: Number of candidates pulled from FAISS before re-ranking
        model: Anthropic model to use (claude-sonnet-4-6 recommended for scientific reasoning;
               claude-haiku-4-5-20251001 for a ~10x cheaper option)
        use_rerank: If False, skip the cross-encoder and use the top-k FAISS results directly
        system_prompt: Override the default SYSTEM_PROMPT (for A/B testing prompt variants)

    Returns:
        RAGResponse with answer, sources, and original query
    """
    # 1. Load retriever — when skipping rerank, retrieve only k (no need for a wide pool)
    fetch_k = k_retrieve if use_rerank else k
    retriever = load_retriever(index_path, fetch_k)
    candidates = retriever.invoke(question)

    # 2. Re-rank candidates with a cross-encoder (slower/precise) and keep top-k, or skip
    if use_rerank:
        docs = rerank(question, candidates, top_k=k)
    else:
        docs = candidates[:k]

    # 3. Build context string
    context = format_context(docs)
    
    # 4. Call LLM
    llm = ChatAnthropic(model=model, temperature=0, max_tokens=1024)
    from langchain.schema import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_prompt or SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, context)),
    ]
    response = llm.invoke(messages)
    answer = response.content
    
    # 5. Extract source metadata for display
    sources = []
    for doc in docs:
        sources.append({
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200] + "...",
        })

    # 6. Also expose the raw chunk texts so RAGAS / other evaluators can score grounding
    contexts = [doc.page_content for doc in docs]

    return RAGResponse(answer=answer, sources=sources, query=question, contexts=contexts)


if __name__ == "__main__":
    # Quick CLI test
    test_q = "What is the spatial resolution achieved by SWOT for sea surface height measurements?"
    print(f"Q: {test_q}\n")
    result = query(test_q)
    print(f"A: {result.answer}\n")
    print("Sources:")
    for s in result.sources:
        print(f"  - {s['source_file']} p.{s['page']}")
