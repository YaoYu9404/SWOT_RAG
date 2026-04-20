"""
rag_engine.py — Phase 2: Load FAISS index → retrieve → generate answer with citations

This module is imported by app.py (Streamlit UI) and can also be used standalone.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()


@dataclass
class RAGResponse:
    """Structured response with answer + cited sources."""
    answer: str
    sources: list
    query: str


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

def load_retriever(index_path: str = "faiss_index", k: int = 5):
    """
    Load FAISS index from disk and return a retriever.
    
    k=5 retrieved chunks is a good default:
    - Enough context for multi-faceted questions
    - Fits comfortably in GPT-4o's context window even with long chunks
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
          model: str = "gpt-4o") -> RAGResponse:
    """
    Main RAG query function.
    
    Args:
        question: The scientific question to answer
        index_path: Path to saved FAISS index
        k: Number of chunks to retrieve
        model: OpenAI model to use (gpt-4o recommended for scientific reasoning)
    
    Returns:
        RAGResponse with answer, sources, and original query
    """
    # 1. Load retriever
    retriever = load_retriever(index_path, k)
    
    # 2. Retrieve relevant chunks
    docs = retriever.invoke(question)
    
    # 3. Build context string
    context = format_context(docs)
    
    # 4. Call LLM
    llm = ChatOpenAI(model=model, temperature=0)
    from langchain.schema import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
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
    
    return RAGResponse(answer=answer, sources=sources, query=question)


if __name__ == "__main__":
    # Quick CLI test
    test_q = "What is the spatial resolution achieved by SWOT for sea surface height measurements?"
    print(f"Q: {test_q}\n")
    result = query(test_q)
    print(f"A: {result.answer}\n")
    print("Sources:")
    for s in result.sources:
        print(f"  - {s['source_file']} p.{s['page']}")
