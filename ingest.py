"""
ingest.py — Phase 1: Load SWOT PDFs → chunk → embed → save FAISS index

Usage:
    python ingest.py --pdf_dir ./papers

Put your SWOT-related PDFs in ./papers/ before running.
Sources: https://swot.jpl.nasa.gov/science/publications/
         Your own papers, co-author papers, Science articles
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def load_pdfs(pdf_dir: str) -> list:
    """Load all PDFs from a directory, preserving metadata."""
    docs = []
    pdf_paths = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    print(f"Found {len(pdf_paths)} PDFs")
    for path in pdf_paths:
        print(f"  Loading: {path.name}")
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        # Enrich metadata with filename as source label
        for page in pages:
            page.metadata["source_file"] = path.name
        docs.extend(pages)

    print(f"  → {len(docs)} total pages loaded\n")
    return docs


def chunk_documents(docs: list, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    """
    Split documents into overlapping chunks.
    
    chunk_size=800 tokens is a good balance for scientific text:
    - Large enough to keep equations + surrounding context together
    - Small enough to stay focused for retrieval
    chunk_overlap=150 ensures context isn't lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph > sentence > word
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})\n")
    return chunks


def build_vectorstore(chunks: list, index_path: str = "faiss_index") -> FAISS:
    """
    Embed chunks with OpenAI and save FAISS index to disk.
    
    text-embedding-3-small: cheap, fast, 1536-dim, great for scientific text.
    Switch to text-embedding-3-large if you need higher accuracy at 3x cost.
    """
    print("Embedding chunks with text-embedding-3-small...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Build index (this calls OpenAI API — costs ~$0.001 per 1M tokens)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk so ingestion only runs once
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to ./{index_path}/\n")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest SWOT PDFs into FAISS vector store")
    parser.add_argument("--pdf_dir", default="./papers", help="Directory containing PDF files")
    parser.add_argument("--index_path", default="faiss_index", help="Output path for FAISS index")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    args = parser.parse_args()

    print("=" * 50)
    print("SWOT RAG — Ingestion Pipeline")
    print("=" * 50 + "\n")

    docs = load_pdfs(args.pdf_dir)
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    build_vectorstore(chunks, args.index_path)

    print("Done! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
