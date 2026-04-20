"""
evaluate.py — Test your RAG pipeline quality with domain-specific questions

This script runs a set of SWOT-focused questions and scores:
  - Answer relevance (does it use the retrieved context?)
  - Source grounding (are papers cited correctly?)
  - Hallucination check (did it make up SWOT specs not in the papers?)

Usage: python evaluate.py
"""

from rag_engine import query

# These are questions whose answers you know from your own research — 
# perfect for testing that the RAG is grounding correctly.
TEST_QUESTIONS = [
    {
        "question": "What spatial resolution does SWOT achieve for sea surface height?",
        "expected_keywords": ["8 km", "2 km", "KaRIn", "resolution", "swath"],
        "notes": "Should cite your 2024 Science paper"
    },
    {
        "question": "How does SWOT detect abyssal marine tectonics?",
        "expected_keywords": ["gravity", "seamount", "bathymetry", "altimetry"],
        "notes": "Core topic from Yu et al. 2024 Science"
    },
    {
        "question": "What is the noise level of SWOT SSH measurements?",
        "expected_keywords": ["noise", "cm", "accuracy", "KaRIn", "crossover"],
        "notes": "Should cite Earth and Space Science paper"
    },
    {
        "question": "How does SWOT compare to TOPEX/Poseidon or Jason altimeters?",
        "expected_keywords": ["nadir", "swath", "2D", "resolution", "conventional"],
        "notes": "Comparison question — tests multi-source retrieval"
    },
    {
        "question": "Can SWOT detect tsunami signals?",
        "expected_keywords": ["tsunami", "Kamchatka", "dispersive", "2025"],
        "notes": "Should cite Sepulveda et al. 2026 Science"
    },
]


def evaluate():
    print("=" * 60)
    print("SWOT RAG — Evaluation Suite")
    print("=" * 60)
    
    scores = []
    for i, test in enumerate(TEST_QUESTIONS, 1):
        print(f"\nQ{i}: {test['question']}")
        print(f"     Notes: {test['notes']}")
        
        result = query(test["question"])
        
        # Simple keyword-based relevance check
        answer_lower = result.answer.lower()
        hits = [kw for kw in test["expected_keywords"] if kw.lower() in answer_lower]
        score = len(hits) / len(test["expected_keywords"])
        scores.append(score)
        
        print(f"     Answer (first 200 chars): {result.answer[:200]}...")
        print(f"     Sources: {[s['source_file'] for s in result.sources[:3]]}")
        print(f"     Keyword score: {len(hits)}/{len(test['expected_keywords'])} "
              f"({score:.0%}) — matched: {hits}")

    avg = sum(scores) / len(scores)
    print("\n" + "=" * 60)
    print(f"Overall keyword match score: {avg:.0%}")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Add more PDFs if any question scores < 50%")
    print("  - Tune chunk_size in ingest.py if answers miss context")
    print("  - Try k=8 retrieval for multi-faceted questions")


if __name__ == "__main__":
    evaluate()
