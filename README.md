# SWOT GeoScience RAG

A retrieval-augmented generation (RAG) system for querying SWOT satellite research papers.
Built by Yao Yu as an LLM portfolio project.

## What this does

Ask natural-language questions about SWOT science and get answers grounded in your paper corpus,
with citations to the source paper and page number.

Example questions:
- "What spatial resolution does SWOT achieve for SSH?"
- "How does KaRIn differ from conventional nadir altimetry?"
- "What abyssal features are detectable by SWOT gravity data?"

---

## Project structure

```
swot_rag/
├── papers/           ← Put your SWOT PDFs here
├── faiss_index/      ← Auto-generated after ingestion
├── ingest.py         ← Phase 1: PDF → chunks → embeddings → FAISS
├── rag_engine.py     ← Phase 2: query → retrieve → LLM → answer
├── app.py            ← Streamlit web UI
├── evaluate.py       ← Quality evaluation suite
├── requirements.txt
└── .env              ← Your OpenAI API key (copy from .env.example)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and paste your key from https://platform.openai.com/api-keys
```

### 3. Add SWOT papers
Download PDFs and place them in `./papers/`. 

Good sources:
- Your own papers (Yu et al. 2024 Science, 2026 Science)
- SWOT Science Team publications: https://swot.jpl.nasa.gov/science/publications/
- Co-author papers (Sandwell, Gille, Dibarboure)
- SWOT Algorithm Theoretical Basis Documents (ATBDs) from NASA

Aim for 10–30 papers to start. More = better coverage.

### 4. Run ingestion (one-time)
```bash
mkdir papers
# copy your PDFs into papers/
python ingest.py --pdf_dir ./papers
```
This will:
- Load and parse all PDFs
- Split into ~800-token chunks with 150-token overlap
- Embed with OpenAI text-embedding-3-small (~$0.001 per 100 pages)
- Save FAISS index to `./faiss_index/`

### 5. Launch the app
```bash
streamlit run app.py
# Or python -m streamlit run app.py
```
Opens at http://localhost:8501

---

## Tuning tips

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `chunk_size` | 800 | Increase to 1200 if answers miss equation context |
| `chunk_overlap` | 150 | Increase to 200 if answers cut off mid-sentence |
| `k` (retrieval) | 5 | Increase to 8 for broad/multi-topic questions |
| model | gpt-4o | Use gpt-4o-mini to cut cost by 10x (slightly lower quality) |

---

## Evaluation

### Quick keyword check

Smoke-test retrieval with a hand-written keyword list:
```bash
python evaluate.py
```
Target: >70% keyword match score. If lower, add more papers or increase chunk overlap.

### RAGAS — LLM-as-a-judge evaluation

For proper, reference-free scoring use the RAGAS suite:
```bash
python ragas_eval.py                         # all questions, defaults
python ragas_eval.py --limit 5               # smoke test on 5
python ragas_eval.py --with-ground-truth-only  # adds recall + correctness
```

Scores the pipeline on:

| Metric | What it catches | Needs reference answer? |
|---|---|---|
| `faithfulness` | answer makes claims not in retrieved chunks (hallucination) | no |
| `answer_relevancy` | answer drifts off-topic | no |
| `context_precision` | retriever pulled irrelevant chunks and ranked them high | no |
| `context_recall` | retriever missed a chunk needed for the reference answer | yes |
| `answer_correctness` | semantic match to a reference answer | yes |

**Design choice — cross-family judge.** The generator is GPT-4o (OpenAI) but the
RAGAS judge is Claude Sonnet (Anthropic). Using a different model family for the
judge avoids "the model agrees with itself" bias and makes scores defensible when
comparing pipelines.

**Required env vars:** `OPENAI_API_KEY` (generator + embedder) and
`ANTHROPIC_API_KEY` (judge), both read from `.env`.

**Questions live in `eval_questions.jsonl`** — one JSON object per line with a
`question` field and optional `ground_truth` / `tags`. Add more as you discover
failure modes; the recall + correctness metrics activate automatically for any
row that has a `ground_truth`.

Results land in `eval_results/ragas_<tag>_<timestamp>.{csv,json}` — the JSON
also includes the retrieved sources per row so failing questions are easy to
triage.

---

## A/B testing

Compare named pipeline variants (model, retrieval depth, reranking, prompt) on the full eval set:

```bash
python ab_test.py                              # all 5 variants, all questions
python ab_test.py --limit 5                    # smoke test on first 5 questions
python ab_test.py --names baseline haiku       # only compare these two
```

Variants are defined in `variants.yaml`. Built-in variants:

| Variant | What it tests |
|---|---|
| `baseline` | claude-sonnet-4-6, k=5, k_retrieve=20, rerank on |
| `haiku` | claude-haiku-4-5-20251001 — cost vs quality tradeoff |
| `wide_retrieval` | k=8, k_retrieve=30 — more context for multi-part questions |
| `no_rerank` | skip the cross-encoder — faster, potentially noisier |
| `concise_prompt` | tighter system prompt variant |

The script scores each variant with RAGAS (faithfulness, answer_relevancy, context_precision)
and prints a comparison table. Per-variant CSVs and a summary land in `eval_results/`.

**UI comparison:** The Streamlit app's **Compare variants** tab lets you run two
configurations side by side on a single question without touching the CLI.

---

## Extension ideas (for your GitHub README)

- [ ] Add metadata filters (e.g. retrieve only from papers post-2022)
- [ ] Swap FAISS for Chroma for easier cloud deployment
- [ ] Add re-ranking step (Cohere or cross-encoder) for better relevance
- [ ] Fine-tune embeddings on SWOT domain text
- [ ] Build comparison mode: same Q answered from different paper subsets
- [ ] Export Q&A sessions as structured JSON for downstream analysis

---

## Tech stack

- **LangChain** — document loading, chunking, retrieval chain
- **FAISS** — fast vector similarity search (Facebook AI)
- **OpenAI** — embeddings (text-embedding-3-small) + generation (GPT-4o)
- **Streamlit** — web UI
- **pypdf** — PDF parsing
