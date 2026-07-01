"""
ragas_eval.py — RAG evaluation with the RAGAS framework

Scores the existing SWOT RAG pipeline on four metrics:
  - faithfulness         : does the answer stay grounded in the retrieved chunks?
  - answer_relevancy     : does the answer actually address the question?
  - context_precision    : were the retrieved chunks relevant (ranking quality)?
  - context_recall*      : did retrieval pull in everything needed? (*only rows with ground_truth)
  - answer_correctness*  : semantic match against a reference answer      (*only rows with ground_truth)

Design notes
------------
Judge is Claude Sonnet (Anthropic) — cross-family from the generator (OpenAI GPT-4o)
so the evaluator can't trivially "agree with itself." This is the standard
fair-eval pattern recommended by the RAGAS team.

Usage:
    # make sure OPENAI_API_KEY and ANTHROPIC_API_KEY are set in .env
    python ragas_eval.py
    python ragas_eval.py --questions eval_questions.jsonl --k 5 --model gpt-4o
    python ragas_eval.py --limit 5                 # smoke test on first 5
    python ragas_eval.py --with-ground-truth-only  # also score recall/correctness
"""

import os
import json
import argparse
import datetime as dt
from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,       # answer_relevancy
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    AnswerCorrectness,
)

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

from rag_engine import query as rag_query

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_JUDGE_MODEL    = "claude-sonnet-4-6"
DEFAULT_EMBED_MODEL    = "text-embedding-3-small"
DEFAULT_GEN_MODEL      = "gpt-4o"
DEFAULT_K              = 5
DEFAULT_RESULTS_DIR    = "eval_results"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_questions(path: str) -> list[dict]:
    """Load questions from JSONL. Each line must have 'question'; 'ground_truth' optional."""
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {i} of {path}: {e}")
    return rows


def run_rag_on_questions(questions: list[dict], index_path: str, k: int, model: str) -> list[dict]:
    """Call the real RAG pipeline for every question. Returns rows ready for RAGAS."""
    rows = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"  [{i}/{len(questions)}] {q[:80]}")
        resp = rag_query(question=q, index_path=index_path, k=k, model=model)
        rows.append({
            "user_input":     q,
            "response":       resp.answer,
            "retrieved_contexts": resp.contexts,
            "reference":      item.get("ground_truth"),   # may be None
            "tags":           item.get("tags", []),
            "sources":        resp.sources,
        })
    return rows


def build_ragas_dataset(rows: list[dict], require_reference: bool = False) -> EvaluationDataset:
    """Wrap rows as RAGAS SingleTurnSamples."""
    samples = []
    for r in rows:
        if require_reference and not r.get("reference"):
            continue
        samples.append(SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r.get("reference"),     # None is fine for ref-free metrics
        ))
    return EvaluationDataset(samples=samples)


def pick_metrics(has_ground_truth: bool):
    """Reference-free metrics always; reference-based metrics only when we have ground truth."""
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]
    if has_ground_truth:
        metrics.append(LLMContextRecall())
        metrics.append(AnswerCorrectness())
    return metrics


def save_results(scores_df, rows: list[dict], out_dir: str, tag: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ragas_{tag}_{ts}"

    csv_path  = Path(out_dir) / f"{stem}.csv"
    json_path = Path(out_dir) / f"{stem}.json"

    # Per-question scores + metadata
    scores_df.to_csv(csv_path, index=False)

    # Richer JSON with the retrieved sources attached (useful for failure triage)
    payload = {
        "timestamp": ts,
        "tag": tag,
        "aggregate": {
            col: float(scores_df[col].mean())
            for col in scores_df.columns
            if scores_df[col].dtype.kind in "fi"
        },
        "rows": [],
    }
    for (_, row), orig in zip(scores_df.iterrows(), rows):
        payload["rows"].append({
            "question": row.get("user_input"),
            "answer":   row.get("response"),
            "reference": row.get("reference"),
            "scores":   {c: (float(row[c]) if row[c] == row[c] else None)  # NaN → None
                         for c in scores_df.columns
                         if scores_df[c].dtype.kind in "fi"},
            "tags":     orig.get("tags", []),
            "sources":  orig.get("sources", []),
        })
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    return csv_path, json_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Evaluate SWOT RAG with RAGAS")
    p.add_argument("--questions",   default="eval_questions.jsonl")
    p.add_argument("--index_path",  default="faiss_index")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--model",       default=DEFAULT_GEN_MODEL, help="Generator LLM for the RAG pipeline")
    p.add_argument("--judge",       default=DEFAULT_JUDGE_MODEL, help="Judge LLM for RAGAS metrics")
    p.add_argument("--embed",       default=DEFAULT_EMBED_MODEL)
    p.add_argument("--limit",       type=int, default=None, help="Smoke-test on first N questions")
    p.add_argument("--with-ground-truth-only", action="store_true",
                   help="Restrict to questions with ground_truth and add recall + correctness")
    p.add_argument("--out-dir",     default=DEFAULT_RESULTS_DIR)
    p.add_argument("--workers",     type=int, default=4, help="Concurrent judge calls (rate-limit friendly)")
    args = p.parse_args()

    if not Path(args.index_path).exists():
        raise SystemExit(f"FAISS index not found at {args.index_path}/. Run ingest.py first.")
    if not Path(args.questions).exists():
        raise SystemExit(f"Questions file not found: {args.questions}")

    print("=" * 60)
    print("SWOT RAG — RAGAS Evaluation")
    print("=" * 60)
    print(f"  Generator : {args.model}")
    print(f"  Judge     : {args.judge}")
    print(f"  Embedder  : {args.embed}")
    print(f"  k         : {args.k}")
    print(f"  questions : {args.questions}")
    print()

    # 1. Load curated questions
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    if args.with_ground_truth_only:
        before = len(questions)
        questions = [q for q in questions if q.get("ground_truth")]
        print(f"Filtered to {len(questions)}/{before} questions with ground_truth\n")

    # 2. Run the real RAG pipeline end-to-end
    print("Running RAG pipeline on questions...")
    rows = run_rag_on_questions(questions, args.index_path, args.k, args.model)
    print()

    # 3. Wire up RAGAS
    has_gt = any(r.get("reference") for r in rows)
    metrics = pick_metrics(has_ground_truth=has_gt)
    dataset = build_ragas_dataset(rows, require_reference=args.with_ground_truth_only)

    judge_llm  = LangchainLLMWrapper(ChatAnthropic(model=args.judge, temperature=0))
    judge_embs = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=args.embed))
    run_cfg    = RunConfig(max_workers=args.workers, timeout=60)

    print(f"Scoring {len(dataset)} samples with {len(metrics)} metrics "
          f"(judge={args.judge}, workers={args.workers})...\n")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embs,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    # 4. Report
    df = result.to_pandas()
    print("\n" + "=" * 60)
    print("Aggregate scores")
    print("=" * 60)
    for col in df.columns:
        if df[col].dtype.kind in "fi":
            print(f"  {col:<32s} {df[col].mean():.3f}")

    tag = "gt_only" if args.with_ground_truth_only else "all"
    csv_path, json_path = save_results(df, rows, args.out_dir, tag)
    print(f"\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    # 5. Flag low-scoring rows so you know where to look
    print("\nLowest-scoring questions (by faithfulness):")
    low = df.sort_values("faithfulness").head(3)
    for _, row in low.iterrows():
        q = str(row.get("user_input", ""))[:70]
        print(f"  faith={row['faithfulness']:.2f}  {q}")


if __name__ == "__main__":
    main()
