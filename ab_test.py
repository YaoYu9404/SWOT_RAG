"""
ab_test.py — Compare multiple RAG pipeline configurations with RAGAS scoring

Loads named variants from variants.yaml, runs each on the same eval questions,
scores them with RAGAS, and prints a side-by-side comparison table.

Usage:
    python ab_test.py                                      # all variants, all questions
    python ab_test.py --limit 5                            # smoke test on first 5
    python ab_test.py --names baseline haiku               # only these two variants
    python ab_test.py --names baseline wide_retrieval --limit 10 --workers 2
"""

import os
import json
import argparse
import datetime as dt
from pathlib import Path

import yaml
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

from rag_engine import query as rag_query

load_dotenv()

DEFAULT_JUDGE   = "claude-sonnet-4-6"
DEFAULT_EMBED   = "text-embedding-3-small"
DEFAULT_QFILE   = "eval_questions.jsonl"
DEFAULT_VFILE   = "variants.yaml"
DEFAULT_INDEX   = "faiss_index"
DEFAULT_OUT     = "eval_results"
SCORE_COLS      = ["faithfulness", "answer_relevancy", "context_precision"]


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_variants(path: str, names: list = None) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    variants = data.get("variants", {})
    if names:
        missing = [n for n in names if n not in variants]
        if missing:
            raise SystemExit(f"Variants not found in {path}: {missing}")
        variants = {n: variants[n] for n in names}
    return variants


def load_questions(path: str, limit: int = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


# ── RAG runner ───────────────────────────────────────────────────────────────

def run_variant(cfg: dict, questions: list[dict], index_path: str) -> list[dict]:
    rows = []
    total = len(questions)
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"    [{i}/{total}] {q[:75]}")
        resp = rag_query(
            question=q,
            index_path=index_path,
            k=cfg.get("k", 5),
            k_retrieve=cfg.get("k_retrieve", 20),
            model=cfg.get("model", "claude-sonnet-4-6"),
            use_rerank=cfg.get("use_rerank", True),
            system_prompt=cfg.get("system_prompt"),
        )
        rows.append({
            "user_input":         q,
            "response":           resp.answer,
            "retrieved_contexts": resp.contexts,
            "reference":          item.get("ground_truth"),
            "tags":               item.get("tags", []),
        })
    return rows


# ── RAGAS scorer ─────────────────────────────────────────────────────────────

def score_rows(rows: list[dict], judge_llm, judge_embs, workers: int) -> pd.DataFrame:
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r.get("reference"),
        )
        for r in rows
    ]
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
        llm=judge_llm,
        embeddings=judge_embs,
        run_config=RunConfig(max_workers=workers, timeout=60),
        raise_exceptions=False,
    )
    return result.to_pandas()


# ── Output ───────────────────────────────────────────────────────────────────

def save_variant_results(df: pd.DataFrame, rows: list[dict], name: str,
                         out_dir: str, ts: str) -> Path:
    path = Path(out_dir) / f"ab_{name}_{ts}.csv"
    df.to_csv(path, index=False)
    return path


def print_comparison(summary: dict[str, dict]):
    cols = SCORE_COLS
    name_w = max(len(n) for n in summary) + 2
    col_w  = 12

    header = f"{'Variant':<{name_w}}" + "".join(f"{c:<{col_w}}" for c in cols)
    print("\n" + "=" * len(header))
    print("A/B TEST RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, scores in summary.items():
        row = f"{name:<{name_w}}" + "".join(
            f"{scores.get(c, float('nan')):<{col_w}.3f}" for c in cols
        )
        print(row)
    print("=" * len(header))

    # highlight best per metric
    print("\nBest variant per metric:")
    for col in cols:
        best = max(summary, key=lambda n: summary[n].get(col, -1))
        print(f"  {col:<32s} → {best}  ({summary[best].get(col, 0):.3f})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="A/B test SWOT RAG variants with RAGAS")
    p.add_argument("--variants",   default=DEFAULT_VFILE, help="Path to variants.yaml")
    p.add_argument("--questions",  default=DEFAULT_QFILE)
    p.add_argument("--index_path", default=DEFAULT_INDEX)
    p.add_argument("--names",      nargs="+", help="Subset of variant names to run")
    p.add_argument("--limit",      type=int,  help="Use only the first N questions (smoke test)")
    p.add_argument("--judge",      default=DEFAULT_JUDGE)
    p.add_argument("--embed",      default=DEFAULT_EMBED)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--out-dir",    default=DEFAULT_OUT)
    args = p.parse_args()

    for req in [args.index_path, args.questions, args.variants]:
        if not Path(req).exists():
            raise SystemExit(f"Not found: {req}")

    variants  = load_variants(args.variants, args.names)
    questions = load_questions(args.questions, args.limit)

    print("=" * 60)
    print("SWOT RAG — A/B Test")
    print("=" * 60)
    print(f"  Variants  : {list(variants)}")
    print(f"  Questions : {len(questions)}")
    print(f"  Judge     : {args.judge}")
    print()

    judge_llm  = LangchainLLMWrapper(ChatAnthropic(model=args.judge, temperature=0))
    judge_embs = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=args.embed))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    summary: dict[str, dict] = {}

    for name, cfg in variants.items():
        print(f"\n{'─'*60}")
        print(f"Variant: {name}")
        cfg_display = {k: v for k, v in cfg.items() if k != "system_prompt"}
        print(f"Config : {cfg_display}")
        print()

        print("  Running RAG pipeline...")
        rows = run_variant(cfg, questions, args.index_path)

        print("\n  Scoring with RAGAS...")
        df = score_rows(rows, judge_llm, judge_embs, args.workers)

        out_path = save_variant_results(df, rows, name, args.out_dir, ts)
        print(f"  Saved: {out_path}")

        summary[name] = {
            col: float(df[col].mean())
            for col in SCORE_COLS
            if col in df.columns
        }

    print_comparison(summary)

    # Save summary CSV
    summary_df = pd.DataFrame(summary).T
    summary_path = Path(args.out_dir) / f"ab_summary_{ts}.csv"
    summary_df.to_csv(summary_path)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
