import json
import os
import time
from typing import List, Dict, Any

import judge

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
INPUT_FILE  = "generated_answers.jsonl"
OUTPUT_FILE = "deepeval_results.jsonl"
LIMIT       = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_nonempty(*items) -> str:
    for x in items:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return ""


def build_context_from_top_sources(top_sources: List[Dict[str, Any]]) -> str:
    """
    Slim context: citation + text only.
    Must be identical to build_context_block() in generator.py so that
    DeepEval judges the answer against exactly the context the model saw.
    """
    blocks = []
    for i, src in enumerate(top_sources or [], start=1):
        citation = first_nonempty(src.get("citation"))
        text     = first_nonempty(src.get("text"))

        if not text:
            continue

        block = (
            f"[مصدر {i}]\n"
            f"الاستشهاد: {citation}\n"
            f"النص: {text}"
        )
        blocks.append(block)

    return "\n\n".join(blocks).strip()


def build_summary(results: List[dict]) -> dict:
    summary = {
        "evaluation_summary": {
            "test_type":              "deepeval_judge_evaluation",
            "total_questions":        len(results),
            "successful_evaluations": len([r for r in results if r.get("scores")]),
            "failed_evaluations":     len([r for r in results if not r.get("scores")]),
        }
    }

    metric_names = set()
    for r in results:
        for name in r.get("scores", {}).keys():
            metric_names.add(name)

    metric_averages = {}
    for metric_name in metric_names:
        scores           = []
        passed_count     = 0
        json_error_count = 0

        for r in results:
            metric_result = r.get("scores", {}).get(metric_name, {})

            if metric_result.get("error") == "JSON_ERROR":
                json_error_count += 1
                continue

            score = metric_result.get("score")
            if score is not None:
                scores.append(score)
                if metric_result.get("passed", False):
                    passed_count += 1

        if scores:
            metric_averages[metric_name] = {
                "average_score":     round(sum(scores) / len(scores), 3),
                "min_score":         round(min(scores), 3),
                "max_score":         round(max(scores), 3),
                "passed_count":      passed_count,
                "pass_rate_percent": round((passed_count / len(scores)) * 100, 1),
                "json_error_count":  json_error_count,
            }

    summary["metric_averages"] = metric_averages
    return summary


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("DEEPEVAL JUDGE EVALUATION")
    print("=" * 70)

    rows = read_jsonl(INPUT_FILE)
    if LIMIT is not None:
        rows = rows[:LIMIT]

    print(f"Loaded rows: {len(rows)}")

    results = []

    for i, row in enumerate(rows, start=1):
        idx              = row.get("idx", i)
        question         = first_nonempty(row.get("question"))
        generated_answer = first_nonempty(row.get("generated_answer"))
        top_sources      = row.get("top_sources") or []

        print(f"\n{'─'*70}")
        print(f"Question {i}/{len(rows)}")
        print(f"{'─'*70}")
        print(f"Q: {question[:80]}...")

        if not question:
            print("Skipped: empty question")
            continue

        if not generated_answer:
            print("Skipped: empty generated answer")
            results.append({
                "idx":               idx,
                "question":          question,
                "generated_answer":  generated_answer,
                "retrieval_context": "",
                "scores":            {},
                "error":             "EMPTY_GENERATED_ANSWER",
            })
            continue

        context = build_context_from_top_sources(top_sources)

        if not context:
            print("Skipped: empty context from top_sources")
            results.append({
                "idx":               idx,
                "question":          question,
                "generated_answer":  generated_answer,
                "retrieval_context": "",
                "scores":            {},
                "error":             "EMPTY_CONTEXT",
            })
            continue

        print("Evaluating with existing judge.py ...")
        scores = judge.evaluate_answer(question, generated_answer, context)

        results.append({
            "idx":               idx,
            "question":          question,
            "generated_answer":  generated_answer,
            "retrieval_context": context,
            "top_sources_count": len(top_sources),
            "scores":            scores,
            "error":             None,
        })

        time.sleep(2)

    summary = build_summary(results)

    final_output = {
        "summary":          summary,
        "detailed_results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults saved: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()