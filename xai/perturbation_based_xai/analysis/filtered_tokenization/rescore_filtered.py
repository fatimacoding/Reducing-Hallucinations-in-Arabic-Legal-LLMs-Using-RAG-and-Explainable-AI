"""
Filter list-number tokenization artifacts and rescore RAG-Ex outputs.

Conservative policy: only drop sentences that are pure list-number markers
emitted by the sentence splitter (e.g. "1", "2", "3", "2.", "3."). All
other sentences — including short legal phrases like "مائة ريال.",
clause fragments like "منازعات التنفيذ؛", and letter bullets — are kept.

For each sample, this script:
  1. Detects list-number-marker sentences in `sentences`.
  2. Drops them (and their parallel `scores` entry) from every strategy.
  3. Recomputes `mean_score` over the kept sentences only.
  4. Records what was filtered for transparency / appendix tables.

Notes on data shape:
  - `sentences` is 0-indexed.
  - In strategy["per_sentence"], `sentence_idx` is 1-indexed (i.e. position
    in `sentences` is sentence_idx - 1). The strategy["scores"] list is
    aligned with per_sentence by position.
"""

import json
import re
import argparse
from copy import deepcopy
from statistics import mean


# ---------- Bad-tokenization detection ----------

# Standalone list-number tokens like "1", "2.", "3 )", "4-", "5:" that the
# splitter emitted as a sentence on their own.
_LIST_NUMBER_RE = re.compile(r"^\s*\d+\s*[\.\)\-:]?\s*$")

# Same, but with Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩).
_LIST_NUMBER_AR_RE = re.compile(r"^\s*[\u0660-\u0669]+\s*[\.\)\-:]?\s*$")


def is_list_number_marker(text: str) -> bool:
    """True iff the sentence is a standalone list-number tokenization artifact."""
    s = (text or "").strip()
    if not s:
        return False
    return bool(_LIST_NUMBER_RE.match(s) or _LIST_NUMBER_AR_RE.match(s))


# ---------- Rescoring ----------

def rescore_sample(sample: dict) -> dict:
    """
    Return a new sample with list-number-marker sentences removed from every
    strategy and the strategy mean recomputed. Original `sentences` array is
    kept for reference; a parallel `kept_sentence_indices_0based` is added.
    """
    out = deepcopy(sample)

    # 1. Identify bad sentences (0-based positions in `sentences`).
    bad_positions = []
    kept_positions = []
    for i, text in enumerate(sample["sentences"]):
        if is_list_number_marker(text):
            bad_positions.append({
                "pos": i,
                "sentence_idx": i + 1,  # 1-based, matches per_sentence
                "text": text,
                "reason": "list_number_marker",
            })
        else:
            kept_positions.append(i)

    bad_sentence_idx_set = {b["sentence_idx"] for b in bad_positions}  # 1-based

    # 2. Filter every strategy.
    for strat in out["strategies"].values():
        per_sentence = strat.get("per_sentence", [])
        scores = strat.get("scores", [])

        kept_per_sentence = []
        kept_scores = []
        dropped = []

        # `per_sentence` and `scores` are aligned by position.
        for idx, ps in enumerate(per_sentence):
            sidx = ps.get("sentence_idx")  # 1-based
            score = scores[idx] if idx < len(scores) else ps.get("score")
            if sidx in bad_sentence_idx_set:
                dropped.append({"sentence_idx": sidx, "score": score})
                continue
            kept_per_sentence.append(ps)
            kept_scores.append(score)

        strat["per_sentence"] = kept_per_sentence
        strat["scores"] = kept_scores
        strat["mean_score"] = round(mean(kept_scores), 4) if kept_scores else None
        strat["dropped_sentences"] = dropped

    # 3. Bookkeeping.
    out["filter_report"] = {
        "policy": "list_number_marker_only",
        "num_sentences_before": len(sample["sentences"]),
        "num_sentences_after": len(kept_positions),
        "num_dropped": len(bad_positions),
        "kept_sentence_indices_0based": kept_positions,
        "dropped": bad_positions,
    }
    out["num_sentences_filtered"] = len(kept_positions)

    return out


def rescore_dataset(data: list) -> tuple[list, dict]:
    """Process every sample and return (filtered_data, summary)."""
    filtered = [rescore_sample(s) for s in data]

    total_before = sum(s["filter_report"]["num_sentences_before"] for s in filtered)
    total_after = sum(s["filter_report"]["num_sentences_after"] for s in filtered)
    total_dropped = sum(s["filter_report"]["num_dropped"] for s in filtered)

    # Strategy-level macro-mean comparison.
    strategy_names = list(filtered[0]["strategies"].keys()) if filtered else []
    strat_compare = {}
    for sname in strategy_names:
        before_means = [s["strategies"][sname]["mean_score"]
                        for s in data
                        if s["strategies"][sname].get("mean_score") is not None]
        after_means = [fs["strategies"][sname]["mean_score"]
                       for fs in filtered
                       if fs["strategies"][sname].get("mean_score") is not None]
        strat_compare[sname] = {
            "macro_mean_before": round(mean(before_means), 4) if before_means else None,
            "macro_mean_after": round(mean(after_means), 4) if after_means else None,
            "delta": (round(mean(after_means) - mean(before_means), 4)
                      if before_means and after_means else None),
            "samples_with_data_before": len(before_means),
            "samples_with_data_after": len(after_means),
        }

    summary = {
        "policy": "list_number_marker_only",
        "num_samples": len(filtered),
        "total_sentences_before": total_before,
        "total_sentences_after": total_after,
        "total_dropped": total_dropped,
        "drop_rate": round(total_dropped / total_before, 4) if total_before else 0,
        "strategy_macro_means": strat_compare,
    }
    return filtered, summary


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Drop list-number tokenization artifacts and rescore RAG-Ex outputs."
    )
    ap.add_argument("--input", "-i", default="/mnt/user-data/uploads/ragex_out_v2.json",
                    help="Path to input ragex JSON.")
    ap.add_argument("--output", "-o", default="ragex_out_v2_filtered.json",
                    help="Path to write filtered + rescored JSON.")
    ap.add_argument("--summary", "-s", default="ragex_filter_summary.json",
                    help="Path to write the filtering summary JSON.")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered, summary = rescore_dataset(data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Loaded {summary['num_samples']} samples from {args.input}")
    print(f"Policy: {summary['policy']}")
    print(f"Sentences: {summary['total_sentences_before']} -> "
          f"{summary['total_sentences_after']} "
          f"(dropped {summary['total_dropped']}, "
          f"{summary['drop_rate']*100:.1f}% of total)")

    print("\nStrategy macro-mean before -> after:")
    print(f"  {'Strategy':<22} {'Before':>8} {'After':>8} {'Delta':>8}")
    for sname, info in summary["strategy_macro_means"].items():
        b = info["macro_mean_before"]
        a = info["macro_mean_after"]
        d = info["delta"]
        b_s = f"{b:.4f}" if b is not None else "  n/a "
        a_s = f"{a:.4f}" if a is not None else "  n/a "
        d_s = f"{d:+.4f}" if d is not None else "  n/a "
        print(f"  {sname:<22} {b_s:>8} {a_s:>8} {d_s:>8}")

    print(f"\nWrote filtered output -> {args.output}")
    print(f"Wrote summary         -> {args.summary}")


if __name__ == "__main__":
    main()
