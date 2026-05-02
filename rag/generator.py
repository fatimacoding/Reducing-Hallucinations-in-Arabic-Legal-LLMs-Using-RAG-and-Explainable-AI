import os
import json
from typing import List, Dict, Any

from llm import call_deepseek

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
INPUT_FILE  = "najiz_top_sources.jsonl"
OUTPUT_FILE = "generated_answers.jsonl"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ── Identical to SP1 prompt template ──────────────────────────────────
PROMPT_WITH_CONTEXT = """أنت مساعد ذكي متخصص في القانون السعودي.

السياق:
{context_block}

السؤال:
{question}

الإجابة:"""


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


def write_jsonl_row(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_nonempty(*items) -> str:
    for x in items:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return ""


def clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def build_context_block(top_sources: List[Dict[str, Any]]) -> str:
    """
    Slim context: citation + text only.
    النظام and المادة are removed — fully redundant with الاستشهاد.
    """
    blocks = []
    for i, src in enumerate(top_sources, start=1):
        citation = first_nonempty(src.get("citation"))
        text     = clean_text(src.get("text") or "")

        if not text:
            continue

        block = (
            f"[مصدر {i}]\n"
            f"الاستشهاد: {citation}\n"
            f"النص: {text}"
        )
        blocks.append(block)

    return "\n\n".join(blocks).strip()


def build_messages(question: str, top_sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_block = build_context_block(top_sources)
    prompt = PROMPT_WITH_CONTEXT.format(
        context_block=context_block,
        question=question.strip()
    )
    return [{"role": "user", "content": prompt}]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is missing. Set it as an environment variable first.")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    rows = read_jsonl(INPUT_FILE)
    print(f"Loaded retrieval rows: {len(rows)}")

    for idx, rec in enumerate(rows, start=1):
        question    = first_nonempty(rec.get("question"))
        top_sources = [s for s in (rec.get("top_sources") or []) if s.get("added_by") == "normal"]

        if not question:
            continue

        if not top_sources:
            print(f"Skipping row {idx}: no normal sources")
            continue

        print("\n" + "=" * 100)
        print(f"[{idx}] {question}")

        messages = build_messages(question, top_sources)

        try:
            generated_answer = call_deepseek(messages)
            print("Generated answer:")
            print(generated_answer)

            out_row = {
                "idx":              rec.get("idx", idx),
                "question":         question,
                "context":          rec.get("context"),
                "reference_answer": rec.get("reference_answer"),
                "generated_answer": generated_answer,
                "generation_error": "",
                "top_sources":      top_sources,
                "selected_k":       len(top_sources),
                "used_fallback":    rec.get("global_used_fallback", False),
                "warning_reason":   rec.get("global_warning_reason"),
            }
            write_jsonl_row(OUTPUT_FILE, out_row)

        except Exception as e:
            print(f"Generation failed for row {idx}: {e}")

            out_row = {
                "idx":              rec.get("idx", idx),
                "question":         question,
                "context":          rec.get("context"),
                "reference_answer": rec.get("reference_answer"),
                "generated_answer": "",
                "generation_error": str(e),
                "top_sources":      top_sources,
                "selected_k":       len(top_sources),
                "used_fallback":    rec.get("global_used_fallback", False),
                "warning_reason":   rec.get("global_warning_reason"),
            }
            write_jsonl_row(OUTPUT_FILE, out_row)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_FILE}")