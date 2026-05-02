import os
import json
from typing import List, Dict, Any

from llm import call_deepseek

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
INPUT_FILE  = "najiz_records.jsonl"
OUTPUT_FILE = "without_context_generated_answers.jsonl"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ── Identical to SP1 prompt template (no context) ─────────────────────
PROMPT_WITHOUT_CONTEXT = """أنت مساعد ذكي متخصص في القانون السعودي.

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


def build_messages(question: str) -> List[Dict[str, str]]:
    prompt = PROMPT_WITHOUT_CONTEXT.format(question=question.strip())
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
    print(f"Loaded records: {len(rows)}")

    for idx, rec in enumerate(rows, start=1):
        question = first_nonempty(rec.get("question"))
        if not question:
            continue

        print("\n" + "=" * 100)
        print(f"[{idx}] {question}")

        messages = build_messages(question)

        try:
            generated_answer = call_deepseek(messages)
            print("Generated answer:")
            print(generated_answer)

            out_row = {
                "idx":              rec.get("idx", idx),
                "question":         question,
                "generated_answer": generated_answer,
                "generation_error": "",
            }
            write_jsonl_row(OUTPUT_FILE, out_row)

        except Exception as e:
            print(f"Generation failed for row {idx}: {e}")
            out_row = {
                "idx":              rec.get("idx", idx),
                "question":         question,
                "generated_answer": "",
                "generation_error": str(e),
            }
            write_jsonl_row(OUTPUT_FILE, out_row)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_FILE}")