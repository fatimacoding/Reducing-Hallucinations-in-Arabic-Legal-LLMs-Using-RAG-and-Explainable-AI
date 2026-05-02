import os
import re
import json
import time
from typing import List, Dict, Any

import requests
from sentence_transformers import CrossEncoder


# =========================================================
# CONFIG
# =========================================================
INCLUDE_HIDDEN_INTERNAL_FIELDS = False
INPUT_JSONL = "generated_answersH.jsonl"
OUTPUT_JSONL = "xai_results_finalA.jsonl"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

REQUEST_TIMEOUT = 120
SLEEP_BETWEEN_CALLS = 0.6

TOP_EVIDENCE_PER_CLAIM = 3
MAX_CLAIMS = 12
MIN_CLAIMS = 2

SAVE_PRETTY_JSON_COPY = True
PRETTY_JSON_OUTPUT = "xai_results_finalA.pretty.json"


# =========================================================
# IO HELPERS
# =========================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pretty_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# =========================================================
# TEXT / PARSING HELPERS
# =========================================================

def count_sentences(text: str) -> int:
    parts = re.split(r"[.!؟!\n]+", text)
    return len([p for p in parts if p.strip()])


def fallback_max_claims(answer: str) -> int:
    n = count_sentences(answer)
    return max(MIN_CLAIMS, min(n, MAX_CLAIMS))


def normalize_priority(priority: str) -> str:
    if not priority:
        return "medium"
    p = str(priority).strip().lower()

    if p in {"high", "عالية", "مرتفع", "مرتفعة"}:
        return "high"
    if p in {"low", "منخفض", "منخفضة"}:
        return "low"
    return "medium"


def normalize_bool(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "نعم", "y"}


def safe_json_extract(text: str) -> Any:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raise ValueError("Could not parse JSON from model output.")


def safe_get_source_text(src: Dict[str, Any]) -> str:
    for k in ["rerank_text", "text", "content", "article_text", "body"]:
        val = src.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def build_source_label(src: Dict[str, Any]) -> str:
    source = src.get("source", "unknown")
    law = src.get("law") or src.get("doc_title") or src.get("title") or "Unknown Law"
    article = src.get("article") or src.get("article_title") or ""
    article_num = src.get("article_num_parsed") or src.get("article_num_raw") or src.get("article_num") or ""
    citation = src.get("citation") or ""

    parts = [f"source={source}", f"law={law}"]
    if article:
        parts.append(f"article={article}")
    if article_num:
        parts.append(f"article_num={article_num}")
    if citation:
        parts.append(f"citation={citation}")

    return " | ".join(parts)


def normalize_claim_key(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def deduplicate_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for claim in claims:
        key = normalize_claim_key(claim.get("claim_text", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(claim)
    return unique


# =========================================================
# DEEPSEEK API
# =========================================================

def deepseek_chat(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is missing. Set it as an environment variable.")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = requests.post(
        DEEPSEEK_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise ValueError(f"Unexpected DeepSeek response format: {data}") from e


# =========================================================
# CLAIM EXTRACTION
# =========================================================

CLAIM_EXTRACTION_SYSTEM = """
You are a strict Arabic legal claim extraction assistant.

Task:
Split the generated Arabic legal answer into atomic, self-contained, verifiable claims.

Strict rules:

1. Atomicity
- Each claim must express exactly one legal/factual idea.
- Do not merge a legal ruling with its condition or exception.
- If a sentence contains a rule and an exception, split them into separate claims.

2. Decontextualization
- Each claim must be understandable on its own.
- Replace vague references with explicit legal entities only when the answer itself makes them clear.
- Add only the minimum necessary context from the answer.
- Do not add outside information.

3. Legal role preservation
- Do NOT change legal roles or entities.
- Preserve the legal subject of the statement exactly.
- If the evidence uses a different legal role/entity than the claim, do not mark it as Supported.

4. Negation and exception preservation
- Preserve the legal meaning of negation and exception.
- You may rephrase for clarity, but do not invert the intended legal meaning.
- Treat exception and negation words as meaning-critical signals.

5. Verifiability filtering
Extract only claims that can be verified against legal evidence.
Ignore:
- opinions
- recommendations
- hypotheticals
- filler text
- restatements of the question
- transition sentences

6. Completeness
- Keep legally important details: numbers, dates, durations, conditions, exceptions, competent authority, penalties, fees, and article references.
- If a claim depends on an exception or condition, include it explicitly or separate it into another claim.

needs_verification:
- true for legal rulings, conditions, exceptions, penalties, fees, dates, durations, numbers, competent authorities, procedural requirements.
- false only for obvious non-legal connectors or purely linguistic definitions.

priority:
- high: legal ruling, exception, condition, penalty, fee, number, duration, competent authority.
- medium: procedural detail, requirement, general legal detail.
- low: simple definition or low-risk factual detail.

claim_type:
- legal_ruling
- condition
- exception
- procedural_detail
- calculation
- definition
- citation_reference
- factual_detail

Output:
- Arabic claim_text only.
- Do not exceed the maximum number of claims provided by the user.
- Return JSON only.

Return exactly:
{
  "claims": [
    {
      "claim_text": "...",
      "needs_verification": true,
      "priority": "high",
      "claim_type": "legal_ruling"
    }
  ]
}
"""


def extract_claims(question: str, generated_answer: str) -> List[Dict[str, Any]]:
    max_claims = fallback_max_claims(generated_answer)

    user_prompt = f"""
قسّم الجواب التالي إلى ادعاءات قانونية قابلة للتحقق.

السؤال:
{question}

الجواب:
{generated_answer}

الحد الأعلى لعدد الادعاءات:
{max_claims}

أرجع JSON فقط.
"""

    raw = deepseek_chat(CLAIM_EXTRACTION_SYSTEM, user_prompt, temperature=0.0)
    parsed = safe_json_extract(raw)

    claims = parsed.get("claims", [])
    out = []

    for c in claims:
        claim_text = str(c.get("claim_text", "")).strip()
        if not claim_text:
            continue

        out.append({
            "claim_text": claim_text,
            "needs_verification": normalize_bool(c.get("needs_verification", True), default=True),
            "priority": normalize_priority(c.get("priority", "medium")),
            "claim_type": str(c.get("claim_type", "factual_detail")).strip() or "factual_detail",
        })

    return deduplicate_claims(out)[:MAX_CLAIMS]


# =========================================================
# CROSS-ENCODER SCORING — RANKING ONLY, NO ROUTING
# =========================================================

class ClaimEvidenceMatcher:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model = CrossEncoder(model_name, device="cpu")

    def rank_sources_for_claim(
        self,
        claim_text: str,
        top_sources: List[Dict[str, Any]],
        top_n: int = TOP_EVIDENCE_PER_CLAIM,
    ) -> List[Dict[str, Any]]:
        valid_sources = []
        pairs = []

        for src in top_sources:
            txt = safe_get_source_text(src)
            if not txt:
                continue
            valid_sources.append(src)
            pairs.append((claim_text, txt))

        if not pairs:
            print(f"  [WARN] No evidence text found for claim: {claim_text[:60]}")
            return []

        scores = self.model.predict(pairs)

        ranked = []
        for src, score in zip(valid_sources, scores):
            ranked.append({
                "score": float(score),
                "source_label": build_source_label(src),
                "source_text": safe_get_source_text(src),
                "source_meta": {
                    "rank": src.get("rank"),
                    "source": src.get("source"),
                    "law": src.get("law"),
                    "article": src.get("article"),
                    "article_num_parsed": src.get("article_num_parsed"),
                    "citation": src.get("citation"),
                    "rerank_score": src.get("rerank_score"),
                    "bm25_score": src.get("bm25_score"),
                    "dense_score": src.get("dense_score"),
                    "added_by": src.get("added_by"),
                    "used_fallback": src.get("used_fallback"),
                    "warning_reason": src.get("warning_reason"),
                },
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_n]


# =========================================================
# VERIFIER — EVERY CLAIM GOES TO DEEPSEEK
# =========================================================

VERIFIER_SYSTEM = """
You are a strict but fair legal groundedness verifier for Arabic legal RAG.

You will receive:
- the user question
- one claim extracted from the generated answer
- retrieved legal evidence passages

Your job:
Check whether the claim is grounded in the retrieved evidence.

Labels:
- Supported
- Unsupported
- Not Enough Evidence

Core principle:
Evaluate each claim as a local claim from a sequential answer.
If the evidence supports the specific claim, label it Supported.
The claim does NOT need to restate the full article, full list, or all related details.

Supported:
The evidence clearly supports the claim, or the claim is a valid direct inference from the evidence.
A claim can be Supported even if the evidence contains additional details, list items, exceptions, or a more specific value.

Unsupported:
The claim contradicts the evidence, changes legal roles/entities, reverses negation/exception meaning, or adds a specific legal condition that is incompatible with the evidence.

Not Enough Evidence:
The evidence is unrelated, too weak, or does not address the claim.

Legal reasoning rules:
- Judge only based on the retrieved evidence.
- Do not use outside knowledge.
- Do not penalize a claim for not mentioning the rest of a legal list.
- Do not treat general rules and implementing regulations as contradictions if they are compatible.
- A maximum/ceiling value is compatible with a smaller specific value unless the claim presents the maximum as the exact final amount.
- Legal roles are critical: المتهم and المجني عليه are not interchangeable.
- Negation and exceptions are critical: لا، ليس، غير، إلا، باستثناء.
- If the claim is Supported, do not mention that it is incomplete, narrower, broader, or missing details.

Return JSON only.

Return schema:
{
  "label": "Supported",
  "reason": "short Arabic explanation",
  "explanation_type": "direct_match",
  "used_evidence_indices": [1]
}

Allowed explanation_type:
- direct_match
- supported_inference
- contradiction
- missing_detail
- irrelevant_evidence
"""


def collapse_label_binary(label: str, explanation_type: str) -> str:
    label = (label or "").strip()
    explanation_type = (explanation_type or "").strip()

    if label == "Supported":
        return "Supported"
    if label == "Partially Supported":
        return "Supported"
    if label == "Unsupported" or explanation_type == "contradiction":
        return "Unsupported"
    return "Not Enough Evidence"


def make_supported_reason(used_evidence_indices: List[int]) -> str:
    if used_evidence_indices:
        idxs = "، ".join(str(i) for i in used_evidence_indices)
        return f"الادعاء مدعوم بالأدلة المسترجعة رقم {idxs}."
    return "الادعاء مدعوم بالأدلة المسترجعة."


def clean_public_explanation(final_label: str, raw_reason: str, used_evidence_indices: List[int]) -> str:
    raw_reason = (raw_reason or "").strip()

    if final_label != "Supported":
        return raw_reason or "لم تتوفر أدلة كافية لدعم الادعاء."

    bad_markers = [
        "ناقص",
        "يفتقد",
        "لم يذكر جميع",
        "لا يذكر جميع",
        "يقتصر على",
        "دون غيرها",
        "لا يشمل",
        "أضيق من الدليل",
        "أوسع من الدليل",
        "غير مكتمل",
        "incomplete",
        "missing",
    ]

    if any(marker in raw_reason for marker in bad_markers):
        return make_supported_reason(used_evidence_indices)

    return raw_reason or make_supported_reason(used_evidence_indices)


def verify_claim_against_top_sources(
    question: str,
    claim_text: str,
    ranked_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evidence_chunks = []

    best_score = max((ev.get("score", 0.0) for ev in ranked_evidence), default=0.0)

    for i, ev in enumerate(ranked_evidence, start=1):
        evidence_chunks.append(
            f"[Evidence {i}]\n"
            f"Relevance Score: {ev.get('score', 0.0):.4f}\n"
            f"Source: {ev.get('source_label', 'Unknown')}\n"
            f"Text:\n{ev.get('source_text', '')}\n"
        )

    evidence_text = "\n".join(evidence_chunks) if evidence_chunks else "[No retrieved evidence was available for this claim.]"

    user_prompt = f"""
السؤال:
{question}

الادعاء:
{claim_text}

أفضل درجة صلة بين الادعاء والأدلة:
{best_score:.4f}

الأدلة المسترجعة:
{evidence_text}

تعليمات مهمة:
- قيّم الادعاء محليًا كجزء من جواب متتابع.
- إذا كان الادعاء نفسه مدعومًا من الدليل، اختر Supported.
- لا تعاقب الادعاء لأنه لا يذكر بقية عناصر القائمة أو بقية تفاصيل المادة.
- لا تعتبر اختلاف مستوى التفصيل تناقضًا.
- إذا كان النص العام يضع حدًا أعلى، والنص التفصيلي يذكر قيمة أقل، فهذا تكامل وليس تناقضًا.
- إذا بدّل الادعاء الدور القانوني مثل المتهم/المجني عليه، فهذا خطأ جوهري.
- أرجع JSON فقط.
"""

    raw = deepseek_chat(VERIFIER_SYSTEM, user_prompt, temperature=0.0)
    parsed = safe_json_extract(raw)

    raw_label = str(parsed.get("label", "")).strip()
    if raw_label not in {"Supported", "Partially Supported", "Unsupported", "Not Enough Evidence"}:
        raw_label = "Not Enough Evidence"

    raw_reason = str(parsed.get("reason", "")).strip()
    explanation_type = str(parsed.get("explanation_type", "")).strip() or "irrelevant_evidence"
    used_evidence_indices = parsed.get("used_evidence_indices", [])
    if not isinstance(used_evidence_indices, list):
        used_evidence_indices = []

    final_label = collapse_label_binary(raw_label, explanation_type)
    public_reason = clean_public_explanation(final_label, raw_reason, used_evidence_indices)

    return {
        "label": final_label,
        "reason": public_reason,
        "explanation_type": explanation_type,
        "used_evidence_indices": used_evidence_indices,
        "hidden_internal": {
            "raw_label": raw_label,
            "raw_reason": raw_reason,
            "collapse_rule": "Partially Supported -> Supported; public explanation cleaned",
        },
    }


# =========================================================
# XAI SUMMARY / AGGREGATION
# =========================================================

def summarize_answer_level(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        "Supported": 0,
        "Unsupported": 0,
        "Not Enough Evidence": 0,
        "Not Applicable": 0,
    }

    hidden_counts = {
        "Supported": 0,
        "Partially Supported": 0,
        "Unsupported": 0,
        "Not Enough Evidence": 0,
        "Not Applicable": 0,
    }

    for c in claims:
        lbl = c.get("final_label", "Not Enough Evidence")
        if lbl not in counts:
            lbl = "Not Enough Evidence"
        counts[lbl] += 1

        raw_lbl = c.get("hidden_internal", {}).get("raw_verifier_label")
        if raw_lbl in hidden_counts:
            hidden_counts[raw_lbl] += 1

    evaluable_claims = counts["Supported"] + counts["Unsupported"] + counts["Not Enough Evidence"]

    if evaluable_claims == 0:
        verdict = "No Verifiable Claims"
    elif counts["Unsupported"] > 0:
        verdict = "Contains Unsupported Content"
    elif counts["Not Enough Evidence"] == evaluable_claims:
        verdict = "Insufficient Evidence"
    elif counts["Supported"] == evaluable_claims:
        verdict = "Fully Grounded"
    else:
        verdict = "Mixed Grounding"

    grounded_ratio = counts["Supported"] / evaluable_claims if evaluable_claims else 0.0

    return {
        "answer_verdict": verdict,
        "num_claims_total": len(claims),
        "num_claims_evaluable": evaluable_claims,
        "label_counts": counts,
        "grounded_ratio": round(grounded_ratio, 4),
        "scoring_mode": "binary_claim_level_presence_based",
        "hidden_internal_summary": {
            "raw_verifier_label_counts": hidden_counts,
            "public_label_mapping": {
                "Supported": "Supported",
                "Partially Supported": "Supported",
                "Unsupported": "Unsupported",
                "Not Enough Evidence": "Not Enough Evidence",
            },
        },
    }


def build_xai_table_rows(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for i, c in enumerate(claims, start=1):
        best_source = None
        best_source_score = None
        if c.get("top_evidence"):
            best_source = c["top_evidence"][0].get("source_label")
            best_source_score = c["top_evidence"][0].get("score")

        rows.append({
            "claim_id": i,
            "claim_text": c.get("claim_text"),
            "claim_type": c.get("claim_type"),
            "priority": c.get("priority"),
            "best_source": best_source,
            "best_score": best_source_score,
            "label": c.get("final_label"),
            "explanation": c.get("verification", {}).get("reason"),
            "explanation_type": c.get("verification", {}).get("explanation_type"),
        })
    return rows


# =========================================================
# MAIN RECORD PROCESSING
# =========================================================

def process_record(record: Dict[str, Any], matcher: ClaimEvidenceMatcher) -> Dict[str, Any]:
    question = str(record.get("question", "")).strip()
    generated_answer = str(
        record.get("generated_answer")
        or record.get("answer")
        or record.get("response")
        or ""
    ).strip()
    top_sources = record.get("top_sources") or []

    output = {
        "idx": record.get("idx"),
        "question": question,
        "generated_answer": generated_answer,
        "selected_k": record.get("selected_k"),
        "used_fallback": record.get("used_fallback"),
        "warning_reason": record.get("warning_reason"),
        "claims": [],
        "answer_summary": {},
        "xai_rows": [],
    }

    if not question:
        output["error"] = "Missing question."
        return output

    if not generated_answer:
        output["error"] = "Missing generated_answer."
        return output

    if not top_sources:
        output["error"] = "Missing top_sources."
        return output

    claims = extract_claims(question, generated_answer)
    time.sleep(SLEEP_BETWEEN_CALLS)

    if not claims:
        output["error"] = "No claims extracted."
        return output

    final_claims = []

    for claim in claims:
        ranked_evidence = matcher.rank_sources_for_claim(
            claim_text=claim["claim_text"],
            top_sources=top_sources,
            top_n=TOP_EVIDENCE_PER_CLAIM,
        )

        best_score = ranked_evidence[0]["score"] if ranked_evidence else 0.0

        claim_result = {
            "claim_text": claim["claim_text"],
            "claim_type": claim["claim_type"],
            "priority": claim["priority"],
            "needs_verification": claim["needs_verification"],
            "best_score": float(best_score),
            "top_evidence": [
                {
                    "score": ev["score"],
                    "source_label": ev["source_label"],
                    "source_text": ev["source_text"],
                    "source_meta": ev["source_meta"],
                }
                for ev in ranked_evidence
            ],
            "verification": None,
            "final_label": None,
        }

        # No routing. No threshold shortcut. Every claim is verified by DeepSeek.
        verification = verify_claim_against_top_sources(
            question=question,
            claim_text=claim["claim_text"],
            ranked_evidence=ranked_evidence,
        )
        time.sleep(SLEEP_BETWEEN_CALLS)

        claim_result["verification"] = {
            "label": verification["label"],
            "reason": verification["reason"],
            "explanation_type": verification["explanation_type"],
            "used_evidence_indices": verification["used_evidence_indices"],
        }
        claim_result["final_label"] = verification["label"]

        if INCLUDE_HIDDEN_INTERNAL_FIELDS:
            claim_result["hidden_internal"] = {
                "raw_verifier_label": verification.get("hidden_internal", {}).get("raw_label"),
                "raw_verifier_reason": verification.get("hidden_internal", {}).get("raw_reason"),
                "collapse_rule": verification.get("hidden_internal", {}).get("collapse_rule"),
            }

        final_claims.append(claim_result)

    output["claims"] = final_claims
    output["answer_summary"] = summarize_answer_level(final_claims)
    output["xai_rows"] = build_xai_table_rows(final_claims)

    return output


# =========================================================
# MAIN WITH RESUME MODE
# =========================================================

def main():
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is missing. Set it as an environment variable first.")

    rows = read_jsonl(INPUT_JSONL)
    matcher = ClaimEvidenceMatcher()

    completed_ids = set()

    if os.path.exists(OUTPUT_JSONL):
        existing_outputs = read_jsonl(OUTPUT_JSONL)
        for row in existing_outputs:
            if row.get("idx") is not None and not row.get("error"):
                completed_ids.add(row["idx"])

    total = len(rows)
    print(f"Loaded {total} input rows.")
    print(f"Found {len(completed_ids)} completed rows in existing output.")

    for idx, row in enumerate(rows, start=1):
        row_id = row.get("idx", idx)

        if row_id in completed_ids:
            print(f"[{idx}/{total}] Skipped (already done)")
            continue

        try:
            result = process_record(row, matcher)
            if result.get("idx") is None:
                result["idx"] = row_id

            append_jsonl(OUTPUT_JSONL, result)
            print(f"[{idx}/{total}] Done")

        except Exception as e:
            error_row = {
                "idx": row_id,
                "question": row.get("question", ""),
                "generated_answer": row.get("generated_answer", row.get("answer", row.get("response", ""))),
                "claims": [],
                "answer_summary": {},
                "xai_rows": [],
                "error": str(e),
            }
            append_jsonl(OUTPUT_JSONL, error_row)
            print(f"[{idx}/{total}] Error: {e}")

    if SAVE_PRETTY_JSON_COPY:
        final_outputs = read_jsonl(OUTPUT_JSONL)
        write_pretty_json(PRETTY_JSON_OUTPUT, final_outputs)
        print(f"Saved pretty JSON to: {PRETTY_JSON_OUTPUT}")


if __name__ == "__main__":
    main()
