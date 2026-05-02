import json
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CORPUS_BOE = "data/articles_corpus_final_clean.jsonl"
CORPUS_MOJ = "data/saudi_legal_corpus_v4_clean.jsonl"
NAJIZ_FILE = "data/najiz_records.jsonl"
OUTPUT_FILE = "najiz_top_sources.jsonl"

ARTIFACTS_DIR     = "artifacts"
TOP_K             = None   # None => dynamic selection after reranking
MIN_K             = 3 
MAX_K             = 3
BM25_TOP_N        = 80
DENSE_TOP_N       = 80
RERANK_CANDIDATES = 120
RRF_K             = 60
DEFAULT_RRF_WEIGHTS = (1.15, 1.0)   # (semantic, bm25) default only; final weights are query-aware
USE_RERANKER      = True

BI_ENCODER_NAME = "BAAI/bge-m3"
RERANKER_NAME   = "BAAI/bge-reranker-v2-m3"

# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
WESTERN      = "0123456789"
_INDIC_TRANS = str.maketrans({a: w for a, w in zip(ARABIC_INDIC, WESTERN)})

AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
AR_TATWEEL    = re.compile(r"\u0640")
AR_SPACES     = re.compile(r"\s+")
AR_NON_WORD   = re.compile(r"[^\w\s]")

def normalize(text: str) -> str:
    text = text or ""
    text = text.translate(_INDIC_TRANS)
    text = AR_DIACRITICS.sub("", text)
    text = AR_TATWEEL.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = AR_NON_WORD.sub(" ", text)
    text = AR_SPACES.sub(" ", text).strip().lower()
    return text

def tokenize(text: str) -> List[str]:
    return normalize(text).split()

def preprocess_query(query: str) -> str:
    return (query or "").translate(_INDIC_TRANS)

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

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Corpus
# ─────────────────────────────────────────────
def load_corpus(path: str, source_label: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    docs = []

    for row in rows:
        law       = row.get("law", {})       if isinstance(row.get("law"), dict) else {}
        article   = row.get("article", {})   if isinstance(row.get("article"), dict) else {}
        retrieval = row.get("retrieval", {}) if isinstance(row.get("retrieval"), dict) else {}

        law_title = first_nonempty(
            row.get("law_title"),
            row.get("doc_title"),
            law.get("title_ar"),
            law.get("title"),
        )

        article_title = first_nonempty(
            row.get("article_title"),
            row.get("article_title_clean"),
            article.get("article_title_norm"),
            article.get("article_title_raw"),
            row.get("raw_title"),
        )

        article_num_raw = first_nonempty(
            row.get("article_number_raw"),
            article.get("article_number_raw"),
        )

        article_num_parsed = first_nonempty(
            str(row.get("article_main_no") or ""),
            str(article.get("article_number_parsed") or ""),
            str(row.get("article_number_parsed") or ""),
            str(article.get("article_index") or ""),
            str(row.get("article_index") or ""),
        )

        base_text = first_nonempty(
            row.get("text"),
            article.get("text"),
            retrieval.get("structural_text"),
            retrieval.get("lead_text"),
            retrieval.get("contextual_text"),
            retrieval.get("dense_text_final"),
            retrieval.get("dense_text"),
        )

        sparse_text = first_nonempty(
            retrieval.get("sparse_text_final"),
            retrieval.get("sparse_text"),
            retrieval.get("primary_retrieval_text"),
            base_text,
        )

        dense_text = first_nonempty(
            retrieval.get("dense_text_final"),
            retrieval.get("dense_text"),
            retrieval.get("contextual_text"),
            base_text,
            sparse_text,
        )

        citation = first_nonempty(
            row.get("citation"),
            f"{law_title}، {article_title}".strip("، "),
        )

        embed_text = f"{law_title} {article_title} {base_text[:700]}"

        rerank_text = "\n".join([
            f"[النظام] {law_title}",
            f"[المادة] {article_title}",
            f"[رقم المادة] {article_num_raw or article_num_parsed}",
            f"[النص] {base_text}",
        ]).strip()

        docs.append({
            "doc_id":              str(row.get("id", "")),
            "source":              source_label,
            "law":                 law_title,
            "article":             article_title,
            "article_num_raw":     article_num_raw,
            "article_num_parsed":  article_num_parsed,
            "citation":            citation,
            "text":                base_text,
            "sparse_text":         sparse_text,
            "dense_text":          dense_text,
            "embed_text":          embed_text,
            "rerank_text":         rerank_text,
        })

    return docs

# ─────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────
def build_bm25(docs: List[Dict[str, Any]], source_name: str):
    ensure_dir(ARTIFACTS_DIR)
    bm25_path = Path(ARTIFACTS_DIR) / f"{source_name}_bm25.pkl"

    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            print(f"Loaded BM25 cache for {source_name}")
            return pickle.load(f)

    tokens = [tokenize(d["sparse_text"]) for d in tqdm(docs, desc=f"BM25 [{source_name}]")]
    bm25 = BM25Okapi(tokens)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Saved BM25 cache for {source_name}")
    return bm25

# ─────────────────────────────────────────────
# FAISS
# ─────────────────────────────────────────────
def build_faiss(docs: List[Dict[str, Any]], source_name: str, embed_model):
    ensure_dir(ARTIFACTS_DIR)
    index_path = Path(ARTIFACTS_DIR) / f"{source_name}_bge.index"

    if index_path.exists():
        print(f"Loaded FAISS cache for {source_name}")
        return faiss.read_index(str(index_path))

    texts = [d["embed_text"] for d in docs]
    embeddings = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    print(f"Saved FAISS cache for {source_name}")
    return index

# ─────────────────────────────────────────────
# RRF
# ─────────────────────────────────────────────
def rrf_fuse(
    sem_ids: List[int],
    bm25_ids: List[int],
    k: int = RRF_K,
    weights: tuple = DEFAULT_RRF_WEIGHTS,
) -> tuple:
    scores: Dict[int, float] = {}

    for w, lst in zip(weights, [sem_ids, bm25_ids]):
        for r, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + r)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return sorted_ids, scores

import re


def has_digit_signal(query: str) -> bool:
    return any(ch.isdigit() for ch in query)


def get_rrf_weights(query: str, **_) -> tuple:
    if has_digit_signal(query):
        return (1.0, 1.15)
    return DEFAULT_RRF_WEIGHTS


def _score_for_dedup(c: Dict[str, Any]) -> float:
    if "rerank_score" in c and c["rerank_score"] is not None:
        return float(c["rerank_score"])
    return float(c.get("rrf_score", 0.0))


DEDUP_SIM_THRESHOLD = 0.85
DEDUP_EPS = 1e-6
SOURCE_PRIORITY = {"moj": 1, "boe": 0}


def _rank_for_dedup(c: Dict[str, Any]) -> int:
    ranks = [r for r in (c.get("bm25_rank"), c.get("dense_rank")) if isinstance(r, int)]
    return min(ranks) if ranks else 10**9


def _source_priority(c: Dict[str, Any]) -> int:
    src = str(c.get("doc", {}).get("source", "")).lower()
    return SOURCE_PRIORITY.get(src, 99)


def _doc_text_for_dedup(doc: Dict[str, Any]) -> str:
    text = first_nonempty(
        doc.get("text"),
        doc.get("rerank_text"),
        doc.get("citation"),
    )
    return normalize(text)


def _lexical_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_better_candidate(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    cand_score = _score_for_dedup(candidate)
    curr_score = _score_for_dedup(current)

    if cand_score > curr_score + DEDUP_EPS:
        return True
    if cand_score < curr_score - DEDUP_EPS:
        return False

    cand_rank = _rank_for_dedup(candidate)
    curr_rank = _rank_for_dedup(current)
    if cand_rank < curr_rank:
        return True
    if cand_rank > curr_rank:
        return False

    return _source_priority(candidate) < _source_priority(current)


def dedup_same_article_candidates(
    candidates: List[Dict[str, Any]],
    threshold: float = DEDUP_SIM_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Collapse near-duplicate BOE/MOJ results after fusion/rerank using lexical
    similarity on normalized article text. Keep only the best version.
    """
    kept: List[Dict[str, Any]] = []

    for cand in candidates:
        cand = dict(cand)
        doc = dict(cand["doc"])
        cand["doc"] = doc

        cand_text = _doc_text_for_dedup(doc)
        matched_idx = None
        matched_sim = 0.0

        for i, existing in enumerate(kept):
            sim = _lexical_similarity(cand_text, _doc_text_for_dedup(existing["doc"]))
            if sim >= threshold:
                matched_idx = i
                matched_sim = sim
                break

        if matched_idx is None:
            cand["duplicate_count"] = 1
            cand["duplicate_sources"] = [doc.get("source")] if doc.get("source") else []
            cand["is_duplicate_article"] = False
            cand["duplicate_in_moj"] = str(doc.get("source", "")).lower() == "moj"
            cand["duplicate_similarity"] = 1.0
            kept.append(cand)
            continue

        existing = kept[matched_idx]

        merged_sources = list(dict.fromkeys(
            (existing.get("duplicate_sources") or [existing["doc"].get("source")]) +
            ([doc.get("source")] if doc.get("source") else [])
        ))
        merged_sources = [s for s in merged_sources if s]

        better = cand if _is_better_candidate(cand, existing) else existing
        better["duplicate_count"] = len(merged_sources)
        better["duplicate_sources"] = merged_sources
        better["is_duplicate_article"] = len(merged_sources) > 1
        better["duplicate_in_moj"] = any((s or "").lower() == "moj" for s in merged_sources)
        better["duplicate_similarity"] = matched_sim

        kept[matched_idx] = better

    return kept

def dynamic_select(
    candidates: List[Dict[str, Any]],
    min_k: int = 1,
    max_k: int = MAX_K,
    gap_threshold: float = 0.04,
    relative_to_top: float = 0.90,
    min_score: Optional[float] = None,
    fallback_k_if_one: int = 3,
):
    if not candidates:
        return [], False, None

    # انسخ candidates عشان نقدر نضيف عليها metadata
    candidates = [dict(c) for c in candidates]

    selected = [candidates[0]]
    selected[0]["added_by"] = "normal"

    top_score = float(candidates[0].get("rerank_score", candidates[0].get("rrf_score", 0.0)))
    prev_score = top_score

    if min_score is None:
        min_score = max(top_score * 0.80, top_score - 0.08)

    warning_reason = None
    used_fallback = False

    for cand in candidates[1:]:
        score = float(cand.get("rerank_score", cand.get("rrf_score", 0.0)))
        gap = prev_score - score
        rel = score / (top_score + 1e-8)

        keep = (gap <= gap_threshold) and (rel >= relative_to_top) and (score >= min_score)

        if keep:
            cand["added_by"] = "normal"
            selected.append(cand)
            prev_score = score
            if len(selected) >= max_k:
                break
        else:
            break

    # fallback: إذا رجّع أقل من 3، كمّلها لأول 3
    if len(selected) < fallback_k_if_one and len(candidates) >= fallback_k_if_one:
        used_fallback = True
        fallback_selected = []

        for i, cand in enumerate(candidates[:fallback_k_if_one]):
            cand = dict(cand)
            if i < len(selected):
                cand["added_by"] = "normal"
            else:
                cand["added_by"] = "fallback"
            fallback_selected.append(cand)

        return fallback_selected, used_fallback, None

    return selected, used_fallback, None
# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
def retrieve_from_source(
    query: str,
    docs: List[Dict[str, Any]],
    bm25,
    index,
    embed_model,
    bm25_top_n: int = BM25_TOP_N,
    dense_top_n: int = DENSE_TOP_N,
):
    q_tokens = tokenize(query)

    # BM25
    bm25_scores = bm25.get_scores(q_tokens).copy()
    bm25_ids = list(map(int, np.argsort(bm25_scores)[::-1][:bm25_top_n]))

    # Dense
    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dense_scores_raw, dense_ids_raw = index.search(q_emb, dense_top_n)
    dense_ids = [int(i) for i in dense_ids_raw[0] if int(i) >= 0]
    dense_score_map = {
        int(dense_ids_raw[0][i]): float(dense_scores_raw[0][i])
        for i in range(len(dense_ids))
    }

    # Fuse with query-aware RRF weights
    rrf_weights = get_rrf_weights(query)
    
    fused_ids, rrf_scores = rrf_fuse(dense_ids, bm25_ids, k=RRF_K, weights=rrf_weights)

    bm25_rank_map  = {v: r + 1 for r, v in enumerate(bm25_ids)}
    dense_rank_map = {v: r + 1 for r, v in enumerate(dense_ids)}

    candidates = []
    for i in fused_ids:
        candidates.append({
            "doc":         docs[i],
            "bm25_rank":   bm25_rank_map.get(i),
            "bm25_score":  float(bm25_scores[i]),
            "dense_rank":  dense_rank_map.get(i),
            "dense_score": dense_score_map.get(i),
            "rrf_score":   round(rrf_scores[i], 6),
            "rrf_weights": rrf_weights,
        })
    return candidates

# ─────────────────────────────────────────────
# Reranker
# ─────────────────────────────────────────────
def load_reranker():
    try:
        return CrossEncoder(RERANKER_NAME, max_length=512)
    except Exception:
        return CrossEncoder(RERANKER_NAME)

def global_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    reranker,
    use_reranker: bool = True,
    top_k: Optional[int] = TOP_K,
):
    if not use_reranker or reranker is None:
        ranked = sorted(candidates, key=lambda x: x["rrf_score"], reverse=True)
        ranked = dedup_same_article_candidates(ranked)

        if top_k is not None:
            return ranked[:top_k], False, None

        selected, used_fallback, warning_reason = dynamic_select(ranked)
        return selected, used_fallback, warning_reason

    pairs = [(query, c["doc"]["rerank_text"]) for c in candidates]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    candidates = dedup_same_article_candidates(candidates)

    if top_k is not None:
        return candidates[:top_k], False, None

    selected, used_fallback, warning_reason = dynamic_select(candidates)
    return selected, used_fallback, warning_reason
# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
class HybridRetrievalPipeline:
    def __init__(
        self,
        corpus_boe: str = CORPUS_BOE,
        corpus_moj: str = CORPUS_MOJ,
        bi_encoder_name: str = BI_ENCODER_NAME,
        reranker_name: str = RERANKER_NAME,
        use_reranker: bool = USE_RERANKER,
    ):
        print("Loading corpora...")
        self.docs_boe = load_corpus(corpus_boe, "boe")
        self.docs_moj = load_corpus(corpus_moj, "moj")
        print(f"BOE: {len(self.docs_boe)} | MOJ: {len(self.docs_moj)}")

        print("\nLoading bi-encoder...")
        self.embed_model = SentenceTransformer(bi_encoder_name)
        self.embed_model.max_seq_length = 256

        self.reranker = None
        self.use_reranker = use_reranker
        if self.use_reranker:
            print("Loading reranker...")
            self.reranker = load_reranker()

        print("\nBuilding / loading indexes...")
        self.bm25_boe  = build_bm25(self.docs_boe, "boe")
        self.bm25_moj  = build_bm25(self.docs_moj, "moj")
        self.faiss_boe = build_faiss(self.docs_boe, "boe", self.embed_model)
        self.faiss_moj = build_faiss(self.docs_moj, "moj", self.embed_model)

    from typing import Tuple, Optional, List, Dict, Any

    def retrieve(self, question: str, top_k: Optional[int] = TOP_K) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
        q = preprocess_query((question or "").strip())
        if not q:
            return [], False, None
        
        boe_cands = retrieve_from_source(
            q,
            self.docs_boe,
            self.bm25_boe,
            self.faiss_boe,
            self.embed_model,
            bm25_top_n=BM25_TOP_N,
            dense_top_n=DENSE_TOP_N,
        )

        moj_cands = retrieve_from_source(
            q,
            self.docs_moj,
            self.bm25_moj,
            self.faiss_moj,
            self.embed_model,
            bm25_top_n=BM25_TOP_N,
            dense_top_n=DENSE_TOP_N,
        )

        all_cands = boe_cands + moj_cands
        all_cands.sort(key=lambda x: x["rrf_score"], reverse=True)
        all_cands = all_cands[:RERANK_CANDIDATES]

        top, used_fallback, warning_reason = global_rerank(
            q,
            all_cands,
            self.reranker,
            use_reranker=self.use_reranker,
            top_k=top_k,
        )

        top_sources = []
        for rank, r in enumerate(top, start=1):
            d = r["doc"]
            row = {
                "rank": rank,
                "source": d["source"],
                "doc_id": d["doc_id"],
                "citation": d["citation"],
                "law": d["law"],
                "article": d["article"],
                "article_num_raw": d["article_num_raw"],
                "article_num_parsed": d["article_num_parsed"],
                "text": d["text"],
                "rerank_text": d["rerank_text"],
                "bm25_rank": r.get("bm25_rank"),
                "bm25_score": r.get("bm25_score"),
                "dense_rank": r.get("dense_rank"),
                "dense_score": r.get("dense_score"),
                "rrf_score": round(r.get("rrf_score", 0.0), 6),
                "rrf_weights": r.get("rrf_weights"),
                "rerank_score": round(r.get("rerank_score", 0.0), 6)
                    if "rerank_score" in r else None,
                "added_by": r.get("added_by", "normal"),
                "used_fallback": r.get("added_by") == "fallback",
                "warning_reason": warning_reason if r.get("added_by") == "fallback" else None,
                "is_duplicate_article": r.get("is_duplicate_article", False),
                "duplicate_count": r.get("duplicate_count", 1),
                "duplicate_sources": r.get("duplicate_sources", [d["source"]]),
                "duplicate_in_moj": r.get("duplicate_in_moj", False),
                "duplicate_similarity": round(r.get("duplicate_similarity", 1.0), 4),
            }
            top_sources.append(row)

        return top_sources, used_fallback, warning_reason

# ─────────────────────────────────────────────
# Save retrieval output for Najiz
# ─────────────────────────────────────────────
def run_retrieval_on_najiz(
    pipeline: HybridRetrievalPipeline,
    najiz_file: str = NAJIZ_FILE,
    output_file: str = OUTPUT_FILE,
    top_k: Optional[int] = TOP_K,
):
    najiz = read_jsonl(najiz_file)
    print(f"\nLoaded Najiz questions: {len(najiz)}")

    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, rec in enumerate(najiz, start=1):
            q_raw = (rec.get("question") or "").strip()
            if not q_raw:
                continue

            print("\n" + "=" * 100)
            print(f"[{idx}] {q_raw}")

            top_sources, used_fallback, warning_reason = pipeline.retrieve(q_raw, top_k=top_k)

            for s in top_sources:
                print(f"  [{s['rank']}] [{s['source'].upper()}] {s['citation']}")
                print(f"      rrf={s['rrf_score']}  rerank={s['rerank_score']}  weights={s['rrf_weights']}")

            fout.write(json.dumps({
            "idx": idx,
            "question": q_raw,
            "context": rec.get("context"),
            "reference_answer": rec.get("reference_answer"),
            "global_used_fallback": used_fallback,
            "global_warning_reason": warning_reason,
            "top_sources": top_sources,
        }, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nSaved retrieval output to: {output_file}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = HybridRetrievalPipeline()
    run_retrieval_on_najiz(pipeline)