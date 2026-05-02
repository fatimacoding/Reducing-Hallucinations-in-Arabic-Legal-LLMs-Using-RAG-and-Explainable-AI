import re
from typing import List, Callable

from ragex_framework.modules.comparator import Comparator
from ragex_framework.utils import normalize_scores, reverse_scores


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


class GenericComparator(Comparator):
    """
    Mirrors generic_comparator.py from the original RAG-Ex repo.
    Original wraps textdistance functions; we use pure-Python implementations
    so there are no extra dependencies in Colab.
    """

    def __init__(self, similarity_fn: Callable[[str, str], float]):
        self._similarity_fn = similarity_fn

    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        scores = [
            self._similarity_fn(_normalize_ws(reference_text), _normalize_ws(t))
            for t in texts
        ]
        if do_normalize_scores:
            scores = normalize_scores(scores)
        return reverse_scores(scores)


# ── Pure-Python similarity functions ──────────────────────────────────────

def _levenshtein_similarity(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j in range(1, len(b) + 1):
        curr = [j] + [0] * len(a)
        for i in range(1, len(a) + 1):
            curr[i] = min(
                curr[i - 1] + 1, prev[i] + 1,
                prev[i - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            )
        prev = curr
    return 1.0 - prev[len(a)] / max(len(a), len(b), 1)


def _jaro_winkler_similarity(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    match_dist = max(len(a), len(b)) // 2 - 1
    a_matches = [False] * len(a)
    b_matches = [False] * len(b)
    matches = transpositions = 0
    for i in range(len(a)):
        lo, hi = max(0, i - match_dist), min(len(b), i + match_dist + 1)
        for j in range(lo, hi):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = b_matches[j] = True
            matches += 1
            break
    if not matches:
        return 0.0
    k = 0
    for i in range(len(a)):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    jaro = (matches / len(a) + matches / len(b) + (matches - transpositions / 2) / matches) / 3
    prefix = sum(1 for i in range(min(4, min(len(a), len(b)))) if a[i] == b[i])
    return jaro + prefix * 0.1 * (1 - jaro)


class LevenshteinComparator(GenericComparator):
    def __init__(self):
        super().__init__(_levenshtein_similarity)


class JaroWinklerComparator(GenericComparator):
    def __init__(self):
        super().__init__(_jaro_winkler_similarity)
