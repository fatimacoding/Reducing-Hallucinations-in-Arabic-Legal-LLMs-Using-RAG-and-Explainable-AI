import re
from typing import List

from ragex_framework.modules.comparator import Comparator
from ragex_framework.utils import normalize_scores


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


class NGramOverlapComparator(Comparator):
    """
    Mirrors n_gram_overlap_comparator.py from the original RAG-Ex repo.
    Original uses nltk.metrics.distance.jaccard_distance + nltk.util.ngrams.
    We use a pure-Python implementation (no NLTK dependency needed).
    """

    def __init__(self, n: int = 2):
        self._n = n

    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        distances = [
            self._distance(_normalize_ws(reference_text), _normalize_ws(t))
            for t in texts
        ]
        if do_normalize_scores:
            distances = normalize_scores(distances)
        # Note: distances are already dissimilarity (Jaccard distance),
        # so we do NOT reverse_scores here (same as original).
        return distances

    def _distance(self, ref: str, text: str) -> float:
        a, b = self._ngrams(ref), self._ngrams(text)
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        try:
            return 1.0 - len(a & b) / len(a | b)
        except ZeroDivisionError:
            return 0.0

    def _ngrams(self, text: str) -> set:
        words = text.split()
        if len(words) < self._n:
            return {tuple(words)} if words else set()
        return set(tuple(words[i:i + self._n]) for i in range(len(words) - self._n + 1))
