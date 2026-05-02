from typing import List

from ragex_framework.modules.comparator import Comparator
from ragex_framework.modules.comparator.embedding_comparator import EmbeddingComparator
from ragex_framework.modules.comparator.generic_comparator import LevenshteinComparator
from ragex_framework.modules.comparator.n_gram_overlap_comparator import NGramOverlapComparator
from ragex_framework.utils import normalize_scores


class LegalHybridComparator(Comparator):
    """
    Weighted blend of semantic + syntactic comparators for Arabic legal text.

    Weights:
      0.75 × Semantic (BGE-M3 embedding cosine similarity)
      0.15 × N-gram Jaccard overlap (bigram)
      0.10 × Levenshtein (normalised edit distance)

    The high embedding weight reflects BERTScore findings (Zhang et al., 2019)
    that contextual embeddings correlate most strongly with human judgment
    (Pearson r ≈ 0.73 on WMT18). Syntactic components (n-gram + Levenshtein)
    serve as auxiliary signals to catch surface-level changes (e.g. negation
    flips) that pure semantic similarity may underestimate.
    """

    def __init__(self, encoder=None,
                 sem_weight: float = 0.75,
                 ng_weight: float = 0.15,
                 lev_weight: float = 0.10):
        self._sem = EmbeddingComparator(encoder=encoder)
        self._lev = LevenshteinComparator()
        self._ng  = NGramOverlapComparator(n=2)
        self._w = (sem_weight, ng_weight, lev_weight)

    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        sem = self._sem.compare(reference_text, texts, do_normalize_scores=False)
        ng  = self._ng.compare(reference_text, texts, do_normalize_scores=False)
        lev = self._lev.compare(reference_text, texts, do_normalize_scores=False)
        ws, wn, wl = self._w
        combined = [ws * s + wn * n + wl * l for s, n, l in zip(sem, ng, lev)]
        if do_normalize_scores:
            combined = normalize_scores(combined)
        return combined
