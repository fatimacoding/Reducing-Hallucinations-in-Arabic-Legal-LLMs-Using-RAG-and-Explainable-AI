from typing import List, Tuple


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalization to [0, 1].
    When there is only one score, return it as-is (no normalization possible)."""
    if not scores:
        return scores
    if len(scores) == 1:
        return scores  # Cannot min-max normalize a single value
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [scores[0]] * len(scores)  # All same — return raw value
    return [(s - mn) / (mx - mn) for s in scores]


def reverse_scores(scores: List[float]) -> List[float]:
    """Invert scores: 1 - s."""
    return [1.0 - s for s in scores]


def sort_similarity_scores(
    features: List[str], scores: List[float]
) -> Tuple[List[str], List[float]]:
    """Sort features by score descending (most important first)."""
    paired = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    if not paired:
        return [], []
    feats, scs = zip(*paired)
    return list(feats), list(scs)
