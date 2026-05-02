import random
from typing import List

from ragex_framework.modules.perturber import Perturber


class OrderManipulationPerturber(Perturber):
    """
    Mirrors reorder_perturber.py from the original RAG-Ex repo.
    Original uses nlpaug RandomWordAug(action="swap", aug_p=1.0).
    We do a controlled local span-move (better for Arabic legal text).
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    @property
    def name(self):
        return "Order Manipulation"

    def perturb(self, text: str, features: List[str]) -> List[str]:
        perturbations = []
        for feature in features:
            reordered = self._reorder(feature)
            perturbations.append(text.replace(feature, reordered).strip())
        return perturbations

    def _reorder(self, sentence: str) -> str:
        words = sentence.split()
        if len(words) < 4:
            return sentence
        start = self._rng.randint(0, max(0, len(words) - 3))
        end = min(len(words), start + self._rng.randint(1, 3))
        span = words[start:end]
        rest = words[:start] + words[end:]
        insert_at = self._rng.randint(0, len(rest))
        candidate = rest[:insert_at] + span + rest[insert_at:]
        out = " ".join(candidate)
        if out == sentence and len(words) >= 5:
            words[0], words[1] = words[1], words[0]
            out = " ".join(words)
        return out
