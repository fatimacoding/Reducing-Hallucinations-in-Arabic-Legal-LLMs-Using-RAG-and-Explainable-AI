from typing import List

from ragex_framework.modules.perturber import Perturber


class LeaveOneOutPerturber(Perturber):
    """Mirrors leave_one_out_perturber.py exactly."""

    def perturb(self, text: str, features: List[str]) -> List[str]:
        perturbations = []
        for token in features:
            perturbations.append(text.replace(token, "").strip())
        return perturbations
