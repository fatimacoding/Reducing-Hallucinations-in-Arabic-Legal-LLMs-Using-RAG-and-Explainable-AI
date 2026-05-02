from abc import abstractmethod
from typing import List, Optional, Tuple

from ragex_framework.dto import ExplanationGranularity, ExplanationDto, FeatureImportance
from ragex_framework.explainer import Explainer
from ragex_framework.modules.comparator import Comparator
from ragex_framework.modules.perturber import Perturber
from ragex_framework.modules.tokenizer import Tokenizer
from ragex_framework.modules.tokenizer.arabic_legal_tokenizer import ArabicLegalTokenizer
from ragex_framework.utils import sort_similarity_scores


class GenericExplainer(Explainer):
    """
    Mirrors generic_explainer.py from the original RAG-Ex repo exactly.
    Same abstract methods, same explain() orchestration flow.
    """

    def __init__(
        self,
        perturber: Perturber,
        comparator: Comparator,
        tokenizer: Tokenizer = None,
        num_threads: Optional[int] = 10,
    ):
        self.tokenizer = tokenizer or ArabicLegalTokenizer()
        self.perturber = perturber
        self.comparator = comparator
        self.num_threads = num_threads

    @abstractmethod
    def get_reference(self, input_text: str): ...

    @abstractmethod
    def get_features(
        self,
        input_text: str,
        reference_text: str,
        reference_score: float,
        granularity: ExplanationGranularity,
    ): ...

    @abstractmethod
    def get_perturbations(
        self, input_text: str, reference_text: str, features: List[str]
    ): ...

    @abstractmethod
    def get_post_perturbation_results(
        self, input_text: str, perturbations: List[str]
    ): ...

    @abstractmethod
    def get_comparator_scores(
        self,
        reference_text: str,
        reference_score: float,
        results: list,
        do_normalize_scores: bool,
    ): ...

    def explain(
        self,
        user_input: str,
        granularity: ExplanationGranularity,
        model_name: Optional[str] = None,
        do_normalize_comparator_scores: bool = True,
        reference_text: Optional[str] = None,
        reference_score: Optional[float] = None,
    ) -> ExplanationDto:
        features = self.get_features(
            input_text=user_input,
            reference_text=reference_text,
            reference_score=reference_score,
            granularity=granularity,
        )

        perturbations = self.get_perturbations(
            input_text=user_input,
            reference_text=reference_text,
            features=features,
        )

        responses = self.get_post_perturbation_results(
            input_text=user_input,
            perturbations=perturbations,
        )

        scores = self.get_comparator_scores(
            reference_text=reference_text,
            reference_score=reference_score,
            results=responses,
            do_normalize_scores=do_normalize_comparator_scores,
        )

        return self._build_dto(features=features, scores=scores, input_text=user_input)

    def _build_dto(
        self, features: List[str], scores: List[float], input_text: str,
        output_text: str = None,
    ) -> ExplanationDto:
        features, scores = sort_similarity_scores(features, scores)
        return ExplanationDto(
            explanations=[
                FeatureImportance(feature=f, score=s)
                for f, s in zip(features, scores)
            ],
            input_text=input_text,
            output_text=output_text,
        )
