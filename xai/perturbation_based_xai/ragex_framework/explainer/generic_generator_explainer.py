from typing import List, Callable, Optional

from ragex_framework.dto import ExplanationGranularity
from ragex_framework.explainer.generic_explainer import GenericExplainer
from ragex_framework.modules.comparator import Comparator
from ragex_framework.modules.perturber import Perturber
from ragex_framework.modules.tokenizer import Tokenizer
from ragex_framework.modules.tokenizer.arabic_legal_tokenizer import ArabicLegalTokenizer


class GenericGeneratorExplainer(GenericExplainer):
    """
    Mirrors generic_generator_explainer.py from the original RAG-Ex repo.

    Original uses a Generator object with generator.generate(texts=[...]).
    We accept a generate_fn callable so we can plug in DeepSeek from the notebook.
    """

    def __init__(
        self,
        perturber: Perturber,
        comparator: Comparator,
        generate_fn: Callable[[str], str],
        tokenizer: Tokenizer = None,
    ):
        super().__init__(
            tokenizer=tokenizer or ArabicLegalTokenizer(),
            perturber=perturber,
            comparator=comparator,
        )
        self._generate_fn = generate_fn

    def get_features(
        self,
        input_text: str,
        reference_text: str,
        reference_score: float,
        granularity: ExplanationGranularity,
    ):
        # Original: tokenizes the input_text (the prompt with context)
        features = self.tokenizer.tokenize(text=input_text, granularity=granularity)
        return features

    def get_perturbations(
        self,
        input_text: str,
        reference_text: str,
        features: List[str],
    ):
        # Original: self.perturber.perturb(text=input_text, features=features)
        perturbations = self.perturber.perturb(text=input_text, features=features)
        return perturbations

    def get_reference(self, input_text: str):
        # Original: self.generator.generate(texts=[input_text])[0]
        return self._generate_fn(input_text)

    def get_post_perturbation_results(
        self,
        input_text: str = None,
        perturbations: List[str] = None,
    ):
        # Original: self.generator.generate(texts=perturbations)
        responses = []
        for p in perturbations:
            responses.append(self._generate_fn(p))
        return responses

    def get_comparator_scores(
        self,
        reference_text: str,
        reference_score: float,
        results: list,
        do_normalize_scores: bool,
    ):
        # Original: self.comparator.compare(reference_text=..., texts=..., ...)
        scores = self.comparator.compare(
            reference_text=reference_text,
            texts=results,
            do_normalize_scores=do_normalize_scores,
        )
        return scores
