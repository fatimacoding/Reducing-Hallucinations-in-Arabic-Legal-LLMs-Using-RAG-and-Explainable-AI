from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class ExplanationGranularity(Enum):
    """Granularity levels supported by RAG-Ex (paper §3)."""
    WORD_LEVEL      = "word"
    PHRASE_LEVEL    = "phrase"
    SENTENCE_LEVEL  = "sentence"
    PARAGRAPH_LEVEL = "paragraph"


@dataclass
class FeatureImportance:
    feature: str
    score: float


@dataclass
class ExplanationDto:
    explanations: List[FeatureImportance]
    input_text: str
    output_text: Optional[str] = None
