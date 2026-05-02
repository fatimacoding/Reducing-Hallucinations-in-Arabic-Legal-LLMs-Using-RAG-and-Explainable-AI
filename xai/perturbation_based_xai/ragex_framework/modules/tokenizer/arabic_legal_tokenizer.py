import re
from typing import List

from ragex_framework.dto import ExplanationGranularity
from ragex_framework.modules.tokenizer import Tokenizer


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


class ArabicLegalTokenizer(Tokenizer):
    """
    Arabic legal text tokenizer.
    Replaces CustomTokenizer (which used spaCy for EN/DE + lingua for detection).
    For Arabic legal text, sentence-level is the strongest practical choice
    because each article/clause is typically one self-contained sentence.
    """

    def tokenize(self, text: str, granularity: ExplanationGranularity) -> List[str]:
        if not text or not text.strip():
            return []

        if granularity == ExplanationGranularity.SENTENCE_LEVEL:
            return self._sent_tokenize(text)
        elif granularity == ExplanationGranularity.WORD_LEVEL:
            return self._word_tokenize(text)
        elif granularity == ExplanationGranularity.PARAGRAPH_LEVEL:
            return self._paragraph_tokenize(text)
        elif granularity == ExplanationGranularity.PHRASE_LEVEL:
            return self._phrase_tokenize(text)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

    def _sent_tokenize(self, text: str) -> List[str]:
        text = text.replace("\r", "\n")
        raw = re.split(r'(?<=[\\.۔؛?!؟\n])\s+', text)
        out = []
        for part in raw:
            sub = re.split(r'(?<=\d)\s*[-–]\s+', part)
            for s in sub:
                s = _normalize_ws(s)
                if s:
                    out.append(s)
        return out

    def _word_tokenize(self, text: str) -> List[str]:
        return [w for w in re.findall(r'[\u0600-\u06FF\u0750-\u077F]+', text) if w.strip()]

    def _paragraph_tokenize(self, text: str) -> List[str]:
        text = re.sub(r"\n+", "\n", text)
        return [_normalize_ws(p) for p in text.split("\n") if p.strip()]

    def _phrase_tokenize(self, text: str) -> List[str]:
        phrases = re.split(r'\s+(?:و|أو|ثم|في|من|على|إلى|عن|بين)\s+', text)
        return [_normalize_ws(p) for p in phrases if p.strip()]
