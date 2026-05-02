from typing import List

from ragex_framework.modules.comparator import Comparator
from ragex_framework.utils import normalize_scores, reverse_scores


class EmbeddingComparator(Comparator):
    """
    Cosine similarity via sentence embeddings.
    Mirrors the original EmbeddingComparator which takes an Encoder.
    We use BGE-M3 (the `embedder` variable from the notebook).
    """

    def __init__(self, encoder=None):
        # encoder is the BGE-M3 SentenceTransformer loaded in the notebook.
        # If None, the demo cell must pass `embedder` at construction time.
        self._encoder = encoder

    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        all_texts = texts + [reference_text]
        try:
            embs = self._encoder.encode(
                all_texts, normalize_embeddings=True, convert_to_tensor=True
            )
            ref_vec = embs[-1]
            scores = []
            for i in range(len(texts)):
                sim = float((embs[i] @ ref_vec).item())
                scores.append(max(0.0, min(1.0, sim)))
        except Exception as e:
            print(f"    ⚠ EmbeddingComparator: {type(e).__name__}")
            scores = [0.0] * len(texts)

        if do_normalize_scores:
            scores = normalize_scores(scores)
        return reverse_scores(scores)
