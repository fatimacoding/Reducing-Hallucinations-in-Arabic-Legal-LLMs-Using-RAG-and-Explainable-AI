import re
import hashlib
from abc import abstractmethod
from typing import List, Dict
from dataclasses import dataclass

from ragex_framework.modules.perturber import Perturber


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


@dataclass
class PerturbationContext:
    question: str
    target_sentence: str
    previous_sentence: str
    next_sentence: str
    strategy_name: str
    strategy_instruction: str


_CACHE: Dict[str, str] = {}


def clear_perturbation_cache():
    _CACHE.clear()
    print("Perturbation cache cleared.")


def _hash_key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _extract_text(raw: str) -> str:
    raw = _normalize_ws(raw)
    raw = re.sub(r'^["\'\`]+|["\'\`]+$', '', raw).strip()
    raw = re.sub(r'^(الناتج|الجملة المعدلة|الجملة المعدّلة|النتيجة)\s*[:：]\s*', '', raw)
    return raw.strip()


def _changed_ratio(a: str, b: str) -> float:
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 0.0
    mx = max(len(wa), len(wb), 1)
    overlap = sum(1 for x, y in zip(wa, wb) if x == y)
    return (mx - overlap) / mx


def _is_valid(original: str, perturbed: str) -> bool:
    original, perturbed = _normalize_ws(original), _normalize_ws(perturbed)
    if not perturbed:
        return False
    if len(perturbed) > 2.5 * len(original):
        return False
    if _changed_ratio(original, perturbed) > 0.55:
        return False
    return True


def _llm_perturb(ctx: PerturbationContext, deepseek_client, timeout: int = 20) -> str:
    original = _normalize_ws(ctx.target_sentence)
    if not original:
        return original

    key = _hash_key(ctx.strategy_name, ctx.question, ctx.previous_sentence,
                    ctx.target_sentence, ctx.next_sentence, ctx.strategy_instruction)
    if key in _CACHE:
        return _CACHE[key]

    system_prompt = f"""\
أنت أداة اضطراب نصوص قانونية عربية لأغراض بحث الذكاء الاصطناعي التفسيري (XAI).

المهمة:
نفّذ تعديلاً واحداً محدداً على الجملة المستهدفة حسب التعليمات أدناه.

{ctx.strategy_instruction}

قواعد صارمة ملزمة:
- عدّل الجملة المستهدفة فقط.
- لا تضف شرحاً أو ملاحظات أو علامات اقتباس.
- أخرج الجملة المعدّلة فقط — سطر واحد فقط."""

    user_prompt = f"""\
الجملة السابقة: {ctx.previous_sentence or 'لا يوجد'}
الجملة المستهدفة: {ctx.target_sentence}
الجملة التالية: {ctx.next_sentence or 'لا يوجد'}"""

    try:
        r = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=250,
            timeout=timeout,
        )
        out = _extract_text(r.choices[0].message.content or "")
        if not _is_valid(original, out):
            out = original
        _CACHE[key] = out
        return out
    except Exception as e:
        print(f"      {ctx.strategy_name}: {type(e).__name__} — fallback")
        return original


class LLMBasedPerturber(Perturber):
    """
    Mirrors llm_based_perturber.py from the original RAG-Ex repo.
    Uses DeepSeek with inline Arabic prompts instead of template files.
    Core logic: for each feature, generate perturbed version, then text.replace().
    """

    INSTRUCTION: str = ""

    def __init__(self, deepseek_client):
        self._client = deepseek_client
        self.current_question: str = ""

    def prepare(self, question: str):
        self.current_question = question or ""

    @property
    @abstractmethod
    def name(self) -> str: ...

    def perturb(self, text: str, features: List[str]) -> List[str]:
        perturbations = []
        for i, feature in enumerate(features):
            ctx = PerturbationContext(
                question=self.current_question,
                target_sentence=feature,
                previous_sentence=features[i - 1] if i > 0 else "",
                next_sentence=features[i + 1] if i < len(features) - 1 else "",
                strategy_name=self.name,
                strategy_instruction=self.INSTRUCTION,
            )
            response = _llm_perturb(ctx, self._client)
            perturbations.append(text.replace(feature, response).strip())
        return perturbations


class RandomNoisePerturber(LLMBasedPerturber):
    INSTRUCTION = """\
أدخل كلمتين عربيتين غير قانونيتين داخل الجملة.

قواعد إضافية:
- لا تحذف أو تغيّر أي كلمة أصلية.
- أدخل كلمتين فقط.
- وزّعهما في موضعين مختلفين داخل الجملة."""

    @property
    def name(self): return "Random Noise"


class EntityManipulationPerturber(LLMBasedPerturber):
    INSTRUCTION = """\
غيّر كياناً واحداً فقط داخل الجملة (مثل رقم، مبلغ، جهة، أو نوع شركة) إلى قيمة مختلفة من نفس النوع.

قواعد إضافية:
- يجب تغيير عنصر واحد فقط.
- لا تغيّر أي جزء آخر من الجملة."""

    @property
    def name(self): return "Entity Manipulation"


class AntonymInjectionPerturber(LLMBasedPerturber):
    INSTRUCTION = """\
اعكس الحكم القانوني للجملة بتغيير كلمة واحدة أو عبارة قصيرة فقط.

قواعد إضافية:
- يجب أن يتغير المعنى القانوني بوضوح.
- لا تغيّر أي كلمات أخرى."""

    @property
    def name(self): return "Antonym Injection"


class SynonymInjectionPerturber(LLMBasedPerturber):
    INSTRUCTION = """\
استبدل كلمة واحدة فقط بمرادف قانوني مناسب.

قواعد إضافية:
- يجب أن يبقى المعنى القانوني نفسه.
- لا تغيّر أي عنصر آخر."""

    @property
    def name(self): return "Synonym Injection"
