from typing import Tuple

from .config import Settings
from .prompts import INSURANCE_QA_PROMPT

class RAGChain:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.LLM_PROVIDER.lower()
        self._openai = None

    def _ensure_openai(self):
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=self.settings.OPENAI_API_KEY)

    def _llm_answer(self, question: str, context: str) -> str:
        self._ensure_openai()
        prompt = INSURANCE_QA_PROMPT.format(question=question, context=context)
        resp = self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    def _extractive_answer(self, question: str, context: str) -> str:
        # simple extractive fallback
        header = "**Răspuns (fără LLM, extractiv din surse):**\n"
        # if you want: pick only blocks with overlapping keywords, else return all context (trimmed by MAX_CONTEXT_CHARS)
        return header + context

    def answer(self, question: str, context: str) -> Tuple[str, str]:
        if self.provider == "openai" and self.settings.OPENAI_API_KEY:
            try:
                return self._llm_answer(question, context), "LLM (OpenAI)"
            except Exception:
                return self._extractive_answer(question, context), "extractive (fallback)"
        else:
            return self._extractive_answer(question, context), "extractive"
