from typing import List
from sentence_transformers import SentenceTransformer
import openai

from .config import Settings

class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.EMBEDDING_PROVIDER.lower()
        self.model_name = settings.EMBEDDING_MODEL
        self._model = None
        self._openai = None

    def _ensure_model(self):
        if self.provider == "local" and self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        elif self.provider == "openai" and self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=self.settings.OPENAI_API_KEY)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        if self.provider == "local":
            # fără progress bar (să nu mai inunde consola) + batch mic stabil
            return self._model.encode(
                texts,
                batch_size=16,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()
        elif self.provider == "openai":
            model = self.model_name or "text-embedding-3-small"
            out = self._openai.embeddings.create(model=model, input=texts)
            return [d.embedding for d in out.data]
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")
