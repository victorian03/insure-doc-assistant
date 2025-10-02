# src/embedder.py
from typing import List
from .config import Settings

class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = (settings.EMBEDDING_PROVIDER or "local").lower()
        self.model_name = settings.EMBEDDING_MODEL or "sentence-transformers/paraphrase-MiniLM-L3-v2"
        self._model = None        # pentru local (SentenceTransformers)
        self._openai = None       # pentru openai (lazy)

    def _ensure_model(self):
        if self.provider == "local":
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
        elif self.provider == "openai":
            if self._openai is None:
                # importăm doar dacă chiar folosim OpenAI
                from openai import OpenAI
                self._openai = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        if self.provider == "local":
            return self._model.encode(
                texts,
                batch_size=16,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()
        else:
            model = self.model_name or "text-embedding-3-small"
            out = self._openai.embeddings.create(model=model, input=texts)
            return [d.embedding for d in out.data]
