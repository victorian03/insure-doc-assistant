from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import re

from .vectorstore import VectorStore

class Retriever:
    def __init__(self, vs: VectorStore):
        self.vs = vs
        self._bm25_cache = None

    def _prepare_bm25(self):
        # ensure meta is loaded
        self.vs._load()
        if self._bm25_cache is not None:
            return self._bm25_cache
        if not self.vs.exists():
            self._bm25_cache = ([], None)
            return self._bm25_cache
        texts = [d["text"] for d in self.vs._meta]
        tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
        bm25 = BM25Okapi(tokenized)
        self._bm25_cache = (texts, bm25)
        return self._bm25_cache

    def get_context(self, question: str, top_k: int = 5, rerank: bool = True, max_chars: int = 6000):
        # semantic retrieve
        sem_hits = self.vs.search(question, top_k=top_k * 3 if rerank else top_k)
        hits = sem_hits
        if rerank:
            texts, bm25 = self._prepare_bm25()
            if bm25 and texts:
                scores = bm25.get_scores(re.findall(r"\w+", question.lower()))
                # attach lexical scores
                for h in hits:
                    try:
                        i = self.vs._meta.index({"text": h["text"], "metadata": h["metadata"]})
                    except ValueError:
                        i = None
                    h["lex_score"] = float(scores[i]) if i is not None else 0.0
                # sort hybrid: semantic + 0.2 * lexical
                hits.sort(key=lambda x: (x.get("score", 0.0) + 0.2 * x.get("lex_score", 0.0)), reverse=True)

        # uniqueness by (source,page) then trim by max_chars
        seen = set()
        uniq = []
        total = 0
        for h in hits:
            key = (h["metadata"].get("source_name"), h["metadata"].get("page"))
            if key in seen:
                continue
            seen.add(key)
            if total + len(h["text"]) > max_chars:
                break
            uniq.append(h)
            total += len(h["text"])

        # build context block
        ctx_parts = []
        for h in uniq[:top_k]:
            meta = h["metadata"]
            ctx_parts.append(f"[source: {meta.get('source_name')}, page: {meta.get('page')}]\\n{h['text']}")
        context = "\n\n".join(ctx_parts)
        return context, uniq[:top_k]
