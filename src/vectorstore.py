# src/vectorstore.py — SimpleVectorStore (numpy, fără DB)
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import json
import os
import uuid
import numpy as np

from .config import Settings
from .embedder import Embedder


META_FILE = "meta.jsonl"
EMB_FILE = "embeddings.npy"
DOC_FILE = "documents.npy"  # păstrăm și textele pentru rezultate


def _normalize(mat: np.ndarray) -> np.ndarray:
    # normalizare L2 (pentru cosine similarity)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


class VectorStore:
    """
    Un vector store minimal, robust pe Windows:
      - salvează embeddings în embeddings.npy
      - salvează textele în documents.npy (array de obiecte)
      - salvează metadata per rând în meta.jsonl
    API compatibil cu restul proiectului:
      - exists()
      - build_from_stream(docs_iter, batch_size)
      - add_batch(docs)
      - search(query, top_k)
    """
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.settings = Settings()
        self.embedder = Embedder(self.settings)

        self.meta_path = self.index_dir / META_FILE
        self.emb_path  = self.index_dir / EMB_FILE
        self.doc_path  = self.index_dir / DOC_FILE

        # lazy cache în memorie (umplem la prima căutare)
        self._emb: np.ndarray | None = None
        self._docs: np.ndarray | None = None
        self._metas: List[Dict] | None = None

    # -------- persistency helpers --------
    def exists(self) -> bool:
        return self.meta_path.exists() and self.emb_path.exists() and self.doc_path.exists()

    def _load_all(self):
        if self._emb is None and self.emb_path.exists():
            self._emb = np.load(self.emb_path)
        if self._docs is None and self.doc_path.exists():
            self._docs = np.load(self.doc_path, allow_pickle=True)
        if self._metas is None and self.meta_path.exists():
            metas: List[Dict] = []
            with self.meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    metas.append(json.loads(line))
            self._metas = metas

    def _append_persist(self, vecs: np.ndarray, texts: List[str], metas: List[Dict]):
        # embeddings
        if self.emb_path.exists():
            old = np.load(self.emb_path)
            allv = np.concatenate([old, vecs], axis=0)
        else:
            allv = vecs
        np.save(self.emb_path, allv)

        # documents (textele)
        texts_arr = np.array(texts, dtype=object)
        if self.doc_path.exists():
            old_docs = np.load(self.doc_path, allow_pickle=True)
            alld = np.concatenate([old_docs, texts_arr], axis=0)
        else:
            alld = texts_arr
        np.save(self.doc_path, alld, allow_pickle=True)

        # metadata (append line-by-line)
        with self.meta_path.open("a", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        # invalidăm cache-ul din memorie (va fi reîncărcat la search)
        self._emb = None
        self._docs = None
        self._metas = None

    # -------- building / adding --------
    def add_batch(self, docs: List[Dict]):
        if not docs:
            return
        texts = [d["text"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]
        # calculează embeddings cu providerul configurat
        vecs_list = self.embedder.embed(texts)
        vecs = np.array(vecs_list, dtype=np.float32)
        vecs = _normalize(vecs)
        self._append_persist(vecs, texts, metas)

    def build_from_stream(self, docs_iter: Iterable[Dict], batch_size: int = 64):
        batch: List[Dict] = []
        for d in docs_iter:
            batch.append(d)
            if len(batch) >= batch_size:
                self.add_batch(batch)
                batch.clear()
        if batch:
            self.add_batch(batch)

    # -------- search --------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.exists():
            return []

        self._load_all()
        assert self._emb is not None and self._docs is not None and self._metas is not None

        # embed query
        q_vec = np.array(self.embedder.embed([query])[0], dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

        # cosine similarity (matmul)
        sims = self._emb @ q_vec  # (N,)
        # top-k
        if top_k >= len(sims):
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, top_k)[:top_k]
            idx = idx[np.argsort(-sims[idx])]

        hits: List[Dict] = []
        for i in idx:
            i = int(i)
            hits.append({
                "text": str(self._docs[i]),
                "metadata": self._metas[i],
                "score": float(sims[i]),
            })
        return hits
