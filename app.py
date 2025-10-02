# app.py  — Insure Doc Assistant (RAG on PDFs) – bundled samples

from pathlib import Path
from glob import glob
import shutil
import streamlit as st

from src.config import Settings
from src.utils import ensure_dirs, Timer

from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.rag_chain import RAGChain

st.set_page_config(page_title="Insure Doc Assistant", page_icon="📄", layout="wide")

# --- Paths & setup
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
OUTPUT_DIR = DATA_DIR / "output"
ensure_dirs(INDEX_DIR, OUTPUT_DIR)


def sample_pdfs():
    """Toate PDF-urile din data/samples/ (bundled cu proiectul)."""
    samples_dir = DATA_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return sorted(glob(str(samples_dir / "*.pdf")))


# --- Config din .env
settings = Settings()

# --- Page header
st.title("📄 Insure Doc Assistant – RAG on PDFs")
st.caption("PDF-urile din `data/samples/` se indexează automat (bundled). Apoi poți întreba și primești citări de surse.")

with st.sidebar:
    st.header("⚙️ Config")
    st.write(f"Embeddings: **{settings.EMBEDDING_PROVIDER}**")
    st.write(f"Model: **{settings.EMBEDDING_MODEL}**")
    st.write(f"LLM: **{settings.LLM_PROVIDER}**")
    TOP_K = settings.TOP_K
    MAX_CONTEXT_CHARS = settings.MAX_CONTEXT_CHARS

# --- Vector store + auto-index la pornire (doar dacă nu există index)
vs = VectorStore(index_dir=INDEX_DIR)
if not vs.exists():
    bundled = sample_pdfs()
    if bundled:
        with st.spinner(f"Building first index from bundled PDFs ({len(bundled)} docs)…"):
            docs = ingest_pdfs(bundled, default_meta={"doc_type": "Bundled"})
            vs.build_or_update(docs)
        st.success(f"Bundled PDFs indexed: {len(docs)} chunks from {len(bundled)} files.")
    else:
        st.info("Nu am găsit PDF-uri în `data/samples/`. Adaugă fișiere acolo sau folosește upload (dacă păstrezi secțiunea).")

# --- Optional: buton de rebuild din samples (fără upload)
st.markdown("### 📦 Bundled samples")
st.write("Indexează/reecreează indexul direct din PDF-urile incluse în proiect (`data/samples/`).")
if st.button("Rebuild index from data/samples/ (no upload)"):
    bundled = sample_pdfs()
    if not bundled:
        st.warning("Nu există PDF-uri în `data/samples/`.")
    else:
        with st.spinner(f"Rebuilding index from {len(bundled)} bundled PDFs…"):
            # șterge indexul vechi pentru a evita dublarea
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
            ensure_dirs(INDEX_DIR)
            _vs = VectorStore(index_dir=INDEX_DIR)
            docs = ingest_pdfs(bundled, default_meta={"doc_type": "Bundled"})
            _vs.build_or_update(docs)
        st.success(f"Rebuilt. Indexed {len(docs)} chunks from {len(bundled)} files.")

# --- Q&A section
st.markdown("### 🔎 Ask a question")
retriever = Retriever(VectorStore(index_dir=INDEX_DIR))
rag = RAGChain(settings=settings)

q_col, k_col, rerank_col = st.columns([6, 1, 2])
with q_col:
    question = st.text_input("Your question (e.g., 'Care este perioada de grație?')")
with k_col:
    top_k = st.number_input("Top-K", 1, 20, value=TOP_K)
with rerank_col:
    do_rerank = st.checkbox("Re-rank (lexical + semantic)", value=True)

if st.button("Answer"):
    if not retriever.vs.exists():
        st.error("Nu există niciun index. Asigură-te că sunt PDF-uri în `data/samples/` și apasă Rebuild.")
    elif not question.strip():
        st.wa
