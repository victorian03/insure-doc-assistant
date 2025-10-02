# src/ingest.py
from pathlib import Path
from typing import Dict, Iterator
import re
from pypdf import PdfReader
from .utils import clean_text

# Heuristici de control
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MIN_CHARS_PER_PAGE = 60     # sub asta considerăm pagina “fără text” (ex: doar logo)
MIN_CHARS_PER_CHUNK = 40    # sară peste fragmente prea scurte (ruperi, headere)

HEADER_FOOTER_PATTERNS = [
    r"^pagina\s+\d+(\s*/\s*\d+)?$",           # “Pagina 2 / 10”
    r"^page\s+\d+(\s+of\s+\d+)?$",            # “Page 2 of 10”
    r"^\d+\s*/\s*\d+$",                       # “2/10”
    r"^confidențial.*$",                      # linii standard de confidențialitate
    r"^copyright.*$",                         # copyright
    r"^\s*all rights reserved.*$",            # rights
]

# dacă știi exact denumirea companiei/logoului – pune aici ca să fie eliminată din headere
COMPANY_HINTS = [
    r"^allianz.*$", r"^groupama.*$", r"^generali.*$", r"^uniqa.*$", r"^nn.*$",
]

HEADER_FOOTER_REGEX = re.compile(
    "|".join(HEADER_FOOTER_PATTERNS + COMPANY_HINTS),
    flags=re.IGNORECASE
)

def _drop_headers_and_footers(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        # linii 100% uppercase, foarte scurte (logo/brand) – ignoră
        if ln.isupper() and len(ln) <= 30:
            continue
        # potrivire pe pattern-urile de header/footer
        if HEADER_FOOTER_REGEX.match(ln):
            continue
        kept.append(ln)
    cleaned = "\n".join(kept)
    return clean_text(cleaned)

def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not text:
        return
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHARS_PER_CHUNK:
            yield chunk
        start = end - overlap if end - overlap > 0 else end

def stream_pdf_chunks(pdf_path: str, default_meta: Dict[str, str] = None) -> Iterator[Dict]:
    """
    Generator: citește PDF-ul pagină cu pagină, păstrează doar textul (pozele sunt ignorate),
    curăță headere/footere/sigle și emite chunk-uri utile.
    """
    p = Path(pdf_path)
    reader = PdfReader(str(p))
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""

        # curățare & filtre de utilitate
        txt = _drop_headers_and_footers(raw)
        if len(txt) < MIN_CHARS_PER_PAGE:
            # probabil pagină cu logo, titlu mare sau scan fără OCR – sare peste
            continue

        for chunk in _split_into_chunks(txt):
            meta = {"source_path": str(p), "source_name": p.name, "page": i}
            if default_meta:
                meta.update(default_meta)
            yield {"text": chunk, "metadata": meta}
