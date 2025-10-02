# count_chunks_one.py — test minimal de chunking pentru 1 PDF (fără ingest.py)
from pathlib import Path
from pypdf import PdfReader

PDF_NAME = "IPID_PAD.pdf"  # schimbă cu oricare PDF din data/samples

BASE = Path.cwd()
PDF = BASE / "data" / "samples" / PDF_NAME

def simple_chunks(text, size=500, overlap=50):
    i = 0
    n = len(text)
    out = 0
    while i < n:
        j = min(n, i + size)
        chunk = text[i:j].strip()
        if chunk:
            out += 1
        i = j - overlap if j - overlap > 0 else j
    return out

def main():
    if not PDF.exists():
        print("Nu găsesc", PDF)
        return
    r = PdfReader(str(PDF))
    total_chars = 0
    total_chunks = 0
    print(f"{PDF.name}: {len(r.pages)} pagini")
    for idx, page in enumerate(r.pages, start=1):
        raw = page.extract_text() or ""
        total_chars += len(raw)
        c = simple_chunks(raw, size=500, overlap=50)
        total_chunks += c
        print(f"  pagina {idx}: chars={len(raw)}, chunks={c}")
    print(f"TOTAL: chars={total_chars}, chunks={total_chunks}")

if __name__ == "__main__":
    main()
