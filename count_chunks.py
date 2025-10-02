# count_chunks.py — verifică numărul de chunk-uri generate per PDF
from pathlib import Path
from glob import glob
from src.ingest import stream_pdf_chunks

BASE = Path.cwd()
SAMPLES = BASE / "data" / "samples"

pdfs = sorted(glob(str(SAMPLES / "*.pdf")))
print(f"Found {len(pdfs)} PDFs")
for path in pdfs:
    chunks = list(stream_pdf_chunks(path, default_meta={"doc_type": "Bundled"}))
    print(f"{Path(path).name}: chunks={len(chunks)}")
    if chunks:
        print("  first chunk preview:", chunks[0]["text"][:120].replace("\n"," "))
