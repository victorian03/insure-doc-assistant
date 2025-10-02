from pathlib import Path
from src.utils import ensure_dirs
from src.vectorstore import VectorStore
from src.ingest import stream_pdf_chunks

# alege un fișier mic și sigur (ex. demo-ul creat la noi)
PDF_NAME = "insure_demo_policy_ro.pdf"   # schimbă cu ce vrei să testezi

BASE = Path.cwd()
DATA = BASE / "data"
SAMPLES = DATA / "samples"
INDEX = DATA / "index"

def main():
    pdf_path = SAMPLES / PDF_NAME
    if not pdf_path.exists():
        print(f"[one] Nu găsesc {pdf_path}. Pune fișierul în data/samples/")
        return

    ensure_dirs(INDEX)
    vs = VectorStore(index_dir=INDEX)

    print(f"[one] Procesez: {pdf_path.name}")
    count = 0
    batch, BATCH_SIZE = [], 48

    for chunk in stream_pdf_chunks(str(pdf_path), default_meta={"doc_type": "Bundled"}):
        batch.append(chunk)
        if len(batch) >= BATCH_SIZE:
            vs.add_batch(batch)
            count += len(batch)
            batch.clear()

    if batch:
        vs.add_batch(batch)
        count += len(batch)
        batch.clear()

    print(f"[one] OK, indexate {count} bucăți (chunks). Index în data/index/")

if __name__ == "__main__":
    main()
