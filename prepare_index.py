from pathlib import Path
from glob import glob
import gc
import time
import traceback

from src.utils import ensure_dirs
from src.vectorstore import VectorStore
from src.ingest import stream_pdf_chunks

BASE = Path.cwd()
DATA = BASE / "data"
SAMPLES = DATA / "samples"
INDEX = DATA / "index"


def build_for_one_pdf(vs: VectorStore, pdf_path: str, batch_size: int = 48) -> int:
    """
    Procesează un PDF incremental și întoarce numărul de chunk-uri indexate.
    Nu ține nimic mare în memorie.
    """
    count = 0
    batch = []
    for chunk in stream_pdf_chunks(pdf_path, default_meta={"doc_type": "Bundled"}):
        batch.append(chunk)
        if len(batch) >= batch_size:
            vs.add_batch(batch)
            count += len(batch)
            batch.clear()
    if batch:
        vs.add_batch(batch)
        count += len(batch)
        batch.clear()
    gc.collect()
    return count


def main():
    pdfs = sorted(glob(str(SAMPLES / "*.pdf")))
    if not pdfs:
        print("[prepare] Nu există PDF-uri în data/samples/.")
        return

    ensure_dirs(INDEX)
    vs = VectorStore(index_dir=INDEX)

    print(f"[prepare] Găsit {len(pdfs)} fișiere. Construiesc incremental…")
    total = 0
    ok_files = 0
    failed = []

    for i, pdf in enumerate(pdfs, start=1):
        name = Path(pdf).name
        print(f"\n[prepare] ({i}/{len(pdfs)}) -> {name}")
        t0 = time.time()
        try:
            n = build_for_one_pdf(vs, pdf, batch_size=48)
            dt = time.time() - t0
            print(f"[prepare]    ✓ Indexat {n} chunk-uri în {dt:.1f}s")
            total += n
            ok_files += 1
        except Exception as e:
            print(f"[prepare]    ✗ Eroare la {name}: {e}")
            traceback.print_exc(limit=1)
            failed.append(name)
            # continuă cu următorul fișier

    print("\n[prepare] — Sumar —")
    print(f"[prepare] Fișiere OK: {ok_files}/{len(pdfs)}")
    if failed:
        print(f"[prepare] Fișiere cu erori: {len(failed)} -> {', '.join(failed)}")
    print(f"[prepare] Total chunk-uri indexate: {total}")
    print("[prepare] Index scris în data/index/. Poți porni:  streamlit run app.py")


if __name__ == "__main__":
    main()
