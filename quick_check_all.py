from pathlib import Path
from glob import glob
from pypdf import PdfReader

BASE = Path.cwd()
SAMPLES = BASE / "data" / "samples"

pdfs = sorted(glob(str(SAMPLES / "*.pdf")))
print(f"Found {len(pdfs)} PDFs in data/samples")
for path in pdfs:
    p = Path(path)
    try:
        r = PdfReader(str(p))
        pages = len(r.pages)
    except Exception as e:
        print(f"  {p.name}: ERROR opening ({e})")
        continue
    chars_total = 0
    per_page = []
    for i, page in enumerate(r.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        n = len(raw)
        chars_total += n
        per_page.append(n)
    print(f"  {p.name}: pages={pages}, total_chars={chars_total}, per_page={per_page}")
