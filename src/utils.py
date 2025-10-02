import hashlib
import contextlib
import time
from pathlib import Path
import re

CLEAN_SPACES = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = CLEAN_SPACES.sub(" ", s)
    return s.strip()

def ensure_dirs(*paths: Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self._t0 = time.time()
        self.elapsed = 0.0
        return self
    def __exit__(self, *exc):
        self.elapsed = time.time() - self._t0
