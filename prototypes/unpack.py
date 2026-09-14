"""Unpack packed prototype sources in this directory."""
from pathlib import Path
import gzip, base64

here = Path(__file__).resolve().parent
blob = (here / "graph_prototype.py.b64.gz.txt").read_text().strip()
(here / "graph_prototype.py").write_bytes(gzip.decompress(base64.b64decode(blob)))
print("wrote", here / "graph_prototype.py")
