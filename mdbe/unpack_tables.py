"""Unpack packed/*.b64.gz.txt into CSVs in this directory."""
from pathlib import Path
import gzip, base64

here = Path(__file__).resolve().parent
pack = here / "packed"
mapping = {
    "ASCII_Linguistics.b64.gz.txt": "ASCII_Linguistics.csv",
    "ASCII_Linguistics_v2.b64.gz.txt": "ASCII_Linguistics_v2.csv",
    "language_mechanics_Mechanics_as_columns.b64.gz.txt": "language_mechanics_Mechanics_as_columns.csv",
    "language_mechanics_workbook_Input.b64.gz.txt": "language_mechanics_workbook_Input.csv",
}
for src_name, dest_name in mapping.items():
    blob = (pack / src_name).read_text().strip()
    dest = here / dest_name
    dest.write_bytes(gzip.decompress(base64.b64decode(blob)))
    print("wrote", dest, dest.stat().st_size)
