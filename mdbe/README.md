# MDBE tables

Mechanically Defined Byte-Level Embedding (MDBE) tables and the
language-mechanics worksheets that define the human-interpretable
column constraints.

These were placed from the working attachments on 2026-09-13.
Spreadsheets are stored as CSV so they stay readable in git and load
from Python without an Excel dependency. Two drafts of the 256-byte
table are kept because they differ (control-byte labels and many cell
values).

## Files

- `DEFINITION.md` — formal definition of MDBE
- `ASCII_Linguistics.csv` — 256-row byte × constraint matrix (draft 1)
- `ASCII_Linguistics_v2.csv` — same matrix, later draft with named control bytes
- `language_mechanics_Mechanics_as_columns.csv`
- `language_mechanics_Your_sentence.csv`
- `language_mechanics_Column_key.csv`
- `language_mechanics_workbook_Input.csv`
- `language_mechanics_workbook_Lists.csv`
- `language_mechanics_workbook_Tokens.csv`

`byte_identity.py` in this repo already implements a sparse, frozen-wire
translation of the MDBE idea. These tables are the dense per-byte
constraint matrix that idea came from.
