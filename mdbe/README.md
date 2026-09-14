# MDBE tables

Mechanically Defined Byte-Level Embedding (MDBE) tables and the
language-mechanics worksheets that define the human-interpretable
column constraints.

Placed from the working attachments on 2026-09-13 into `jdnitrap/esgr`.

## Definition

See `DEFINITION.md`. Rows are raw byte values (0x00–0xFF). Columns are
hardcoded linguistic/grammatical constraints. Cells are floating-point
alignment scores.

## Files already decoded in this folder

- `DEFINITION.md`
- `language_mechanics_Column_key.csv`
- `language_mechanics_Your_sentence.csv`
- `language_mechanics_workbook_Lists.csv`
- `language_mechanics_workbook_Tokens.csv`

## Packed tables (decode locally)

The two 256-row ASCII matrices and two larger worksheets are stored
under `packed/` as gzip+base64 so they survive a text-only commit path.
Materialize the CSVs with:

```bash
python3 mdbe/unpack_tables.py
```

That writes:

- `ASCII_Linguistics.csv` — draft 1 of the 256-byte × constraint matrix
- `ASCII_Linguistics_v2.csv` — later draft (named control bytes)
- `language_mechanics_Mechanics_as_columns.csv`
- `language_mechanics_workbook_Input.csv`

Original attachment names:

- `Mechanically Defined Byte-Level Embedding (MDBE) table.md`
- `ASCII_Linguistics.xlsx` / `ASCII_Linguistics-1.xlsx`
- `language_mechanics_columns.xlsx` / `language_mechanics_columns-1.xlsx`

`byte_identity.py` in this repo already implements a sparse, frozen-wire
translation of the MDBE idea. These tables are the dense per-byte
constraint matrix that idea came from.
