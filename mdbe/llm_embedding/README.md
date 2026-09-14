# Toy LLM embedding table

Teaching contrast with MDBE.

A normal LLM embedding matrix is **rows = token IDs, columns = unlabeled dims**.
Token ID looks up a whole row. `dim_0` is not "verb".

MDBE (see `../DEFINITION.md`) is the opposite on the column axis:
rows stay locked to raw bytes, but **columns are named linguistic constraints**.

Source workbooks (duplicates of the same toy table):
- `llm_embedding_sample.xlsx` — table + cosine + how-to-read
- `llm_embedding_sample-1.xlsx` — same + transposed view (this dump)
- `llm_embedding_sample-2.xlsx` — same + Graphs / PCA sheet

Ignored already-in-repo attachments: MDBE definition, ASCII_Linguistics, language_mechanics.
