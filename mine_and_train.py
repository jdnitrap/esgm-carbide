"""Real word-level training (task 3): mines the actual 5MB corpus for
genuine occurrences of our vocabulary in real context, instead of hand-
written sentences. The earlier byte-level pass over this same corpus
never engaged word tiles at all (confirmed: mat->on decayed to zero) —
this pass fixes that by feeding REAL adjacent-word runs as simultaneous
pokes, the same mechanic as `ask`, just driven by real text instead of
a person typing.
"""
import re
import json
import time
import torch
from graph import ESGRGraph

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
STIM_VALUE = 3.0
STEP_TICKS = 4
MAX_GAP = 1  # allow up to this many non-vocab tokens between vocab words in a "run"

with open("tiles.json") as f:
    TILES = json.load(f)
LOWER_TILES = {k.lower(): v for k, v in TILES.items() if k not in ("A", "space", "newline")}

with open(CORPUS_PATH, encoding="utf-8", errors="ignore") as f:
    text = f.read()

tokens = re.findall(r"[A-Za-z']+", text.lower())
print(f"Corpus: {len(text):,} chars, {len(tokens):,} word tokens")

runs = []
current_run = []
gap = 0
for tok in tokens:
    if tok in LOWER_TILES:
        current_run.append(LOWER_TILES[tok])
        gap = 0
    else:
        gap += 1
        if gap > MAX_GAP and current_run:
            if len(current_run) >= 2:
                runs.append(current_run)
            current_run = []
if len(current_run) >= 2:
    runs.append(current_run)

print(f"Found {len(runs):,} real runs of >=2 adjacent-in-context vocabulary words")

from collections import Counter
word_counts = Counter()
for r in runs:
    for n in r:
        word_counts[n] += 1
rev = {v: k for k, v in LOWER_TILES.items()}
print("Top 15 most-seen vocabulary words in real runs:")
for node, count in word_counts.most_common(15):
    print(f"  {rev[node]:10s} {count}")

if not runs:
    print("\nNo real runs found — nothing to train on. Stopping.")
else:
    g = ESGRGraph.load_json("graph.json")
    t0 = time.time()
    for i, run in enumerate(runs):
        stim = torch.zeros(g.n)
        for node in run:
            stim[node] = STIM_VALUE
        for _ in range(STEP_TICKS):
            g.tick(external_input=stim)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{len(runs):,} runs trained ({time.time()-t0:.1f}s)")
    g.save_json("graph.json")
    print(f"\nDone: {len(runs):,} real runs trained in {time.time()-t0:.1f}s, saved.")
