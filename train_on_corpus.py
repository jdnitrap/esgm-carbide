"""Massive-data training run: feeds the real 5MB Carbide text corpus
through the ESGR graph byte-by-byte, exercising the MDBE byte-level
foundation with real English text at scale for the first time. Same
underlying mechanism as everything else (tick() with external_input) —
no new architecture, just a large, real data source instead of hand-
written sentences.
"""
import time
import torch
from graph import ESGRGraph
from decode import decode

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
SAVE_EVERY = 200_000
REPORT_EVERY = 50_000

g = ESGRGraph.load_json("graph.json")
print(f"Loaded graph.json: n={g.n}, n_edges={g.src.shape[0]}, n_confirmed={int(g.confirmed.sum())}")

with open(CORPUS_PATH, "rb") as f:
    data = f.read()
print(f"Corpus: {len(data):,} bytes\n")

t0 = time.time()
nan_count = 0
for i, b in enumerate(data):
    stim = torch.zeros(g.n)
    stim[b] = 3.0
    info = g.tick(external_input=stim)
    if info["nan"]:
        nan_count += 1

    if (i + 1) % REPORT_EVERY == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(data) - (i + 1)) / rate
        print(f"  {i+1:>9,}/{len(data):,} bytes  "
              f"({rate:.0f} b/s, eta {eta/60:.1f} min)  nan_count={nan_count}  "
              f"active_frac={info['active_fraction']:.4f}", flush=True)

    if (i + 1) % SAVE_EVERY == 0:
        g.save_json("graph.json")

g.save_json("graph.json")
elapsed = time.time() - t0
print(f"\nDone: {len(data):,} bytes in {elapsed/60:.1f} min ({len(data)/elapsed:.0f} b/s)")
print(f"Total nan_count: {nan_count}")
print(f"Final n_confirmed: {int(g.confirmed.sum())}  n_suspended: {int(g.suspended.sum())}")
