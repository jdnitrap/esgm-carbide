"""One real, full-scale training run: warm-starts from the existing
checkpoint, trains on the full 5MB corpus, then teaches the graph back
with the freshly trained model. Prints progress as it goes.
"""
import json
import time
import torch
from graph import ESGRGraph
from retrain_head import retrain_head, CORPUS_PATH, SEQ_LEN
from head import build_tag_table
from supervise import supervised_step

FULL_CORPUS_BYTES = 5_000_000
EPOCHS = 25  # heavy run, at explicit user direction -- was 4

print(f"loading graph...")
g = ESGRGraph.load_json("graph.json")
frozen_idx = g.frozen.nonzero().flatten().tolist()
w_frozen_before = g.w[frozen_idx].clone()
tau_frozen_before = g.tau[frozen_idx].clone()
n_before = g.n

t0 = time.time()
print(f"training on {FULL_CORPUS_BYTES:,} real corpus bytes, {EPOCHS} epochs, warm-started if a checkpoint exists...")
stats = retrain_head(g, n_epochs=EPOCHS, sample_bytes=FULL_CORPUS_BYTES, verbose=True)
elapsed = time.time() - t0

print(f"\n=== training done in {elapsed:.1f}s ===")
print(f"had_checkpoint={stats['had_checkpoint']} warm_started={stats['warm_started']} "
      f"({stats['old_dim']} -> {stats['new_dim']} cols)")
print(f"held-out accuracy before: {stats['before_acc']:.4f}")
print(f"held-out accuracy after {stats['epochs']} epochs: {stats['after_acc']:.4f}")

# teach-back pass with the freshly trained model, on a real held-out slice
print("\n=== teach-back pass ===")
from head import NextByteRNN
from head_checkpoint import load_head_checkpoint
model, _, _, _ = load_head_checkpoint("head_checkpoint.pt")

tiles = json.load(open("tiles.json"))
from grammar_extra import load_hub_ids
hub_ids = load_hub_ids()
with open(CORPUS_PATH, "rb") as f:
    teach_data = f.read()[FULL_CORPUS_BYTES - 500_000:FULL_CORPUS_BYTES]  # a real, held-out-ish tail slice
teach_tags = build_tag_table(g, hub_ids, tiles, teach_data)

x = torch.tensor([list(teach_data[i:i+SEQ_LEN]) for i in range(0, len(teach_data)-SEQ_LEN-1, SEQ_LEN)], dtype=torch.long)
y = torch.tensor([list(teach_data[i+1:i+SEQ_LEN+1]) for i in range(0, len(teach_data)-SEQ_LEN-1, SEQ_LEN)], dtype=torch.long)
t = torch.stack([teach_tags[i:i+SEQ_LEN] for i in range(0, len(teach_data)-SEQ_LEN-1, SEQ_LEN)])

with torch.no_grad():
    logits, _ = model(x, t)
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)

CONF_THRESHOLD = 0.5
candidate_pool = set(range(256))
taught = 0
for b in range(x.shape[0]):
    for i in range(x.shape[1]):
        if conf[b, i].item() < CONF_THRESHOLD:
            continue
        if int(pred[b, i]) != int(y[b, i]):
            continue
        supervised_step(g, int(x[b, i]), int(y[b, i]), candidate_pool)
        taught += 1

print(f"confident-and-correct predictions taught back to the graph: {taught}")

w_frozen_after = g.w[frozen_idx]
tau_frozen_after = g.tau[frozen_idx]
moved = int(((w_frozen_after != w_frozen_before) | (tau_frozen_after != tau_frozen_before)).sum().item())
print(f"frozen edges disturbed: {moved} (must be 0)")
print(f"n unchanged: {g.n == n_before} (must be True)")

g.save_json("graph.json")
print(f"\nsaved graph.json: n={g.n}, edges={g.src.shape[0]}, confirmed={int(g.confirmed.sum())}")
