"""Trains NextByteRNN on real corpus byte sequences and checks it
against TWO baselines, not one: the weak order-1 (single previous byte)
table NextByteHead already failed to beat, and a harder order-2
(previous 2 bytes) table -- a real sequence model should beat order-1
easily just by having more context; beating order-2 is the actual test
of whether the recurrence is doing something a slightly-richer count
table couldn't already do. Never touches the graph's own weights/edges
-- verified explicitly, not just by code inspection.
"""
import json
import random
import time
import torch
import torch.nn as nn
from collections import Counter
from graph import ESGRGraph
from head import NextByteRNN, build_tag_table

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
SAMPLE_BYTES = 300_000
SEQ_LEN = 32
EPOCHS = 5
BATCH_SIZE = 64


def load_sequences(n_bytes=SAMPLE_BYTES, seq_len=SEQ_LEN):
    with open(CORPUS_PATH, "rb") as f:
        data = f.read(n_bytes)
    starts = list(range(0, len(data) - seq_len - 1, seq_len))
    return data, starts


def chunks(data, starts, seq_len=SEQ_LEN):
    return [data[s:s + seq_len + 1] for s in starts]


def order_n_baseline_accuracy(train_seqs, held_seqs, order):
    counts = {}
    for seq in train_seqs:
        for i in range(order, len(seq)):
            ctx = bytes(seq[i - order:i])
            counts.setdefault(ctx, Counter())[seq[i]] += 1
    best = {ctx: c.most_common(1)[0][0] for ctx, c in counts.items()}
    hits = total = 0
    for seq in held_seqs:
        for i in range(order, len(seq)):
            ctx = bytes(seq[i - order:i])
            total += 1
            if best.get(ctx) == seq[i]:
                hits += 1
    return hits / total


def to_tensors(seqs):
    x = torch.tensor([list(s[:-1]) for s in seqs], dtype=torch.long)
    y = torch.tensor([list(s[1:]) for s in seqs], dtype=torch.long)
    return x, y


def train_variant(use_embedding, use_tags, train_x, train_y, train_tags,
                   held_x, held_y, held_tags):
    model = NextByteRNN(use_embedding=use_embedding, use_tags=use_tags)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    n = train_x.shape[0]
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            bx, by, bt = train_x[idx], train_y[idx], train_tags[idx]
            logits, _ = model(bx, bt)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), by.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        logits, _ = model(held_x, held_tags)
        preds = logits.argmax(dim=-1)
        acc = (preds == held_y).float().mean().item()
    return acc


def main():
    random.seed(0)
    torch.manual_seed(0)

    g = ESGRGraph.load_json("graph.json")
    snap = {"w": g.w.clone(), "tau": g.tau.clone(), "confirmed": g.confirmed.clone(),
            "modulation": g.modulation.clone(), "n": g.n, "n_edges": g.src.shape[0]}
    tiles = json.load(open("tiles.json"))
    hub_ids = json.load(open("grammar_extra_hubs.json"))

    data, starts = load_sequences()
    t0 = time.time()
    full_tags = build_tag_table(g, hub_ids, tiles, data)  # (len(data), 32), computed ONCE
    print(f"built full tag table for {len(data):,} bytes in {time.time()-t0:.1f}s "
          f"(includes both byte-level and word-mechanics columns)")

    random.shuffle(starts)
    split = int(len(starts) * 0.8)
    train_starts, held_starts = starts[:split], starts[split:]
    print(f"real sequences: {len(starts):,} of length {SEQ_LEN} "
          f"(train {len(train_starts):,} / held-out {len(held_starts):,})")

    train_seqs = chunks(data, train_starts)
    held_seqs = chunks(data, held_starts)
    base1 = order_n_baseline_accuracy(train_seqs, held_seqs, order=1)
    base2 = order_n_baseline_accuracy(train_seqs, held_seqs, order=2)
    print(f"order-1 (single previous byte) baseline: {base1:.4f}")
    print(f"order-2 (previous 2 bytes) baseline:      {base2:.4f}")

    train_x, train_y = to_tensors(train_seqs)
    held_x, held_y = to_tensors(held_seqs)
    train_tags = torch.stack([full_tags[s:s + SEQ_LEN] for s in train_starts])
    held_tags = torch.stack([full_tags[s:s + SEQ_LEN] for s in held_starts])

    for name, use_emb, use_tags in [
        ("full (embedding + fixed+mechanics columns)", True, True),
        ("embedding-only", True, False),
        ("tags-only (no learned embedding)", False, True),
    ]:
        acc = train_variant(use_emb, use_tags, train_x, train_y, train_tags, held_x, held_y, held_tags)
        beat1 = "beats order-1" if acc > base1 else "does not beat order-1"
        beat2 = "beats order-2" if acc > base2 else "does not beat order-2"
        print(f"{name:44s} held-out accuracy: {acc:.4f}  ({beat1}, {beat2})")

    unchanged = (
        torch.equal(g.w, snap["w"]) and torch.equal(g.tau, snap["tau"])
        and torch.equal(g.confirmed, snap["confirmed"]) and torch.equal(g.modulation, snap["modulation"])
        and g.n == snap["n"] and g.src.shape[0] == snap["n_edges"]
    )
    print(f"\ngraph completely unchanged by head training: {unchanged} (must be True)")


if __name__ == "__main__":
    main()
