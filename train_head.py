"""Trains NextByteHead on real corpus bytes and replays Carbide's own
3-seed-ablation methodology (full vs embedding-only vs tags-only) on
ESGR's real data, rather than assuming the Carbide result transfers.
Never touches the graph's own weights/edges -- verified explicitly
below, not just true by code inspection.
"""
import random
import torch
import torch.nn as nn
from collections import Counter
from graph import ESGRGraph
from head import NextByteHead, byte_columns_batch, N_BYTES

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
SAMPLE_BYTES = 300_000  # real data, subsampled for a fast honest run -- not the full 5MB
EPOCHS = 5
BATCH_SIZE = 512


def load_byte_pairs(n_bytes=SAMPLE_BYTES):
    with open(CORPUS_PATH, "rb") as f:
        data = f.read(n_bytes)
    return [(data[i], data[i + 1]) for i in range(len(data) - 1)]


def trivial_baseline_accuracy(train_pairs, held_out):
    counts = {}
    for a, b in train_pairs:
        counts.setdefault(a, Counter())[b] += 1
    best = {a: c.most_common(1)[0][0] for a, c in counts.items()}
    hits = sum(1 for a, b in held_out if best.get(a) == b)
    return hits / len(held_out)


def train_variant(name, use_embedding, use_tags, train_pairs, held_out, tag_table):
    model = NextByteHead(use_embedding=use_embedding, use_tags=use_tags)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_x = torch.tensor([a for a, b in train_pairs], dtype=torch.long)
    train_y = torch.tensor([b for a, b in train_pairs], dtype=torch.long)
    held_x = torch.tensor([a for a, b in held_out], dtype=torch.long)
    held_y = torch.tensor([b for a, b in held_out], dtype=torch.long)

    n = train_x.shape[0]
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            bx, by = train_x[idx], train_y[idx]
            tags = tag_table[bx]
            logits = model(bx, tags)
            loss = loss_fn(logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

    with torch.no_grad():
        held_tags = tag_table[held_x]
        preds = model(held_x, held_tags).argmax(dim=-1)
        acc = (preds == held_y).float().mean().item()
    return acc


def main():
    random.seed(0)
    torch.manual_seed(0)

    g = ESGRGraph.load_json("graph.json")

    # Prove this training run never touches the graph itself, not just
    # by code inspection -- snapshot every real tensor and diff after.
    snap = {
        "w": g.w.clone(), "tau": g.tau.clone(), "confirmed": g.confirmed.clone(),
        "modulation": g.modulation.clone(), "n": g.n, "n_edges": g.src.shape[0],
    }

    tag_table = byte_columns_batch(g)

    pairs = load_byte_pairs()
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train_pairs, held_out = pairs[:split], pairs[split:]
    print(f"real byte pairs: {len(pairs):,} (train {len(train_pairs):,} / held-out {len(held_out):,})")

    baseline = trivial_baseline_accuracy(train_pairs, held_out)
    print(f"trivial per-byte-majority baseline: {baseline:.4f}")

    for name, use_emb, use_tags in [
        ("full (embedding + fixed columns)", True, True),
        ("embedding-only", True, False),
        ("tags-only (fixed columns, no learned embedding)", False, True),
    ]:
        acc = train_variant(name, use_emb, use_tags, train_pairs, held_out, tag_table)
        print(f"{name:55s} held-out accuracy: {acc:.4f}")

    # Verify: graph completely untouched by any of the above.
    unchanged = (
        torch.equal(g.w, snap["w"]) and torch.equal(g.tau, snap["tau"])
        and torch.equal(g.confirmed, snap["confirmed"]) and torch.equal(g.modulation, snap["modulation"])
        and g.n == snap["n"] and g.src.shape[0] == snap["n_edges"]
    )
    print(f"\ngraph completely unchanged by head training: {unchanged} (must be True)")


if __name__ == "__main__":
    main()
