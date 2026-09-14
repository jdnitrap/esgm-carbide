"""The human-gated retrain trigger: wraps warm-start checkpointing
(head_checkpoint.py) and real training (train_head_rnn.py's approach)
into one callable a person decides to fire -- from the shell's
`retrain_head` command, informed by real uncertainty signals already in
the graph (how many edges are currently proposed-but-unconfirmed).
Never runs on its own; nothing in this repo calls this automatically.
"""
import json
import os
import random
import torch
import torch.nn as nn
from graph import ESGRGraph
from head import NextByteRNN, build_tag_table, N_COLUMNS
from head_checkpoint import save_head_checkpoint, load_head_checkpoint

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
CHECKPOINT_PATH = "head_checkpoint.pt"
SAMPLE_BYTES = 300_000
SEQ_LEN = 32
BATCH_SIZE = 64


def uncertainty_signal(graph, gate):
    """Real, checkable count of 'things the graph has noticed but isn't
    sure about yet' -- edges currently proposed but not confirmed or
    rejected. A person uses this to decide WHEN retrain_head is worth
    running; nothing here decides that automatically."""
    return int(gate.proposed.sum().item())


def _load_data(sample_bytes=SAMPLE_BYTES):
    with open(CORPUS_PATH, "rb") as f:
        data = f.read(sample_bytes)
    starts = list(range(0, len(data) - SEQ_LEN - 1, SEQ_LEN))
    random.shuffle(starts)
    split = int(len(starts) * 0.8)
    return data, starts[:split], starts[split:]


def _xyt(data, starts, tag_table):
    x = torch.tensor([list(data[s:s + SEQ_LEN]) for s in starts], dtype=torch.long)
    y = torch.tensor([list(data[s + 1:s + SEQ_LEN + 1]) for s in starts], dtype=torch.long)
    t = torch.stack([tag_table[s:s + SEQ_LEN] for s in starts])
    return x, y, t


def _eval_acc(model, x, y, t):
    with torch.no_grad():
        logits, _ = model(x, t)
        return (logits.argmax(-1) == y).float().mean().item()


def retrain_head(graph, n_epochs=3, checkpoint_path=CHECKPOINT_PATH,
                  tiles_path="tiles.json", hubs_path="grammar_extra_hubs.json",
                  sample_bytes=SAMPLE_BYTES, verbose=False):
    """Loads the existing checkpoint (warm-starting if the graph's tag
    structure grew since it was saved) or starts fresh if none exists
    yet, trains n_epochs on a real corpus sample, saves the result.
    Returns a stats dict -- never touches graph.w/tau/confirmed/etc,
    the graph is read-only input here, same as everywhere else the head
    touches it."""
    tiles = json.load(open(tiles_path))
    hub_ids = json.load(open(hubs_path)) if os.path.exists(hubs_path) else {}

    data, train_starts, held_starts = _load_data(sample_bytes)
    tag_table = build_tag_table(graph, hub_ids, tiles, data)

    had_checkpoint = os.path.exists(checkpoint_path)
    if had_checkpoint:
        model, warm_started, old_dim, new_dim = load_head_checkpoint(checkpoint_path, new_tag_dim=N_COLUMNS)
        held_x, held_y, held_t = _xyt(data, held_starts, tag_table)
        before_acc = _eval_acc(model, held_x, held_y, held_t)
    else:
        model = NextByteRNN(emb_dim=16, hidden=64, use_embedding=True, use_tags=True)
        warm_started, old_dim, new_dim = False, None, N_COLUMNS
        held_x, held_y, held_t = _xyt(data, held_starts, tag_table)
        before_acc = _eval_acc(model, held_x, held_y, held_t)

    train_x, train_y, train_t = _xyt(data, train_starts, tag_table)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n = train_x.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            logits, _ = model(train_x[idx], train_t[idx])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), train_y[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if verbose:
            epoch_acc = _eval_acc(model, held_x, held_y, held_t)
            print(f"  epoch {epoch+1}/{n_epochs}: loss={total_loss/n_batches:.4f} held-out_acc={epoch_acc:.4f}", flush=True)

    after_acc = _eval_acc(model, held_x, held_y, held_t)
    save_head_checkpoint(model, checkpoint_path, emb_dim=16, hidden=64, tag_dim=N_COLUMNS)

    return {
        "had_checkpoint": had_checkpoint, "warm_started": warm_started,
        "old_dim": old_dim, "new_dim": new_dim,
        "before_acc": before_acc, "after_acc": after_acc, "epochs": n_epochs,
    }
