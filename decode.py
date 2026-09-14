"""Tiny decoder. READ-ONLY — never calls tick(), never mutates graph
state, never touches propose()'s or byte_identity's logic. No softmax
attention, no learned unembed, no dense hidden state persisted across
ticks: this function has no state of its own at all, it just reads
graph.x / graph.confirmed / graph.rejected / graph.suspended fresh
every call.

CATEGORY_LABELS is a frozen dict (not a trained layer) — only used for
printing, not for the emit decision itself.
"""
import json
from byte_identity import CATEGORY_NAMES, CATEGORY_OFFSET

CATEGORY_LABELS = {CATEGORY_OFFSET + i: name for i, name in enumerate(CATEGORY_NAMES)}


def decode(graph, cap=16, split_cap=False):
    # Read fresh every call, not cached at import — so a tile added
    # mid-session (the `tile` command) is visible immediately, per
    # "ask uses updated tiles.json immediately." Still read-only.
    with open("tiles.json") as f:
        tile_node_ids = set(json.load(f).values())

    active = (graph.x > 0).nonzero().flatten().tolist()

    # Priority check across ALL confirmed edges, active or not — "do
    # not talk over a fight." Activity does not have to prove the
    # fight; a suspended contradiction silences decode regardless of
    # whether its nodes happen to be firing this tick.
    fight = graph.confirmed & (~graph.rejected) & graph.suspended
    if bool(fight.any()):
        return {"bytes": [], "status": "suspend"}

    tiled_candidates = []
    byte_candidates = []
    for u in active:
        if u in tile_node_ids:
            # Any active tiled node emits directly — this was the byte-
            # only filter Grok flagged (`if not (0<=u<=255)`); word
            # tiles (262+) were being skipped outright regardless of
            # tiles.json. Fixed: tile membership alone now qualifies,
            # not just being in the byte range.
            tiled_candidates.append((graph.x[u].item(), u))
            continue
        if not (0 <= u <= 255):
            continue  # untiled, out of byte range — nothing to emit
        result = graph.propose(u)
        if result["status"] == "HIT":
            i = result["edges"][0]
            score = graph.tau[i].item() * graph.w[i].item() * graph.x[u].item()
            byte_candidates.append((score, u))

    if not tiled_candidates and not byte_candidates:
        return {"bytes": [], "status": "silence"}

    if not split_cap:
        # Original behavior, byte-for-byte unchanged: one shared
        # ranked list, top `cap` overall regardless of tiled/byte mix.
        candidates = tiled_candidates + byte_candidates
        candidates.sort(key=lambda t: t[0], reverse=True)
        return {"bytes": [u for _, u in candidates[:cap]], "status": "emit"}

    # split_cap=True: tiled (word-level) candidates get their own
    # reserved half of the cap, so byte-level competition -- however
    # intense -- can never fully crowd them out of the emitted list.
    # Same capacity-not-just-speed principle as graph.py's
    # split_sparsity_at, one layer up the pipeline. Found necessary by
    # testing (2026-09-14): even after fixing kWTA's shared budget, a
    # tiled node could still win its own kWTA seat and then get pushed
    # out of this function's OWN separate, shared top-16 ranking by
    # enough individually high-scoring bytes -- a second, independent
    # instance of the same "one shared budget" bug, one layer removed.
    cap_tiled = cap // 2
    cap_byte = cap - cap_tiled
    tiled_candidates.sort(key=lambda t: t[0], reverse=True)
    byte_candidates.sort(key=lambda t: t[0], reverse=True)
    kept = [u for _, u in tiled_candidates[:cap_tiled]] + [u for _, u in byte_candidates[:cap_byte]]
    return {"bytes": kept, "status": "emit"}
