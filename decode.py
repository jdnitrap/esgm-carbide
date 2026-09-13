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


def decode(graph, cap=16):
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

    candidates = []
    for u in active:
        if u in tile_node_ids:
            # Any active tiled node emits directly — this was the byte-
            # only filter Grok flagged (`if not (0<=u<=255)`); word
            # tiles (262+) were being skipped outright regardless of
            # tiles.json. Fixed: tile membership alone now qualifies,
            # not just being in the byte range.
            candidates.append((graph.x[u].item(), u))
            continue
        if not (0 <= u <= 255):
            continue  # untiled, out of byte range — nothing to emit
        result = graph.propose(u)
        if result["status"] == "HIT":
            i = result["edges"][0]
            score = graph.tau[i].item() * graph.w[i].item() * graph.x[u].item()
            candidates.append((score, u))

    if not candidates:
        return {"bytes": [], "status": "silence"}

    candidates.sort(key=lambda t: t[0], reverse=True)
    return {"bytes": [u for _, u in candidates[:cap]], "status": "emit"}
