"""Integration stress test — exercises the WHOLE system together against
the real, accumulated graph.json, not isolated units. Checks:
  1. Long-run stability under mixed stimuli (bytes, MDBE-linked bytes,
     word tiles, all mixed) — no NaN, active_frac stays in band.
  2. Frozen structure survives heavy unrelated activity: MDBE
     byte->category AND the new letter->word / word->role edges must
     still be exactly correct after 1000+ ticks of unrelated stimulus.
  3. Full pipeline in one scenario: ask -> propose -> confirm a NEW
     fact -> manufacture a contradiction -> suspend triggers -> decode
     respects it -> save -> fresh-process reload -> everything matches.
  4. Timing: does it stay fast at scale, not just in short smoke tests.
"""
import time
import torch
from graph import ESGRGraph
from fact_gate import FactGate
from byte_identity import CATEGORY_OFFSET
from word_structure import ROLE_OFFSET, ROLE_NAMES
from decode import decode

g = ESGRGraph.load_json("graph.json")
gate = FactGate(g, json_path="stress_test_graph.json")
print(f"Loaded real graph.json: n={g.n}, n_edges={g.src.shape[0]}, "
      f"n_confirmed={int(g.confirmed.sum())}, n_suspended={int(g.suspended.sum())}\n")

# ---- 0. Snapshot frozen structure BEFORE stress, to compare after ----
frozen_idx = g.frozen.nonzero().flatten().tolist()
frozen_w_before = g.w[frozen_idx].clone()
frozen_tau_before = g.tau[frozen_idx].clone()
print(f"=== 0. Snapshotting {len(frozen_idx)} frozen edges before stress ===\n")

# ---- 1 & 2. Long-run stability under heavy MIXED stimulus ----
print("=== 1&2. Long-run stability + frozen-structure survival (1500 mixed ticks) ===")
torch.manual_seed(0)
t0 = time.time()
nan_ticks = 0
active_fracs = []
for t in range(1500):
    stim = torch.zeros(g.n)
    # Rotate through varied stimulus patterns: raw bytes, MDBE-linked
    # bytes, category nodes, word tiles, role hubs — hit everything.
    pattern = t % 5
    if pattern == 0:
        stim[torch.randint(0, 256, (8,))] = 2.0
    elif pattern == 1:
        stim[ord('c')] = 3.0
        stim[ord('a')] = 3.0
        stim[ord('t')] = 3.0
    elif pattern == 2:
        stim[CATEGORY_OFFSET:CATEGORY_OFFSET + 6] = 1.5
    elif pattern == 3:
        stim[262:275] = 1.5  # all word tiles
    else:
        stim[ROLE_OFFSET:ROLE_OFFSET + len(ROLE_NAMES)] = 1.5
    info = g.tick(external_input=stim)
    if info["nan"]:
        nan_ticks += 1
    active_fracs.append(info["active_fraction"])
elapsed = time.time() - t0

print(f"  1500 mixed-stimulus ticks in {elapsed:.2f}s ({1500/elapsed:.0f} ticks/sec)")
print(f"  nan_ticks={nan_ticks} (must be 0)")
print(f"  active_fraction: min={min(active_fracs):.4f} max={max(active_fracs):.4f} "
      f"(target band 0.046-0.051)")

frozen_w_after = g.w[frozen_idx]
frozen_tau_after = g.tau[frozen_idx]
moved = int(((frozen_w_after != frozen_w_before) | (frozen_tau_after != frozen_tau_before)).sum().item())
print(f"  frozen edges moved after 1500 ticks of heavy unrelated activity: {moved} (must be 0)")

stress1_pass = (nan_ticks == 0 and moved == 0 and all(0.045 <= f <= 0.052 for f in active_fracs))
print(f"  VERDICT: {'PASS' if stress1_pass else 'FAIL'}\n")

# ---- 3. Full pipeline in one scenario ----
print("=== 3. Full pipeline: ask -> confirm -> contradiction -> suspend -> decode -> save/reload ===")

# Pick two DIFFERENT, currently-unconfirmed edges from the same source
# node to manufacture a fresh contradiction (avoid MDBE/frozen edges).
src_for_test = None
for node in range(280, g.n):
    edges = (g.src == node).nonzero().flatten().tolist()
    unconfirmed = [e for e in edges if not bool(g.confirmed[e]) and not bool(g.frozen[e])]
    if len(unconfirmed) >= 2:
        src_for_test = node
        e_A, e_B = unconfirmed[0], unconfirmed[1]
        break
assert src_for_test is not None, "could not find a clean test node with 2 unconfirmed edges"
dst_A, dst_B = int(g.dst[e_A]), int(g.dst[e_B])
print(f"  test contradiction pair: node {src_for_test} -> {dst_A} (A) vs -> {dst_B} (B)")

g.register_contradiction(e_A, e_B)
gate.confirm(src_for_test, dst_A)
suspended_at = None
for t in range(30):
    g.tick(external_input=None)
    if t == 3:
        gate.confirm(src_for_test, dst_B)
    if bool(g.suspended[e_A]) and suspended_at is None:
        suspended_at = t

decode_result = decode(g)
print(f"  suspended at tick {suspended_at}")
print(f"  decode() during active suspend: {decode_result}")
pipeline_pass = suspended_at is not None and decode_result["status"] == "suspend"
print(f"  VERDICT: {'PASS' if pipeline_pass else 'FAIL'}\n")

# ---- Save, then verify in a genuinely fresh reload ----
g.save_json("stress_test_graph.json")
g2 = ESGRGraph.load_json("stress_test_graph.json")
reload_ok = (
    torch.equal(g.confirmed, g2.confirmed) and torch.equal(g.rejected, g2.rejected)
    and torch.equal(g.suspended, g2.suspended) and torch.allclose(g.tau, g2.tau)
    and torch.allclose(g.w, g2.w) and torch.equal(g.frozen, g2.frozen)
    and g.contradiction_pairs == g2.contradiction_pairs
)
frozen_after_reload_ok = (
    torch.equal(g2.w[frozen_idx], frozen_w_after) and torch.equal(g2.tau[frozen_idx], frozen_tau_after)
)
print(f"=== Reload check ===")
print(f"  full state reload match: {reload_ok}")
print(f"  frozen structure survives reload intact: {frozen_after_reload_ok}")
print(f"  VERDICT: {'PASS' if reload_ok and frozen_after_reload_ok else 'FAIL'}\n")

overall = stress1_pass and pipeline_pass and reload_ok and frozen_after_reload_ok
print(f"=== OVERALL: {'PASS' if overall else 'FAIL'} ===")
