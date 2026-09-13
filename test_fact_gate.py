"""Verifies the fact gate's ACTUAL current behavior — rewritten because
the previous version tested a superseded design (step() used to write
confirmed state and graph.json directly; it now only proposes candidates,
and confirmed/rejected state lives on the graph itself, not on FactGate).
Confirmed against fact_gate.py's real API, not the old one:
  1. step() proposes a sustained-high-trust edge as a CANDIDATE only —
     it does not confirm it and does not write graph.json.
  2. confirm(src,dst) is what actually commits, sets tau=1.0, and is
     the thing that writes graph.json.
  3. A never-driven edge is never even proposed (no false candidates).
  4. save_json is a full-state snapshot (one entry per edge, always) —
     not an append log, so "duplicate commits" isn't a coherent concept
     anymore; what's actually checked now is that confirming twice is a
     safe no-op and the file never grows a phantom second entry.
"""
import os
import json
import torch
from graph import ESGRGraph
from fact_gate import FactGate

TEST_JSON = "test_graph_output.json"


def test_step_only_proposes_confirm_actually_commits():
    if os.path.exists(TEST_JSON):
        os.remove(TEST_JSON)
    torch.manual_seed(0)
    g = ESGRGraph(n_nodes=512, mean_out_degree=10, active_fraction=0.1, seed=1)
    tracked_edge = 0
    g.w[tracked_edge] = 1.0  # force positive so it reliably drives its target active
    gate = FactGate(g, trust_threshold=0.8, min_consecutive_ticks=5, json_path=TEST_JSON)

    u, v = int(g.src[tracked_edge]), int(g.dst[tracked_edge])
    stimulus = torch.zeros(g.n)
    stimulus[u] = 3.0

    proposed_tick = None
    for t in range(200):
        g.tick(external_input=stimulus)
        proposed = gate.step()
        if tracked_edge in proposed and proposed_tick is None:
            proposed_tick = t

    assert proposed_tick is not None, "a consistently, strongly driven edge was never even proposed"
    assert not bool(g.confirmed[tracked_edge]), "step() must never confirm on its own — only propose"
    assert not os.path.exists(TEST_JSON), "step() must never write graph.json — only confirm()/reject() do"
    print(f"[PASS] proposed (not confirmed) at tick {proposed_tick}; "
          f"graph.confirmed={bool(g.confirmed[tracked_edge])}, no graph.json written yet")

    gate.confirm(u, v)
    assert bool(g.confirmed[tracked_edge]), "confirm() did not set graph.confirmed"
    assert g.tau[tracked_edge].item() == 1.0, "confirm() did not set tau=1.0"
    assert os.path.exists(TEST_JSON), "confirm() did not write graph.json"
    with open(TEST_JSON) as f:
        facts = json.load(f)
    matching = [e for e in facts["edges"] if e["src"] == u and e["dst"] == v]
    assert len(matching) == 1 and matching[0]["confirmed"] is True
    print(f"[PASS] confirm({u},{v}) actually committed: graph.confirmed=True, "
          f"tau=1.0, graph.json entry confirmed=True")


def test_never_driven_edge_is_never_proposed_or_confirmed():
    if os.path.exists(TEST_JSON):
        os.remove(TEST_JSON)
    torch.manual_seed(0)
    g = ESGRGraph(n_nodes=512, mean_out_degree=10, active_fraction=0.1, seed=2)
    gate = FactGate(g, trust_threshold=0.8, min_consecutive_ticks=5, json_path=TEST_JSON)

    stimulus = torch.zeros(g.n)
    stimulus[100:110] = 1.0
    ever_proposed = set()
    for t in range(200):
        g.tick(external_input=stimulus)
        ever_proposed.update(gate.step())

    never_touched_src = 0  # a node far outside the stimulated range
    untouched_edges = (g.src == never_touched_src).nonzero().flatten().tolist()
    for e in untouched_edges:
        assert e not in ever_proposed, f"edge {e} (never driven) was incorrectly proposed"
        assert not bool(g.confirmed[e]), f"edge {e} (never driven) was incorrectly confirmed"
    print(f"[PASS] {len(untouched_edges)} never-driven edge(s) never proposed and never confirmed "
          f"(no false positives, and nothing auto-confirms without an explicit confirm() call)")


def test_confirm_twice_is_a_safe_no_op():
    if os.path.exists(TEST_JSON):
        os.remove(TEST_JSON)
    torch.manual_seed(0)
    g = ESGRGraph(n_nodes=256, mean_out_degree=10, active_fraction=0.1, seed=3)
    gate = FactGate(g, trust_threshold=0.8, min_consecutive_ticks=5, json_path=TEST_JSON)
    u, v = int(g.src[0]), int(g.dst[0])

    gate.confirm(u, v)
    with open(TEST_JSON) as f:
        n_edges_after_first = len(json.load(f)["edges"])

    gate.confirm(u, v)  # confirm the SAME edge again
    with open(TEST_JSON) as f:
        data = json.load(f)
    n_edges_after_second = len(data["edges"])

    assert n_edges_after_first == n_edges_after_second == g.src.shape[0], (
        "save_json is a full-state snapshot — confirming twice must not grow or shrink the edge count"
    )
    matching = [e for e in data["edges"] if e["src"] == u and e["dst"] == v]
    assert len(matching) == 1, "the confirmed edge must still appear exactly once, not duplicated"
    print(f"[PASS] confirming the same edge twice is a safe no-op — "
          f"edge count stays {n_edges_after_second} (== total graph edges), no duplicate entry")


if __name__ == "__main__":
    test_step_only_proposes_confirm_actually_commits()
    test_never_driven_edge_is_never_proposed_or_confirmed()
    test_confirm_twice_is_a_safe_no_op()
    if os.path.exists(TEST_JSON):
        os.remove(TEST_JSON)
    print("\nAll fact gate checks passed (rewritten to match the current propose/confirm API).")
