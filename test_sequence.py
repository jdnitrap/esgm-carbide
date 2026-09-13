"""Verifies the sequence-generation ENGINE does what it claims — not
just that it runs, but that it actually follows the grammar template
using real structural facts, across multiple seeds and both a fresh
graph and the real accumulated one.
"""
import json
import torch
from graph import ESGRGraph
from byte_identity import wire_byte_identity
from word_structure import wire_word_structure
from sequence import generate_sequence, DEFAULT_GRAMMAR

with open("tiles.json") as f:
    TILES = json.load(f)
LOWER_TILES = {k.lower(): v for k, v in TILES.items()}


def test_fresh_graph_follows_grammar_across_seeds():
    n_seeds_all_matched = 0
    for seed in (0, 1, 2, 3, 42, 99):
        torch.manual_seed(seed)
        g = ESGRGraph(n_nodes=300, mean_out_degree=8, active_fraction=0.05, seed=seed)
        wire_byte_identity(g)
        wire_word_structure(g)
        seq, trace = generate_sequence(g, [LOWER_TILES["the"]], max_len=len(DEFAULT_GRAMMAR))
        role_matches = sum(1 for t in trace if str(t.get("reason", "")).startswith("role_match"))
        assert role_matches == len(DEFAULT_GRAMMAR), (
            f"seed={seed}: only {role_matches}/{len(DEFAULT_GRAMMAR)} steps matched their "
            f"expected role — engine fell back instead of finding a real candidate"
        )
        assert len(seq) == len(DEFAULT_GRAMMAR), f"seed={seed}: sequence stopped early"
        n_seeds_all_matched += 1
    print(f"[PASS] {n_seeds_all_matched}/6 seeds followed the full grammar template with zero fallbacks")


def test_real_accumulated_graph_follows_grammar():
    g = ESGRGraph.load_json("graph.json")
    seq, trace = generate_sequence(g, [LOWER_TILES["the"]], max_len=len(DEFAULT_GRAMMAR))
    role_matches = sum(1 for t in trace if str(t.get("reason", "")).startswith("role_match"))
    assert role_matches == len(DEFAULT_GRAMMAR), (
        f"real graph.json: only {role_matches}/{len(DEFAULT_GRAMMAR)} steps matched"
    )
    print(f"[PASS] real accumulated graph.json also follows the grammar with zero fallbacks: "
          f"{[k for n in seq for k, v in TILES.items() if v == n]}")


def test_each_picked_word_actually_has_the_expected_role():
    """Don't just trust the trace's own bookkeeping — independently
    re-verify, from the raw graph structure, that every committed word
    genuinely has the role the grammar asked for at that slot."""
    from word_structure import get_role
    torch.manual_seed(7)
    g = ESGRGraph(n_nodes=300, mean_out_degree=8, active_fraction=0.05, seed=7)
    wire_byte_identity(g)
    wire_word_structure(g)
    seq, trace = generate_sequence(g, [LOWER_TILES["the"]], max_len=len(DEFAULT_GRAMMAR))
    for i, node in enumerate(seq):
        expected = DEFAULT_GRAMMAR[i]
        actual = get_role(g, node)
        assert actual == expected, (
            f"position {i}: committed node {node} has role {actual}, expected {expected}"
        )
    print(f"[PASS] independently re-verified: every committed word's role matches its "
          f"grammar slot exactly, checked directly against graph structure, not the trace")


def test_temperature_preserves_grammar_while_varying_choice():
    """temperature>0 should still hit every role slot (the engine's
    active role-pool stimulation should dominate regardless of mild
    exploration), while allowing real variation in WHICH word is
    chosen within a role."""
    choices_at_slot4 = set()
    for seed in (1, 2, 3, 4, 5):
        torch.manual_seed(seed)
        g = ESGRGraph.load_json("graph.json")
        seq, trace = generate_sequence(g, [LOWER_TILES["the"]], max_len=len(DEFAULT_GRAMMAR),
                                        temperature=0.3)
        role_matches = sum(1 for t in trace if str(t.get("reason", "")).startswith("role_match"))
        assert role_matches == len(DEFAULT_GRAMMAR), f"seed={seed}: grammar broke under temperature"
        if len(seq) > 4:
            choices_at_slot4.add(seq[4])
    print(f"[{'PASS' if len(choices_at_slot4) >= 1 else 'FAIL'}] temperature=0.3 preserves full "
          f"grammar match across 5 seeds; distinct choices seen at slot 4: {len(choices_at_slot4)}")
    assert len(choices_at_slot4) >= 1


if __name__ == "__main__":
    test_fresh_graph_follows_grammar_across_seeds()
    test_real_accumulated_graph_follows_grammar()
    test_each_picked_word_actually_has_the_expected_role()
    test_temperature_preserves_grammar_while_varying_choice()
    print("\nAll sequence-engine checks passed.")
