"""The generation ENGINE: produces an ORDERED sequence, one token per
step, instead of decode()'s simultaneous set. Entirely graph-native, no
new architecture — no attention, no softmax over a vocabulary, no
learned unembed, no dense hidden state:

  - A fixed GRAMMAR TEMPLATE (a sequence of expected roles) is given for
    free, the same move as MDBE's byte categories and word->role wiring
    — a hand-specified structural fact, not something learned.
  - At each step, the engine actively queries the STRUCTURAL word->role
    wiring (words_with_role) to find which words are already wired to
    the CURRENTLY needed role, and stimulates that pool directly —
    rather than passively hoping decode()'s currently-active set happens
    to contain one. This was a real bug found by testing: without this,
    the engine correctly matched the first word to its role, then got
    stuck reinjecting that SAME word forever, because nothing was
    driving activation toward the next role's candidates.
  - The single best-scoring candidate from that role's pool is
    committed to the output sequence and reinjected as context (along
    with continuing to drive the NEXT expected role) for the next step
    — true one-at-a-time autoregressive generation.

Deliberately tested against the SAME small tile set already built (13
words, 5 roles) — this is the engine, not the vocabulary. Whether it
produces GOOD English depends on a much larger, trained vocabulary
(explicitly out of scope here); this only has to prove the mechanism
can chain tokens in a real, ordered sequence using existing structure.
"""
import torch
from decode import decode
from word_structure import get_role, words_with_role

STIM_VALUE = 3.0

DEFAULT_GRAMMAR = ["ARTICLE", "NOUN", "VERB", "PREPOSITION", "ARTICLE", "NOUN"]

GRAMMARS = {
    "simple": ["ARTICLE", "NOUN", "VERB"],
    "with_adjective": ["ARTICLE", "ADJECTIVE", "NOUN", "VERB"],
    "prepositional": ["ARTICLE", "NOUN", "VERB", "PREPOSITION", "ARTICLE", "NOUN"],
    "pronoun_subject": ["PRONOUN", "VERB"],
    "pronoun_object": ["PRONOUN", "VERB", "ARTICLE", "ADJECTIVE", "NOUN"],
}


def _score(graph, node):
    return graph.x[node].item()


def generate_sequence(graph, prompt_nodes, grammar=None, max_len=12,
                       temperature=0.0, seed_ticks=4, step_ticks=2):
    """Returns (sequence, trace) — sequence is the ordered list of
    committed node ids; trace is per-step detail for verification.
    """
    grammar = grammar if grammar is not None else DEFAULT_GRAMMAR
    sequence = []
    trace = []
    slot = 0

    seed_stim = torch.zeros(graph.n)
    for node in prompt_nodes:
        seed_stim[node] = STIM_VALUE
    for _ in range(seed_ticks):
        graph.tick(external_input=seed_stim, temperature=temperature)

    prev_committed = prompt_nodes[-1] if prompt_nodes else None

    for step in range(max_len):
        expected_role = grammar[slot] if slot < len(grammar) else None
        role_pool = words_with_role(graph, expected_role) if expected_role else []

        # ENGINE: actively drive toward the expected role's real,
        # structurally-wired candidates, plus keep the previous word
        # active for context continuity.
        stim = torch.zeros(graph.n)
        if prev_committed is not None:
            stim[prev_committed] = STIM_VALUE
        for w in role_pool:
            stim[w] = STIM_VALUE
        for _ in range(step_ticks):
            graph.tick(external_input=stim, temperature=temperature)

        result = decode(graph)
        if result["status"] in ("silence", "suspend"):
            trace.append({"step": step, "status": result["status"], "picked": None,
                          "expected_role": expected_role})
            break

        candidates = result["bytes"]
        picked = None
        picked_reason = None
        if expected_role is not None:
            role_matches = [c for c in candidates if c in role_pool]
            if role_matches:
                picked = max(role_matches, key=lambda c: _score(graph, c))
                picked_reason = f"role_match:{expected_role}"

        if picked is None and candidates:
            picked = max(candidates, key=lambda c: _score(graph, c))
            picked_reason = "fallback_best_score"

        trace.append({
            "step": step, "status": result["status"], "candidates": candidates,
            "role_pool": role_pool, "expected_role": expected_role,
            "picked": picked, "reason": picked_reason,
        })

        if picked is None:
            break

        sequence.append(picked)
        prev_committed = picked
        if picked_reason == f"role_match:{expected_role}":
            slot += 1
        if slot >= len(grammar):
            break

    return sequence, trace
