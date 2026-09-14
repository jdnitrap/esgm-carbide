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
    # Two VERB slots specifically to exercise real tense agreement (see
    # _mechanics_bonus below) -- none of the templates above have a
    # second verb, so tense-consistency scoring had nothing to act on
    # until this one existed. Not fully fluent English (needs a
    # conjunction to be natural), same level of abstraction the other
    # templates already operate at.
    "two_actions": ["PRONOUN", "VERB", "VERB"],
}

# Which mechanics dimension constrains agreement with which role, and
# which prior role it should agree WITH -- e.g. a second VERB should
# share TENSE with the first VERB already committed. Real English
# grammar rules, not invented: tense consistency across a compound verb
# phrase, number agreement between a pronoun and an earlier noun.
_AGREEMENT_RULES = {
    "VERB": ("TENSE", "VERB"),
    "PRONOUN": ("NUMBER", "NOUN"),
}


def _mechanics_bonus(graph, hub_ids, candidate, expected_role, committed_by_role):
    """Real, hand-given grammar agreement, not invented: if this
    candidate's mechanics value (e.g. TENSE) matches the same dimension
    on an already-committed word of the role it should agree with,
    return a bonus so scoring prefers it. Returns 0 if hub_ids wasn't
    given (mechanics-agreement off, exact old behavior), the rule
    doesn't apply to this role, no prior word of that role has been
    committed yet, or the candidate's value for that dimension is
    genuinely "none" -- a real absence, not something to force."""
    if hub_ids is None or expected_role not in _AGREEMENT_RULES:
        return 0.0
    from grammar_extra import get_value
    dim, agree_with_role = _AGREEMENT_RULES[expected_role]
    prior = committed_by_role.get(agree_with_role)
    if prior is None:
        return 0.0
    prior_value = get_value(graph, hub_ids, dim, prior)
    if prior_value is None:
        return 0.0
    candidate_value = get_value(graph, hub_ids, dim, candidate)
    return 2.0 if candidate_value == prior_value else 0.0


def _score(graph, node):
    return graph.x[node].item()


def generate_sequence(graph, prompt_nodes, grammar=None, max_len=12,
                       temperature=0.0, seed_ticks=4, step_ticks=2, hub_ids=None):
    """Returns (sequence, trace) — sequence is the ordered list of
    committed node ids; trace is per-step detail for verification.

    hub_ids (optional, default None): grammar_extra.py's hub id map.
    When given, candidates that share a real mechanics agreement (verb
    tense, pronoun/noun number) with what's already been committed are
    preferred over ones that don't, on top of raw activation score.
    None (default) preserves the exact original behavior -- no existing
    caller is affected unless it opts in.
    """
    grammar = grammar if grammar is not None else DEFAULT_GRAMMAR
    sequence = []
    trace = []
    slot = 0
    committed_by_role = {}

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

        # split_cap=True: reserve tiled (word-level) candidates their
        # own share of decode()'s cap, so heavy byte-level activity
        # (e.g. after teaching a trained head's predictions back into
        # the graph) can't crowd a real word candidate out of the list
        # -- found necessary by testing, 2026-09-14.
        result = decode(graph, split_cap=True)
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
                picked = max(role_matches, key=lambda c: _score(graph, c)
                             + _mechanics_bonus(graph, hub_ids, c, expected_role, committed_by_role))
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
            committed_by_role[expected_role] = picked
            slot += 1
        if slot >= len(grammar):
            break

    return sequence, trace
