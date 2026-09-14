"""Local, single-hop, ground-truth-corrected learning -- the specific
piece plain Hebbian can't do: push toward the real correct next word
even when it isn't currently active, and suppress a wrong edge that
currently out-scores it. No backprop, no hidden layers, no opaque head:
one edge gets nudged, using only locally-available information (the
context node, the real next word, and the graph's own current scoring)
plus one piece of outside information a person already has while
training on real text -- what the real next word actually was.

This is NOT the same claim as Hebbian's co-fire rule in graph.py: that
rule only ever touches (u, v) that were BOTH active on the same tick.
supervised_step() deliberately updates the (context, correct_next) edge
regardless of whether correct_next happened to be active -- that's the
one thing an error signal buys you that co-firing alone cannot.
"""
import torch


def top_candidate(graph, context_node, candidate_pool):
    """The graph's own current best guess for 'what comes after
    context_node', scored by trust*weight -- the same scoring
    convention graph.propose() already uses. candidate_pool restricts
    the guess to real word-tile ids, not every node in the graph."""
    out_edges = (graph.src == context_node).nonzero().flatten().tolist()
    scored = [(i, (graph.tau[i] * graph.w[i]).item())
              for i in out_edges if int(graph.dst[i]) in candidate_pool]
    if not scored:
        return None
    return max(scored, key=lambda t: t[1])[0]


def supervised_step(graph, context_node, correct_node, candidate_pool, lr=0.2, tau_lr=0.1):
    """One correction, given one real (context, correct_next) pair from
    real text. Returns True if the graph's own guess was already right
    before the correction (so accuracy can be measured honestly, on the
    guess BEFORE this step's update touches anything)."""
    correct_idx = graph.find_edge(context_node, correct_node)
    if correct_idx is None:
        correct_idx = graph.add_learnable_edge(context_node, correct_node)

    guess_idx = top_candidate(graph, context_node, candidate_pool)
    was_correct = (guess_idx == correct_idx)

    if was_correct:
        # reinforce a correct call a bit further, same direction tick()
        # would already push it, just faster and not contingent on
        # actually co-firing this tick. Clamped to graph.max_weight --
        # real bug found by testing at scale (2026-09-14, full 5MB
        # corpus): this used to be an unbounded += with no ceiling, and
        # an edge taught hundreds of times (extremely common byte pairs
        # like ','->' ' occur constantly in real text) grew to w=536,
        # dwarfing everything else in the graph (frozen edges sit at
        # 1.0) and drowning out word-level generation entirely in kWTA
        # competition. tick()'s own Hebbian growth is naturally bounded
        # by the m_raw/(1+abs(m_raw)) saturation in message-passing;
        # supervised_step writes directly and has no such saturation of
        # its own, so it needs an explicit one.
        if not bool(graph.frozen[correct_idx]):
            graph.w[correct_idx] = torch.clamp(graph.w[correct_idx] + lr * 0.5, max=graph.max_weight)
            graph.tau[correct_idx] = torch.clamp(graph.tau[correct_idx] + tau_lr * 0.5, max=1.0)
    else:
        if not bool(graph.frozen[correct_idx]):
            graph.w[correct_idx] = torch.clamp(graph.w[correct_idx] + lr, max=graph.max_weight)
            graph.tau[correct_idx] = torch.clamp(graph.tau[correct_idx] + tau_lr, max=1.0)
        if guess_idx is not None and not bool(graph.frozen[guess_idx]):
            graph.w[guess_idx] = torch.clamp(graph.w[guess_idx] - lr, min=0.0)
            graph.tau[guess_idx] = torch.clamp(graph.tau[guess_idx] - tau_lr, min=0.0)

    return was_correct
