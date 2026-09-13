"""Text generation for ESGR — a sequential loop around the EXISTING
mechanisms (tick, decode), not a new architecture: no attention, no
softmax over a vocabulary, no learned unembed. The "model" is still
just the graph; this is the control loop that drives it repeatedly and
reads what it says at each step, the same relationship Carbide's own
generate() had to its model.

Mechanism: seed with a prompt (poke it in), then either let the graph
free-run on its own dynamics, or close the loop by re-injecting
whatever it just emitted as the next tick's stimulus (autoregressive-
style — what was just "said" becomes new context for what's said next).
"""
import torch
from decode import decode

STIM_VALUE = 3.0


def resolve_prompt(prompt_tokens, tiles):
    """prompt_tokens: list of name-or-int strings."""
    lower_tiles = {k.lower(): v for k, v in tiles.items()}
    nodes = []
    for t in prompt_tokens:
        try:
            nodes.append(int(t))
        except ValueError:
            n = lower_tiles.get(t.lower())
            if n is not None:
                nodes.append(n)
    return nodes


def generate(graph, prompt_nodes, n_ticks=30, reinject=True, seed_ticks=4):
    """Returns the list of decode() results, one per tick after seeding."""
    stim = torch.zeros(graph.n)
    for node in prompt_nodes:
        stim[node] = STIM_VALUE
    for _ in range(seed_ticks):
        graph.tick(external_input=stim)

    log = []
    for _ in range(n_ticks):
        result = decode(graph)
        log.append(result)
        if reinject and result["status"] == "emit" and result["bytes"]:
            next_stim = torch.zeros(graph.n)
            for b in result["bytes"]:
                next_stim[b] = STIM_VALUE
            graph.tick(external_input=next_stim)
        else:
            graph.tick(external_input=None)
    return log
