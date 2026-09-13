"""Letter-grounding and grammatical-role wiring for word tiles, layered
on top of the existing MDBE byte->category structure without touching
it. Two new kinds of frozen, confirmed edges, following the exact same
pattern byte_identity.py already established for byte->category:

  1. byte (letter) -> word: each word tile gets a fixed edge FROM each
     distinct byte spelling it — many bytes -> one hub, same shape as
     byte->category, just with the word as the hub instead of a class.
  2. word -> role: each word tile gets a fixed edge TO a grammatical
     role hub, IF it has one. Not every word gets a role — that's a
     real, valid state ("none"), not a gap to fill in later.

Existing MDBE wiring, tiles.json, and all shell commands are untouched.
Uses node ids 275-279 for the 5 role hubs — well within the existing
n=300 node budget, so this does NOT grow n.
"""
import json

ROLE_NAMES = ["NOUN", "VERB", "ARTICLE", "PRONOUN", "PREPOSITION", "ADJECTIVE"]
ROLE_OFFSET = 275

ROLE_MAP = {
    "the": "ARTICLE",
    "a": "ARTICLE",
    "cat": "NOUN",
    "dog": "NOUN",
    "mat": "NOUN",
    "sat": "VERB",
    "is": "VERB",
    "on": "PREPOSITION",
    "I": "PRONOUN",
    "you": "PRONOUN",
    # "not", "yes", "no" intentionally have no role: real "none" state.
    "bird": "NOUN",
    "fish": "NOUN",
    "man": "NOUN",
    "house": "NOUN",
    "ran": "VERB",
    "likes": "VERB",
    "eats": "VERB",
    "big": "ADJECTIVE",
    "small": "ADJECTIVE",
    "red": "ADJECTIVE",
    "in": "PREPOSITION",
    "under": "PREPOSITION",
    "near": "PREPOSITION",
    "he": "PRONOUN",
    "she": "PRONOUN",
    "it": "PRONOUN",
    "we": "PRONOUN",
    "they": "PRONOUN",
}

ROLE_LABELS = {ROLE_OFFSET + i: name for i, name in enumerate(ROLE_NAMES)}


def wire_word_structure(graph, tiles_path="tiles.json"):
    assert graph.n >= ROLE_OFFSET + len(ROLE_NAMES), (
        f"graph needs >= {ROLE_OFFSET + len(ROLE_NAMES)} nodes for role hubs"
    )
    with open(tiles_path) as f:
        tiles = json.load(f)
    word_tiles = {k: v for k, v in tiles.items() if k not in ("A", "space", "newline")}

    n_letter_edges = 0
    n_role_edges = 0
    for word, node in word_tiles.items():
        for ch in set(word):
            byte_val = ord(ch)
            graph.add_fixed_edge(byte_val, node, weight=1.0, trust=1.0)
            n_letter_edges += 1

        role = ROLE_MAP.get(word)
        if role is not None:
            role_node = ROLE_OFFSET + ROLE_NAMES.index(role)
            graph.add_fixed_edge(node, role_node, weight=1.0, trust=1.0)
            n_role_edges += 1

    return n_letter_edges, n_role_edges


def get_role(graph, word_node):
    """Returns the role NAME (e.g. "NOUN") for a word node, or None if
    it has no role edge (a real, valid "none" state, not missing data).
    Read-only — checks for a confirmed word->role_hub edge.
    """
    for i, role_node in enumerate(range(ROLE_OFFSET, ROLE_OFFSET + len(ROLE_NAMES))):
        e = graph.find_edge(word_node, role_node)
        if e is not None and bool(graph.confirmed[e]):
            return ROLE_NAMES[i]
    return None


def words_with_role(graph, role_name):
    """Structural query, the reverse direction of get_role: which word
    nodes have a confirmed edge INTO this role hub? Direct use of
    already-wired facts, not a search through currently-active nodes —
    the engine (sequence.py) uses this to actively drive activation
    toward grammatically appropriate candidates instead of passively
    hoping one shows up.
    """
    role_node = ROLE_OFFSET + ROLE_NAMES.index(role_name)
    incoming = (graph.dst == role_node).nonzero().flatten().tolist()
    return [int(graph.src[e]) for e in incoming if bool(graph.confirmed[e])]
