"""More hand-given English mechanics, layered on top of word_structure.py
without touching it -- same exact pattern (frozen, confirmed edges,
words wired to hub nodes for a category), just more dimensions than the
original 6 syntactic roles cover. Given for free, same reasoning as
ROLE_MAP: real structural facts about English, cheaper to state once
than to make the system re-derive from a corpus that's too sparse to
teach it (see the held-out accuracy experiment this replaces -- widening
the adjacency window only made things worse).

Every word below is a REAL word already in tiles.json -- nothing here
invents a category with no member. Words deliberately left out of a
given map (e.g. "you" from NUMBER) are a real "none" state, same
convention as ROLE_MAP's "not"/"yes"/"no".

Uses graph.grow() for hub nodes instead of hunting for free ids in the
cramped 262-298 range -- the reason that's now safe to do at all.
"""

TENSE_NAMES = ["PAST", "PRESENT"]
TENSE_MAP = {
    "sat": "PAST",
    "ran": "PAST",
    "is": "PRESENT",
    "likes": "PRESENT",
    "eats": "PRESENT",
}

NUMBER_NAMES = ["SINGULAR", "PLURAL"]
NUMBER_MAP = {
    "I": "SINGULAR", "he": "SINGULAR", "she": "SINGULAR", "it": "SINGULAR",
    "we": "PLURAL", "they": "PLURAL",
    # "you" deliberately absent: English collapses 2nd-person singular
    # and plural into one word -- a real ambiguity, not a gap to fill.
}

ANIMACY_NAMES = ["ANIMATE", "INANIMATE"]
ANIMACY_MAP = {
    "cat": "ANIMATE", "dog": "ANIMATE", "bird": "ANIMATE",
    "fish": "ANIMATE", "man": "ANIMATE",
    "mat": "INANIMATE", "house": "INANIMATE",
}

# "yes"/"no"/"not" are real words with no syntactic ROLE (word_structure.py
# leaves them role-less on purpose) -- but they aren't roleless in
# English grammar generally, they're just a DIFFERENT kind of category
# (discourse particles / negation) than NOUN/VERB/etc. Giving them their
# own dimension is the linguistically correct fix, not a patch onto ROLE.
DISCOURSE_NAMES = ["AFFIRM", "NEGATE", "NEGATOR"]
DISCOURSE_MAP = {
    "yes": "AFFIRM",
    "no": "NEGATE",
    "not": "NEGATOR",
}

DIMENSIONS = {
    "TENSE": (TENSE_NAMES, TENSE_MAP),
    "NUMBER": (NUMBER_NAMES, NUMBER_MAP),
    "ANIMACY": (ANIMACY_NAMES, ANIMACY_MAP),
    "DISCOURSE": (DISCOURSE_NAMES, DISCOURSE_MAP),
}


def wire_grammar_extra(graph, tiles_path="tiles.json"):
    """Grows one hub node per category value across all four dimensions,
    then wires each mapped word to its hub with a frozen, confirmed
    edge -- identical mechanism to word_structure.py's word->role wiring,
    just a different set of hubs. Returns {dimension_name: {value_name:
    hub_node_id}} so a caller (or a future generation step) can look
    hubs up by name without re-deriving ids.
    """
    import json
    with open(tiles_path) as f:
        tiles = json.load(f)

    hub_ids = {}
    n_edges = 0
    for dim_name, (names, word_map) in DIMENSIONS.items():
        new_ids = graph.grow(len(names))
        hub_ids[dim_name] = dict(zip(names, new_ids))
        for word, value in word_map.items():
            node = tiles.get(word) or tiles.get(word.lower())
            if node is None:
                continue  # word not tiled yet -- skip, don't invent an id
            hub_node = hub_ids[dim_name][value]
            graph.add_fixed_edge(node, hub_node, weight=1.0, trust=1.0)
            n_edges += 1
    return hub_ids, n_edges


def get_value(graph, hub_ids, dim_name, word_node):
    """Mirror of word_structure.get_role() for any of these dimensions:
    returns the value name (e.g. "PAST") for a word node, or None if it
    has no confirmed edge into this dimension's hubs -- a real 'none'
    state, not missing data."""
    for value_name, hub_node in hub_ids[dim_name].items():
        e = graph.find_edge(word_node, hub_node)
        if e is not None and bool(graph.confirmed[e]):
            return value_name
    return None


def words_with_value(graph, hub_ids, dim_name, value_name):
    """Mirror of word_structure.words_with_role(): every word node with
    a confirmed edge into this specific hub."""
    hub_node = hub_ids[dim_name][value_name]
    incoming = (graph.dst == hub_node).nonzero().flatten().tolist()
    return [int(graph.src[e]) for e in incoming if bool(graph.confirmed[e])]
