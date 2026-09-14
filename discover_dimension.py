"""Lets the graph propose a genuinely NEW category/dimension for itself
from real data, instead of only filling in categories a human already
named (ROLE/TENSE/DISCOURSE/etc). Explicit user direction, 2026-09-14:
"make it have the ability to add dimension to itself."

Real technique, not invented: the distributional hypothesis (words that
occur in similar contexts tend to share grammatical function) is a
standard, citable basis for unsupervised word clustering in
linguistics/NLP. Here the "context" is kept simple and directly
grounded in structure the graph already trusts: for each word with NO
confirmed ROLE and no confirmed value in any grammar_extra dimension
(a real gap, not noise), look at the ROLE of the word immediately
before and after it, mined from the real corpus. Words that share the
same dominant (prev_role, next_role) signature behave alike, and if
enough of them cluster together, that's real evidence they share some
category the graph doesn't have a name for yet.

Same human-gated pattern as fact_gate.py: this module only PROPOSES
clusters (propose_dimensions, read-only, no graph writes). Nothing
commits automatically -- a person reviews a proposal and calls
confirm_dimension() explicitly to make it real (grow a hub node, wire
frozen/confirmed edges, persist to discovered_dimensions.json). No
different in spirit from the propose/confirm split everywhere else in
this repo, just one level up: proposing a CATEGORY instead of a FACT.
"""
import json
import os
import re
from collections import Counter, defaultdict

from word_structure import ROLE_MAP
from grammar_extra import DIMENSIONS

DISCOVERED_PATH = "discovered_dimensions.json"


def _already_categorized(word):
    """True if word already has a confirmed value somewhere (ROLE or
    any grammar_extra dimension) -- these words are NOT candidates,
    the whole point is finding what's NOT covered yet."""
    if ROLE_MAP.get(word) is not None:
        return True
    for _, (_, word_map) in DIMENSIONS.items():
        if word in word_map:
            return True
    return False


def mine_context_signatures(corpus_path, sample_bytes=2_000_000):
    """Real pass over the actual corpus: for every word, count how
    often each (prev_role, next_role) pair occurs around it, using
    ROLE_MAP as the only source of truth for neighbor roles (None for
    an untagged neighbor -- a real, honest 'unknown neighbor' state,
    not guessed). Returns {word: Counter({(prev_role, next_role): n})}.
    """
    with open(corpus_path, "rb") as f:
        text = f.read(sample_bytes).decode("ascii", errors="replace").lower()
    words = re.findall(r"[a-z']+", text)

    sigs = defaultdict(Counter)
    for i in range(1, len(words) - 1):
        w = words[i]
        if _already_categorized(w):
            continue
        prev_role = ROLE_MAP.get(words[i - 1])
        next_role = ROLE_MAP.get(words[i + 1])
        sigs[w][(prev_role, next_role)] += 1
    return sigs


def propose_dimensions(corpus_path, sample_bytes=2_000_000, min_freq=15, min_cluster_size=5):
    """Read-only. Mines real context signatures, then groups
    uncategorized words by their DOMINANT (prev_role, next_role)
    signature -- words sharing the same most-common neighbor pattern
    are proposed as one cluster. Only clusters with >= min_cluster_size
    real distinct words, each seen >= min_freq times, are returned --
    small or noisy clusters are real evidence of nothing and are
    dropped rather than proposed.

    Returns a list of {signature, words, total_occurrences} dicts,
    largest cluster first. Does not touch the graph at all.
    """
    sigs = mine_context_signatures(corpus_path, sample_bytes)
    by_dominant_sig = defaultdict(list)
    for word, counter in sigs.items():
        total = sum(counter.values())
        if total < min_freq:
            continue
        dominant_sig, count = counter.most_common(1)[0]
        by_dominant_sig[dominant_sig].append((word, total))

    proposals = []
    for sig, word_counts in by_dominant_sig.items():
        if len(word_counts) < min_cluster_size:
            continue
        word_counts.sort(key=lambda wc: -wc[1])
        proposals.append({
            "signature": sig,
            "words": [w for w, _ in word_counts],
            "total_occurrences": sum(c for _, c in word_counts),
        })
    proposals.sort(key=lambda p: -p["total_occurrences"])
    return proposals


def confirm_dimension(graph, tiles, dimension_name, words, value_name="MEMBER"):
    """Human-gated commit: grows ONE new hub node, wires every given
    word to it with a frozen, confirmed edge -- same mechanism as
    word_structure.py/grammar_extra.py use for every other dimension --
    then persists {dimension_name: {value_name: hub_id, "words": [...]}}
    into discovered_dimensions.json so it survives a reload and can be
    picked up by future tag-table building. Only ever called explicitly
    by a person reviewing a propose_dimensions() result; nothing in
    this repo calls this automatically."""
    hub_id = graph.grow(1)[0]
    wired = []
    for word in words:
        node = tiles.get(word) or tiles.get(word.lower())
        if node is None or node >= graph.n:
            continue
        graph.add_fixed_edge(node, hub_id, weight=1.0, trust=1.0)
        wired.append(word)

    discovered = {}
    if os.path.exists(DISCOVERED_PATH):
        with open(DISCOVERED_PATH) as f:
            discovered = json.load(f)
    discovered[dimension_name] = {"hub_id": hub_id, "value_name": value_name, "words": wired}
    with open(DISCOVERED_PATH, "w") as f:
        json.dump(discovered, f, indent=2)

    return hub_id, wired
