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

    # 2026-09-13: the top 100 real words (by real corpus frequency) not
    # already in the original 31 -- mined the same way the original
    # vocabulary was chosen, see EXPERIMENT_LOG.md. Assigned only where
    # confident regardless of context; genuinely ambiguous ones (and,
    # that, but, yet, how, both, than, constantly, only, simultaneously,
    # present, study, enabling) are deliberately left with no role here,
    # same "real none state" convention as the original 31.
    "of": "PREPOSITION", "to": "PREPOSITION", "through": "PREPOSITION",
    "with": "PREPOSITION", "from": "PREPOSITION", "as": "PREPOSITION",
    "across": "PREPOSITION", "for": "PREPOSITION", "at": "PREPOSITION",
    "by": "PREPOSITION",
    "an": "ARTICLE",
    "our": "PRONOUN", "us": "PRONOUN", "their": "PRONOUN", "what": "PRONOUN",
    "these": "PRONOUN", "ourselves": "PRONOUN", "itself": "PRONOUN",
    "its": "PRONOUN", "each": "PRONOUN",
    "human": "NOUN", "world": "NOUN", "life": "NOUN", "patterns": "NOUN",
    "stories": "NOUN", "billions": "NOUN", "knowledge": "NOUN", "power": "NOUN",
    "universe": "NOUN", "consciousness": "NOUN", "history": "NOUN",
    "percent": "NOUN", "revolution": "NOUN", "learning": "NOUN",
    "information": "NOUN", "millions": "NOUN", "ocean": "NOUN", "mind": "NOUN",
    "experience": "NOUN", "existence": "NOUN", "choices": "NOUN",
    "question": "NOUN", "beliefs": "NOUN", "truth": "NOUN", "meaning": "NOUN",
    "matter": "NOUN", "philosophy": "NOUN", "physics": "NOUN", "effect": "NOUN",
    "reality": "NOUN", "experiment": "NOUN", "particle": "NOUN", "years": "NOUN",
    "story": "NOUN", "time": "NOUN", "literature": "NOUN",
    # NOTE: "space" (the real English noun) was here, but "space" is
    # also the reserved tiles.json key for the space CHARACTER (byte
    # 32) -- it collided and got dropped during the 2026-09-13
    # vocabulary expansion (see EXPERIMENT_LOG.md). This ROLE_MAP entry
    # was accidentally left behind afterward: wire_word_structure()'s
    # own word_tiles filter (line ~98) happens to exclude "space" by
    # name, so it was silently harmless here, but grammar_extra.py's
    # SYNTAX dimension (derived from this dict) had no equivalent
    # filter and would have wired the space CHARACTER into SYNTAX=HEAD
    # as if it were the noun -- found by testing. Removed rather than
    # re-added under an alternate key, consistent with the original fix.
    "era": "NOUN", "characters": "NOUN", "language": "NOUN", "reader": "NOUN",
    "data": "NOUN", "networks": "NOUN", "times": "NOUN",
    "are": "VERB", "has": "VERB", "was": "VERB", "becomes": "VERB",
    "reveals": "VERB", "does": "VERB", "live": "VERB", "lead": "VERB",
    "make": "VERB", "know": "VERB", "remain": "VERB",
    "vast": "ADJECTIVE", "ordinary": "ADJECTIVE", "own": "ADJECTIVE",
    "deep": "ADJECTIVE", "complex": "ADJECTIVE", "same": "ADJECTIVE",
    "unexamined": "ADJECTIVE", "classical": "ADJECTIVE", "dark": "ADJECTIVE",
    "great": "ADJECTIVE", "natural": "ADJECTIVE", "written": "ADJECTIVE",

    # 2026-09-14: the ~226 words added by autoexpand/autopilot after the
    # 130-word pass above never got role assignments (a real, flagged
    # gap -- see EXPERIMENT_LOG.md). Same rule as before: only words
    # confident regardless of context. Left deliberately unassigned
    # (real "none" state, not an oversight): and, that, but, yet, how,
    # both, than, constantly, only, simultaneously, present, study,
    # enabling, processing, democratized, better(kept ADJECTIVE below
    # instead), cycles, past, shows, suffering, shape, truly, more,
    # often, function, well, being, born, predetermined, rather, mean,
    # authentically, examined, worth, living, challenge, wherever, may,
    # knowing, use, found, created, forever, seeking, construct, help,
    # sense, even, granted, doing, so, search, far, stranger, imagined,
    # existing, states, observed, this, plays, double, depending,
    # measure, defying, cannot(kept VERB below instead), position,
    # estimated, trillion, billion, two, made, literally, composed,
    # forged, best, worst, shadow, love, sacrifice, transport, long,
    # finish, reading, mirror, reflecting, hopes, fears, dreams.
    "systems": "NOUN", "ai": "NOUN", "technology": "NOUN", "people": "NOUN",
    "forest": "NOUN", "oxygen": "NOUN", "water": "NOUN", "depths": "NOUN",
    "destruction": "NOUN", "climate": "NOUN", "system": "NOUN",
    "biodiversity": "NOUN", "sea": "NOUN", "food": "NOUN", "humans": "NOUN",
    "essence": "NOUN", "nature": "NOUN", "actions": "NOUN",
    "responsibility": "NOUN", "freedom": "NOUN", "burden": "NOUN",
    "opportunity": "NOUN", "assumptions": "NOUN", "wisdom": "NOUN",
    "pursuit": "NOUN", "happiness": "NOUN", "attainment": "NOUN",
    "engagement": "NOUN", "thing": "NOUN", "chaos": "NOUN",
    "randomness": "NOUN", "narratives": "NOUN", "distinction": "NOUN",
    "foundations": "NOUN", "certainty": "NOUN", "assumption": "NOUN",
    "particles": "NOUN", "superposition": "NOUN", "notions": "NOUN",
    "locality": "NOUN", "uncertainty": "NOUN", "principle": "NOUN",
    "momentum": "NOUN", "precision": "NOUN", "cosmos": "NOUN",
    "galaxies": "NOUN", "stars": "NOUN", "sun": "NOUN", "star": "NOUN",
    "galaxy": "NOUN", "planet": "NOUN", "bang": "NOUN", "epoch": "NOUN",
    "energy": "NOUN", "stardust": "NOUN", "elements": "NOUN",
    "furnaces": "NOUN", "structure": "NOUN", "tale": "NOUN",
    "cities": "NOUN", "resurrection": "NOUN", "redemption": "NOUN",
    "individuals": "NOUN", "loyalty": "NOUN", "word": "NOUN",
    "minds": "NOUN", "hearts": "NOUN", "truths": "NOUN",
    "imagination": "NOUN", "page": "NOUN", "observer": "NOUN",
    "participation": "NOUN", "role": "NOUN", "fabric": "NOUN",
    "slit": "NOUN", "wave": "NOUN", "entanglement": "NOUN",
    "connections": "NOUN", "distances": "NOUN", "mechanics": "NOUN",

    "understand": "VERB", "create": "VERB", "tries": "VERB",
    "define": "VERB", "seek": "VERB", "must": "VERB", "exist": "VERB",
    "suggests": "VERB", "demonstrates": "VERB", "behaves": "VERB",
    "creates": "VERB", "contains": "VERB", "sustains": "VERB",
    "spans": "VERB", "comprise": "VERB", "teaches": "VERB",
    "examine": "VERB", "take": "VERB", "discover": "VERB",
    "leads": "VERB", "precedes": "VERB", "cannot": "VERB",
    "grapple": "VERB", "inhabit": "VERB", "transcends": "VERB",
    "speak": "VERB", "have": "VERB",

    "mental": "ADJECTIVE", "curious": "ADJECTIVE", "true": "ADJECTIVE",
    "useful": "ADJECTIVE", "rare": "ADJECTIVE", "eternal": "ADJECTIVE",
    "quantum": "ADJECTIVE", "subatomic": "ADJECTIVE", "multiple": "ADJECTIVE",
    "fundamental": "ADJECTIVE", "instantaneous": "ADJECTIVE",
    "perfect": "ADJECTIVE", "mysterious": "ADJECTIVE", "stellar": "ADJECTIVE",
    "mathematical": "ADJECTIVE", "underlying": "ADJECTIVE",
    "universal": "ADJECTIVE", "dependent": "ADJECTIVE", "blue": "ADJECTIVE",
    "hot": "ADJECTIVE", "better": "ADJECTIVE",

    "about": "PREPOSITION", "until": "PREPOSITION", "beyond": "PREPOSITION",
    "after": "PREPOSITION",

    "every": "ARTICLE",
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
        if node >= graph.n:
            # tiles.json can reference a bigger, grown graph than the one
            # passed in here (e.g. a fresh, small test graph) -- skip
            # what doesn't fit rather than crash; same "real, valid
            # skip" convention as a word with no role.
            continue
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
    Read-only — checks for a confirmed, non-suspended word->role_hub
    edge. Checking confirmed alone was a real bug found by testing:
    contradiction-suspension (graph.py's E_contr auto-suspend) sets
    only `suspended`, deliberately leaving `confirmed` untouched so a
    later confirm()/reject() has something to resolve -- so a
    confirmed-then-suspended edge kept reporting its role as if the
    fight never happened. decode() already treats suspend as silence;
    this now matches that.
    """
    for i, role_node in enumerate(range(ROLE_OFFSET, ROLE_OFFSET + len(ROLE_NAMES))):
        e = graph.find_edge(word_node, role_node)
        if e is not None and bool(graph.confirmed[e]) and not bool(graph.suspended[e]):
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
