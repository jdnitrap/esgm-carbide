"""Vocabulary expansion, mined from the real 5MB corpus -- the "safe"
self-expansion path: frequency alone decides WHAT is worth adding,
unattended, no person hand-picks a word list. auto_expand_vocab() is the
reusable core (importable from shell.py's `autoexpand` command or
anywhere else); main() is the one-shot CLI script that first used it
2026-09-13 to take the vocabulary 31 -> 130 words.

What this still is NOT: the graph does not decide *when* or *how many*
words to add on its own -- a person (or a future policy) still triggers
a run and picks n_words. True self-directed expansion (the system
deciding for itself, with no external trigger, that it needs more
structure) is a real, separate, harder capability -- see
EXPERIMENT_LOG.md, 2026-09-14 entry.
"""
import re
import json
from collections import Counter
from graph import ESGRGraph
from byte_identity import CATEGORY_OFFSET, CATEGORY_NAMES
from word_structure import wire_word_structure, ROLE_OFFSET, ROLE_NAMES
from grammar_extra import hub_node_ids

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
GRAPH_PATH = "graph.json"
TILES_PATH = "tiles.json"
N_NEW_WORDS = 100


def auto_expand_vocab(graph, tiles, n_words=20, corpus_path=CORPUS_PATH,
                       min_frequency=1, tiles_path=TILES_PATH):
    """Mines corpus_path for the n_words most common real words not
    already in `tiles` (case-insensitive; the reserved A/space/newline
    character tiles are always excluded, real bug found 2026-09-13 --
    "space" is also a real word and collided with the reserved tile).
    Tiles each one via the exact free-slot-then-grow() path shell.py's
    `tile` command uses, wires letters/roles. Mutates `tiles` (a dict)
    and `graph` in place; writes tiles.json to disk immediately (same
    convention as `tile`) but does NOT save graph.json -- caller still
    decides when to persist the grown graph.

    Returns {"mined": [...new words...], "grown": n_new_nodes,
    "letter_edges": n, "role_edges": n}.
    """
    reserved_names = {k.lower() for k in tiles}
    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    freq = Counter(tokens)
    new_words = [w for w, c in freq.most_common()
                 if w not in reserved_names and c >= min_frequency][:n_words]

    n_grown = 0
    grammar_hubs = hub_node_ids()  # dynamically-allocated, not a fixed offset -- see hub_node_ids()
    for name in new_words:
        reserved = set(tiles.values())
        reserved.update(range(CATEGORY_OFFSET, CATEGORY_OFFSET + len(CATEGORY_NAMES)))
        reserved.update(range(ROLE_OFFSET, ROLE_OFFSET + len(ROLE_NAMES)))
        reserved.update(grammar_hubs)
        node = 274
        while node in reserved:
            node += 1
        if node >= graph.n:
            node = graph.grow(1)[0]
            n_grown += 1
        tiles[name] = node

    with open(tiles_path, "w") as f:
        json.dump(tiles, f, indent=2)

    n_letter, n_role = wire_word_structure(graph, tiles_path=tiles_path)
    return {"mined": new_words, "grown": n_grown, "letter_edges": n_letter, "role_edges": n_role}


def main():
    graph = ESGRGraph.load_json(GRAPH_PATH)
    with open(TILES_PATH) as f:
        tiles = json.load(f)
    before = len(tiles) - 3

    stats = auto_expand_vocab(graph, tiles, n_words=N_NEW_WORDS,
                               corpus_path=CORPUS_PATH, tiles_path=TILES_PATH)

    print(f"mined {len(stats['mined'])} new real words from the corpus, not already tiled")
    print(f"tiles.json updated: {before} -> {len(tiles) - 3} words "
          f"(excluding A/space/newline); grew the graph {stats['grown']} times")
    print(f"wired {stats['letter_edges']} letter->word and {stats['role_edges']} word->role "
          f"frozen edges (covers old + new words, idempotent for old ones)")

    graph.save_json(GRAPH_PATH)
    print(f"saved: n={graph.n}, edges={graph.src.shape[0]}, "
          f"confirmed={int(graph.confirmed.sum())}")


if __name__ == "__main__":
    main()
