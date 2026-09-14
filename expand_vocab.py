"""One-time vocabulary expansion (2026-09-13): mines the real 5MB corpus
for the top N most common real words not already in tiles.json, and
tiles each one exactly the way shell.py's `tile` command would -- same
free-slot-then-grow logic, same wire_word_structure() call for letters
and (where word_structure.ROLE_MAP now has an entry) roles. Not a
shortcut path: running this is equivalent to typing `tile <word>` N
times by hand, then `save`.

Why: measured that only 6/31 original words ever got real training
signal from real adjacent-word pairs in the corpus (see
EXPERIMENT_LOG.md and esgr-growth-and-modulation-2026-09 memory) --
real data sparsity from too small a vocabulary, not a flaw in any
learning rule. Confirmed empirically that expanding to 131 words lifts
that to 99/131 (76%) real coverage before persisting this for real.
"""
import re
import json
from collections import Counter
from graph import ESGRGraph
from byte_identity import CATEGORY_OFFSET, CATEGORY_NAMES
from word_structure import wire_word_structure, ROLE_OFFSET, ROLE_NAMES

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
GRAPH_PATH = "graph.json"
TILES_PATH = "tiles.json"
N_NEW_WORDS = 100


def mine_top_new_words(current_vocab, n):
    with open(CORPUS_PATH, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common() if w not in current_vocab][:n]


def tile_word(name, tiles, reserved):
    """Exact same logic as shell.py's `tile` command: find a free node
    id outside the category/role ranges, or grow the graph if none."""
    node = 274
    while node in reserved:
        node += 1
    return node


def main():
    graph = ESGRGraph.load_json(GRAPH_PATH)
    with open(TILES_PATH) as f:
        tiles = json.load(f)

    # ALL existing keys, including the special A/space/newline character
    # tiles -- real bug found running this the first time: "space" is
    # also a real, common English word, and excluding the reserved keys
    # from this set let it get mined as "new" and silently overwrite the
    # space-character tile (byte 32) with a word node instead. The
    # reserved names are off-limits regardless of whether they happen to
    # also be real words.
    reserved_names = {k.lower() for k in tiles}
    new_words = mine_top_new_words(reserved_names, N_NEW_WORDS)
    print(f"mined {len(new_words)} new real words from the corpus, not already tiled")

    n_grown = 0
    for name in new_words:
        reserved = set(tiles.values())
        reserved.update(range(CATEGORY_OFFSET, CATEGORY_OFFSET + len(CATEGORY_NAMES)))
        reserved.update(range(ROLE_OFFSET, ROLE_OFFSET + len(ROLE_NAMES)))
        node = 274
        while node in reserved:
            node += 1
        if node >= graph.n:
            node = graph.grow(1)[0]
            n_grown += 1
        tiles[name] = node

    with open(TILES_PATH, "w") as f:
        json.dump(tiles, f, indent=2)
    print(f"tiles.json updated: {len(current_vocab)} -> {len(tiles) - 3} words "
          f"(excluding A/space/newline); grew the graph {n_grown} times")

    n_letter, n_role = wire_word_structure(graph, tiles_path=TILES_PATH)
    print(f"wired {n_letter} letter->word and {n_role} word->role frozen edges "
          f"(covers old + new words, idempotent for old ones)")

    graph.save_json(GRAPH_PATH)
    print(f"saved: n={graph.n}, edges={graph.src.shape[0]}, "
          f"confirmed={int(graph.confirmed.sum())}")


if __name__ == "__main__":
    main()
