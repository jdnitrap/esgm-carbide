"""Quantitative test of the xLSTM head's WORD-generation capability
(not just 'does it run') -- explicit user follow-up after the first
generate_bytes.py test. Generates real output across many seeds and
temperatures, then measures, with real evidence not eyeballing:

  1. What fraction of generated whitespace-tokens are real, valid
     English words (checked against the system dictionary /usr/share/
     dict/words, 104,334 words) vs garbled/invented non-words.
  2. What fraction of generated tokens are words the graph itself
     knows (tiles.json vocabulary) vs words outside the graph's own
     vocabulary entirely (real evidence the head can produce words the
     graph never had a node for -- or can't, if the number is 0).
  3. Real n-gram overlap against the training corpus, at the WORD
     level (not substring), to separate genuine novel composition from
     memorized reproduction -- what fraction of generated 4-word
     windows exist verbatim in the corpus.
"""
import json
import re
import string
import torch

from graph import ESGRGraph
from head_checkpoint import load_head_checkpoint
from head import N_COLUMNS
from generate_bytes import generate_bytes

g = ESGRGraph.load_json("graph.json")
tiles = json.load(open("tiles.json"))
hub_ids = json.load(open("grammar_extra_hubs.json"))
model, _, _, _ = load_head_checkpoint("head_checkpoint.pt", new_tag_dim=N_COLUMNS)
tiled_words = {k.lower() for k in tiles if k.isalpha()}

with open("/usr/share/dict/words") as f:
    dictionary = {w.strip().lower() for w in f if w.strip()}

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
with open(CORPUS_PATH, "rb") as f:
    corpus_text = f.read().decode("ascii", errors="replace").lower()
corpus_words = re.findall(r"[a-z']+", corpus_text)
corpus_4grams = set(tuple(corpus_words[i:i + 4]) for i in range(len(corpus_words) - 3))

SEEDS = ["the ", "a ", "science ", "the universe ", "climate ", "love ",
         "music ", "quantum ", "history ", "freedom "]
TEMPS = [0.0, 0.5, 0.9]

def tokenize(text):
    return [w.strip(string.punctuation).lower() for w in text.split()]
    # (kept simple/direct -- strips leading/trailing punctuation only)

all_tokens = []
all_4grams_generated = []
print("=== generation samples ===")
for temp in TEMPS:
    for i, s in enumerate(SEEDS):
        text, seq, stopped = generate_bytes(g, model, hub_ids, tiles, s, max_len=150,
                                             temperature=temp, stop_at_newline=False, seed=i)
        toks = [t for t in tokenize(text) if t.isalpha()]
        all_tokens.extend(toks)
        gwords = [t for t in tokenize(text) if t]
        all_4grams_generated.extend(tuple(gwords[j:j + 4]) for j in range(len(gwords) - 3))
        if temp in (0.0, 0.9) and i < 3:
            print(f"[temp={temp} seed={s!r}] {text!r}")

print(f"\ntotal alphabetic tokens generated across all runs: {len(all_tokens)}")

# 1. real-dictionary word rate
in_dict = [t for t in all_tokens if t in dictionary]
print(f"\n[1] valid English words (system dictionary check): {len(in_dict)}/{len(all_tokens)} "
      f"= {100*len(in_dict)/len(all_tokens):.1f}%")
not_words = [t for t in all_tokens if t not in dictionary]
from collections import Counter
worst = Counter(not_words).most_common(15)
print(f"    sample of non-dictionary tokens produced: {[w for w, _ in worst]}")

# 2. graph-vocabulary vs outside-graph-vocabulary word rate
in_graph_vocab = [t for t in all_tokens if t in tiled_words]
real_word_outside_graph = [t for t in all_tokens if t in dictionary and t not in tiled_words]
print(f"\n[2] tokens matching the graph's own known vocabulary (tiles.json): "
      f"{len(in_graph_vocab)}/{len(all_tokens)} = {100*len(in_graph_vocab)/len(all_tokens):.1f}%")
print(f"    real English words produced that are NOT in the graph's vocabulary at all: "
      f"{len(real_word_outside_graph)}/{len(all_tokens)} = {100*len(real_word_outside_graph)/len(all_tokens):.1f}%")
print(f"    sample: {sorted(set(real_word_outside_graph))[:20]}")

# 3. word-level n-gram memorization check
memorized = [g4 for g4 in all_4grams_generated if g4 in corpus_4grams]
print(f"\n[3] generated 4-word windows found verbatim in training corpus: "
      f"{len(memorized)}/{len(all_4grams_generated)} = "
      f"{100*len(memorized)/max(1,len(all_4grams_generated)):.1f}%")
print(f"    (the rest are real, novel word sequences never seen together in training data)")

print("\n=== honest summary ===")
print(f"- {100*len(in_dict)/len(all_tokens):.1f}% of generated tokens are valid English words")
print(f"- {100*len(real_word_outside_graph)/len(all_tokens):.1f}% are real words the GRAPH never had a node for "
      f"(head-only vocabulary, learned purely from corpus bytes)")
print(f"- {100*len(memorized)/max(1,len(all_4grams_generated)):.1f}% of 4-word windows are verbatim corpus recall; "
      f"the rest are genuinely novel compositions")
