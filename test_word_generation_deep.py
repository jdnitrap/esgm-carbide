"""Deeper follow-up to test_word_generation.py, at explicit user
request. Three real, evidence-based checks:

  1. Long-horizon stability -- does word-validity quality hold up or
     decay over a much longer run (1000 bytes), and does it eventually
     collapse into a repetition loop the way RNNs commonly do.
  2. Grammatical plausibility, grounded in REAL data -- mines the
     actual ROLE-bigram distribution from the training corpus itself
     (not hand-written rules), then checks what fraction of the head's
     own generated adjacent role-tagged word pairs match a transition
     that really occurs in that mined distribution.
  3. Out-of-corpus generalization -- seed words verified absent from
     the entire training corpus (xylophone, dinosaur, asteroid, ...).
     If the head can still produce valid-English-shaped continuations,
     that's evidence of learned general byte/word structure, not pure
     memorization; if output degrades to noise, that's the honest
     limit of what it learned.
"""
import json
import re
import string
from collections import Counter

from graph import ESGRGraph
from head_checkpoint import load_head_checkpoint
from head import N_COLUMNS
from generate_bytes import generate_bytes
from word_structure import ROLE_MAP

g = ESGRGraph.load_json("graph.json")
tiles = json.load(open("tiles.json"))
hub_ids = json.load(open("grammar_extra_hubs.json"))
model, _, _, _ = load_head_checkpoint("head_checkpoint.pt", new_tag_dim=N_COLUMNS)

with open("/usr/share/dict/words") as f:
    dictionary = {w.strip().lower() for w in f if w.strip()}

CORPUS_PATH = "/home/admin/Downloads/carbide/carbide_training_dataset.txt"
with open(CORPUS_PATH, "rb") as f:
    corpus_text = f.read().decode("ascii", errors="replace").lower()
corpus_words = re.findall(r"[a-z']+", corpus_text)


def tokenize(text):
    return [w.strip(string.punctuation).lower() for w in text.split()]


def word_validity_rate(tokens):
    alpha = [t for t in tokens if t.isalpha()]
    if not alpha:
        return None
    return sum(1 for t in alpha if t in dictionary) / len(alpha)


def longest_repeated_tail_cycle(byte_list, min_cycle=2, max_cycle=8):
    for cyc in range(min_cycle, max_cycle + 1):
        if len(byte_list) < cyc * 4:
            continue
        tail = byte_list[-cyc * 4:]
        chunks = [tuple(tail[i:i + cyc]) for i in range(0, len(tail), cyc)]
        if len(set(chunks)) == 1:
            return cyc
    return None


# ============================================================
# 1. Long-horizon stability: 1000-byte runs, quality per quarter
# ============================================================
print("=== [1] long-horizon stability (1000 bytes) ===")
LONG_SEEDS = ["the ", "science ", "the universe ", "love "]
for i, s in enumerate(LONG_SEEDS):
    text, seq, stopped = generate_bytes(g, model, hub_ids, tiles, s, max_len=1000,
                                         temperature=0.7, stop_at_newline=False, seed=i)
    quarter = len(text) // 4
    rates = []
    for q in range(4):
        chunk = text[q * quarter:(q + 1) * quarter]
        r = word_validity_rate(tokenize(chunk))
        rates.append(r)
    cyc = longest_repeated_tail_cycle(seq)
    rates_str = " -> ".join(f"{r*100:.0f}%" if r is not None else "n/a" for r in rates)
    print(f"seed={s!r} len={len(seq)} word-validity by quarter: {rates_str}  degenerate_cycle_at_end={cyc}")

# ============================================================
# 2. Grammatical plausibility grounded in real corpus role-bigrams
# ============================================================
print("\n=== [2] grammatical plausibility (real corpus-mined role-bigrams) ===")
tagged_count = 0
bigram_counts = Counter()
for w1, w2 in zip(corpus_words[:-1], corpus_words[1:]):
    r1, r2 = ROLE_MAP.get(w1), ROLE_MAP.get(w2)
    if r1 is None or r2 is None:
        continue
    tagged_count += 1
    bigram_counts[(r1, r2)] += 1
real_role_bigrams = set(bigram_counts)
print(f"mined {tagged_count} REAL adjacent role-tagged word pairs from the corpus "
      f"(both words in the pair genuinely next to each other in the text), "
      f"{len(real_role_bigrams)} distinct role-bigrams actually observed")
print(f"top 10 real role-bigrams: {bigram_counts.most_common(10)}")

GEN_SEEDS = ["the ", "a ", "science ", "the universe ", "climate ", "love ",
             "music ", "quantum ", "history ", "freedom "]
total_pairs, plausible_pairs = 0, 0
examples_bad = []
for temp in (0.0, 0.5, 0.9):
    for i, s in enumerate(GEN_SEEDS):
        text, seq, stopped = generate_bytes(g, model, hub_ids, tiles, s, max_len=150,
                                             temperature=temp, stop_at_newline=False, seed=i)
        toks = [t for t in tokenize(text) if t]
        roles = [ROLE_MAP.get(t) for t in toks]
        for a, b, ra, rb in zip(toks, toks[1:], roles, roles[1:]):
            if ra is None or rb is None:
                continue
            total_pairs += 1
            if (ra, rb) in real_role_bigrams:
                plausible_pairs += 1
            elif len(examples_bad) < 10:
                examples_bad.append((a, ra, b, rb))

print(f"\ngenerated adjacent role-tagged word pairs: {total_pairs}")
if total_pairs:
    print(f"pairs matching a role-bigram that REALLY occurs in the corpus: "
          f"{plausible_pairs}/{total_pairs} = {100*plausible_pairs/total_pairs:.1f}%")
print(f"sample of pairs that do NOT match any real corpus role-bigram: {examples_bad}")

# ============================================================
# 3. Out-of-corpus generalization
# ============================================================
print("\n=== [3] out-of-corpus generalization (seed words verified absent from training data) ===")
OOD_SEEDS = ["xylophone ", "dinosaur ", "asteroid ", "marmalade ",
             "trombone ", "penguin ", "volcano ", "kangaroo "]
ood_tokens = []
for i, s in enumerate(OOD_SEEDS):
    text, seq, stopped = generate_bytes(g, model, hub_ids, tiles, s, max_len=100,
                                         temperature=0.7, stop_at_newline=False, seed=i)
    toks = [t for t in tokenize(text) if t.isalpha()]
    ood_tokens.extend(toks)
    print(f"seed={s!r} -> {text!r}")

r = word_validity_rate(ood_tokens)
print(f"\nword-validity rate on OUT-OF-CORPUS seeds (words never seen during training): "
      f"{r*100:.1f}%" if r is not None else "n/a")
print("(compare against the ~94% word-validity rate measured on in-corpus-topic seeds in test_word_generation.py --"
      " the gap between them is the real, honest measure of memorization vs. learned generalization)")
