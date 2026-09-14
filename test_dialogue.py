"""Tests the dialogue-fine-tuned head (head_checkpoint_dialogue.pt)
against the original essay-only head (head_checkpoint.pt), on two real
questions:

  1. Did it actually learn the Q:/A: turn structure -- does it now
     produce a relevant-ish answer and stop, instead of drifting?
  2. Did fine-tuning on the small dialogue corpus damage its general
     essay-continuation fluency (the catastrophic-forgetting risk named
     before training)? Reuses the same word-validity dictionary check
     as test_word_generation.py so the before/after numbers are
     directly comparable.
"""
import json
from graph import ESGRGraph
from head_checkpoint import load_head_checkpoint
from head import N_COLUMNS
from generate_bytes import generate_bytes

g = ESGRGraph.load_json("graph.json")
tiles = json.load(open("tiles.json"))
hub_ids = json.load(open("grammar_extra_hubs.json"))

essay_model, _, _, _ = load_head_checkpoint("head_checkpoint.pt", new_tag_dim=N_COLUMNS)
dialogue_model, _, _, _ = load_head_checkpoint("head_checkpoint_dialogue.pt", new_tag_dim=N_COLUMNS)

with open("/usr/share/dict/words") as f:
    dictionary = {w.strip().lower() for w in f if w.strip()}

def word_validity(text):
    import string
    toks = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation).isalpha()]
    if not toks:
        return None
    return sum(1 for t in toks if t in dictionary) / len(toks)

# ============================================================
# 1. Real Q&A behavior -- NOVEL questions, not verbatim in the training set
# ============================================================
print("=== [1] dialogue-fine-tuned head answering NOVEL questions ===")
NOVEL_QUESTIONS = [
    "What is the ocean?",
    "What is a black hole?",
    "What is trust?",
    "What is a memory?",
    "What is friendship?",
]
with open("dialogue_corpus.txt") as f:
    dialogue_raw = f.read()

for q in NOVEL_QUESTIONS:
    prompt = f"Q: {q}\nA:"
    verbatim = f"Q: {q}\n" in dialogue_raw
    text, seq, stopped = generate_bytes(g, dialogue_model, hub_ids, tiles, prompt,
                                         max_len=150, temperature=0.5, stop_at_newline=True, seed=0)
    answer = text[len(prompt):].strip()
    print(f"Q: {q}  (verbatim in training set: {verbatim})")
    print(f"A:{answer}")
    print(f"  stopped_at_newline={stopped}")
    print()

print("=== compare: the ORIGINAL essay-only head on the same prompt ===")
q = "What is the ocean?"
prompt = f"Q: {q}\nA:"
text, seq, stopped = generate_bytes(g, essay_model, hub_ids, tiles, prompt,
                                     max_len=150, temperature=0.5, stop_at_newline=True, seed=0)
print(f"Q: {q}")
print(f"A:{text[len(prompt):].strip()}")
print(f"  stopped_at_newline={stopped}")

# ============================================================
# 2. Regression check: general essay fluency, same seeds as before
# ============================================================
print("\n=== [2] essay-fluency regression check (same seeds/metric as test_word_generation.py) ===")
ESSAY_SEEDS = ["the ", "a ", "science ", "the universe ", "climate ", "love ",
               "music ", "quantum ", "history ", "freedom "]

for label, model in [("essay-only (before)", essay_model), ("dialogue-fine-tuned (after)", dialogue_model)]:
    rates = []
    for i, s in enumerate(ESSAY_SEEDS):
        text, seq, stopped = generate_bytes(g, model, hub_ids, tiles, s, max_len=150,
                                             temperature=0.5, stop_at_newline=False, seed=i)
        r = word_validity(text)
        if r is not None:
            rates.append(r)
    avg = sum(rates) / len(rates)
    print(f"{label}: avg word-validity across {len(rates)} essay seeds = {avg*100:.1f}%")
