"""Fine-tunes the already-trained head on real dialogue-shaped data
(dialogue_corpus.txt) so it learns turn-taking structure ("Q: ... /
A: ...") -- explicit user direction, 2026-09-14: "so teach it how to
talk." Warm-started from the existing checkpoint (same head_checkpoint
mechanism already proven mid-session), not trained from scratch --
the goal is to ADD Q&A behavior on top of the general English fluency
already learned from the 5MB essay corpus, not replace it.

2026-09-14, second attempt, per the user's own diagnosis after the
first attempt failed: 51 Q&A pairs x6 repeats x200 epochs just
memorized harder (held-out accuracy flat 84.05%->84.03%) and actively
hurt both turn-stopping (3/5 -> 1/5) and essay fluency (96.3% ->
86.6%). Two real fixes applied together this time:
  1. A real, EXPANDED dialogue_corpus.txt (build_dialogue_corpus.py):
     203 pairs (66 facts x 3 real question phrasings + 5 meta) instead
     of 51x6 repeats of the same content -- real variety instead of
     memorization fuel.
  2. head.py's new TURN column (QUESTION/ANSWER/none), so the head
     gets an explicit, never-learned signal for which part of the
     input it's in, instead of inferring turn boundaries purely from
     noisy byte statistics -- same "hand-given structure" pattern
     already proven (SYNTAX/MORPHOLOGY) to help elsewhere in this repo.
Fewer repeats and a bigger essay-mix slice below, specifically to keep
guarding against the forgetting risk that was real and measured last
time -- not assumed fixed just because the dataset grew.
"""
import json
import shutil
import time
import torch
from graph import ESGRGraph
from retrain_head import retrain_head, CORPUS_PATH, SEQ_LEN
from head import build_tag_table

DIALOGUE_PATH = "dialogue_corpus.txt"
DIALOGUE_REPEATS = 1  # was 2 -- round 3's real corpus is ~100KB (827 pairs), no artificial repetition needed at all now
ESSAY_MIX_BYTES = 400_000  # unchanged -- same fluency anchor that worked last round
EPOCHS = 40  # back to the round-3 baseline -- 150 epochs plateaued cleanly but caused real mode collapse in conversation, more epochs isn't the lever anymore; this run isolates the effect of the new DISCOURSE/discovered-dimension columns instead

with open(DIALOGUE_PATH, "rb") as f:
    dialogue_bytes = f.read()
with open(CORPUS_PATH, "rb") as f:
    essay_bytes = f.read(ESSAY_MIX_BYTES)

combined = (dialogue_bytes * DIALOGUE_REPEATS) + essay_bytes
print(f"fine-tuning corpus: {len(dialogue_bytes)} bytes dialogue x{DIALOGUE_REPEATS} "
      f"+ {len(essay_bytes)} bytes essay slice = {len(combined)} bytes total")

g = ESGRGraph.load_json("graph.json")
frozen_idx = g.frozen.nonzero().flatten().tolist()
w_frozen_before = g.w[frozen_idx].clone()
tau_frozen_before = g.tau[frozen_idx].clone()

# Warm-start FROM the existing fully-trained checkpoint: copy it to the
# dialogue checkpoint path first, so retrain_head() loads real prior
# weights instead of starting fresh (retrain_head warm-starts only when
# checkpoint_path already exists).
shutil.copy("head_checkpoint.pt", "head_checkpoint_dialogue.pt")

t0 = time.time()
stats = retrain_head(g, n_epochs=EPOCHS, sample_bytes=len(combined),
                      corpus_bytes=combined, verbose=True,
                      checkpoint_path="head_checkpoint_dialogue.pt")
elapsed = time.time() - t0

print(f"\n=== dialogue fine-tune done in {elapsed:.1f}s ===")
print(f"held-out accuracy before: {stats['before_acc']:.4f}")
print(f"held-out accuracy after {stats['epochs']} epochs: {stats['after_acc']:.4f}")

w_frozen_after = g.w[frozen_idx]
tau_frozen_after = g.tau[frozen_idx]
moved = int(((w_frozen_after != w_frozen_before) | (tau_frozen_after != tau_frozen_before)).sum().item())
print(f"frozen edges disturbed: {moved} (must be 0)")
print(f"saved new checkpoint: head_checkpoint_dialogue.pt "
      f"(original head_checkpoint.pt left untouched for comparison)")
