# Experiment / Verification Log

## 2026-09-12 — Generation engine built, vocab/grammar expanded, real
corpus training

- Built the autoregressive generation engine (`sequence.py`). Found and
  fixed a real bug along the way: passive reinjection alone gets stuck
  repeating one word forever; actively querying `words_with_role()` for
  the current grammar slot and stimulating that pool directly is what
  makes generation actually work.
- Vocabulary expanded 13 → 31 words; 5 named grammar templates added
  (`simple`, `with_adjective`, `prepositional`, `pronoun_subject`,
  `pronoun_object`), selectable via `gen <grammar> <prompt>` in the
  shell.
- Real word-level training at scale: `mine_and_train.py` mined the
  actual 5MB Carbide corpus for genuine adjacent-word runs of the
  vocabulary (22,081 found) and trained on those directly — distinct
  from, and additional to, an earlier synthetic 8-sentence pass and an
  earlier byte-level-only pass over the same 5MB corpus (which never
  engaged word tiles at all).
- Full test suite passed at the time: `test_graph_sanity.py`,
  `test_fact_gate.py`, `test_integration_stress.py` (all 4 sub-checks),
  `test_sequence.py` (all 4 checks, including against the expanded
  vocab/grammars on the real trained graph).

## 2026-09-12 — Scale bug found and fixed: idle-edge trust decay with
no floor

**Symptom:** `active_fraction` (should hold steady at 0.05, the kWTA
quota fraction) dipped to ~0.037 under certain narrow single-region
stimulus patterns — but only on the real, massively-trained
`graph.json`. A fresh graph never showed it, meaning short smoke tests
had never caught this.

**Root cause:** edge trust (`tau`) decayed by a flat `-0.001` every
tick an edge went unused, with no floor, trending toward 0. Over the
real 5,000,000-tick byte-level corpus training run, mean tau collapsed
from 0.5 to 0.169 — pushing most non-frozen edges below the
`tau >= 0.3` participation threshold that gates message-passing
entirely. Under a stimulus touching only a handful of directly-relevant
nodes, too few edges remained trusted enough to propagate signal to
fill all k=15 quota slots with genuinely positive values — the kWTA
hard quota then included zero-valued "phantom" slots that don't count
as active, producing the dip. **A genuine scale bug** — works fine in
short smoke tests, breaks specifically at millions of ticks — not
something introduced by the generation-engine changes made the same
day.

**Fix:** idle decay now floors at the trust threshold itself
(`TAU_TRUST_THRESHOLD = 0.3`) instead of 0 —
`torch.clamp(self.tau - 0.001, min=TAU_TRUST_THRESHOLD)`. An edge that
goes quiet settles at "just barely trusted" rather than being fully
pruned; used edges still climb normally via `+0.01`; frozen/
contradiction-pair logic untouched. **Self-healing:** applying the fix
to the already-collapsed `graph.json` snaps affected edges back to 0.3
on their very next idle-decay tick — no need to touch the saved JSON
directly. **Verified:** `active_fraction` now holds exactly 0.0500
across 1500 mixed-stimulus stress-test ticks.

**Standing caution for future large-scale runs:** any future training
run that pushes well past where this one broke (millions of ticks)
should be re-checked for the same class of bug — unconditional
per-tick accumulation/decay with no floor — before assuming
short-test behavior generalizes.

## 2026-09-13 — Documentation pass, fresh test re-verification

Brought this repo's documentation up to the README+EXPERIMENT_LOG
standard used for the user's other split-out projects. Re-ran the full
test suite directly against the real, trained `graph.json` to confirm
nothing has regressed since 2026-09-12:

- `test_graph_sanity.py` — **PASS** (Hebbian consolidation: sustained
  stimulus keeps more originally-stimulated nodes active than brief
  stimulus, as expected)
- `test_fact_gate.py` — **PASS** (confirm/reject commit correctly;
  double-confirm is a safe no-op; no false-positive auto-confirms)
- `test_integration_stress.py` — **PASS** (full pipeline: propose →
  confirm → contradiction → suspend → decode → save/reload; frozen
  structure survives 1500 ticks of unrelated activity with 0 drift;
  full state reload matches exactly)
- `test_sequence.py` — **PASS** (real accumulated `graph.json` follows
  its grammar templates with zero fallbacks; every committed word's
  role independently re-verified against graph structure, not just the
  generation trace)

No code changes made this session — documentation only.

## 2026-09-13 — Reward-modulated learning, dynamic growth, real vocabulary
expansion 31 → 130 words, four bugs found and fixed

**Reward-modulated (three-factor) Hebbian learning.** `ESGRGraph.modulation`
(per-node, neutral=1.0) now scales the Hebbian weight update
(`eta*mod*x_u*x_v - lambda*w`); `graph.reward()`/`graph.punish()`, called
from `FactGate.confirm()`/`reject()`, nudge it. Neutral (mod=1.0, i.e. the
old formula exactly) unless something's actually been confirmed/rejected
nearby — verified byte-identical numbers on the existing sanity test.
Tuned against real data, not guessed: replayed the real corpus-mined
training data and the real graph's own already-trained state; amount=0.5
nearly pinned modulation at its ceiling the moment several real
confirmations landed close together, so tuned down to amount=0.3,
modulation_decay=0.97 (half-life ~23 ticks).

**Dynamic node growth (`ESGRGraph.grow()`) — the 300-node ceiling is no
longer architecturally fixed.** Append-only: new nodes + random-topology
edges, verified 0 frozen edges disturbed by growth. Wired into `tile`.
Also added `add_learnable_edge()` — an ordinary (non-frozen) edge can now
be created at runtime too, not just frozen ones via `add_fixed_edge`.

**`FactGate(auto_confirm=True)`** — an edge that clears sustained trust can
confirm itself with no human call, gated by a stricter
`auto_confirm_min_tau=0.95` floor than the normal 0.8 propose threshold.
Off by default; deliberately overrides "not a fact until confirm()" at
explicit user request.

**`supervise.py` — built, measured honestly, real negative result.** A
local, single-hop, ground-truth-corrected rule (not backprop — pushes
toward the real next word from real text even when it isn't currently
active). Measured against a proper held-out split of real corpus pairs
AND against a trivial "per-context majority vote" baseline: the graph
mechanism did NOT beat plain counting (51.8% vs 54.8% at the time).
Root cause: only 6/31 words ever appeared as a training "context" in real
adjacent-word pairs — severe vocabulary-driven data sparsity, not a flaw
in the rule. Widening the adjacency-gap tolerance (1/3/8/20 tried) made it
worse, not better (plateaued at 11/31 context words, baseline gap widened).
Left in the repo, working and honestly documented, currently unused
downstream.

**Vocabulary expansion 31 → 130 real words (`expand_vocab.py`), which
fixed the actual sparsity problem `supervise.py` couldn't:** mined the top
100 most-common real words in the corpus not already tiled, tiled via the
same free-slot-then-`grow()` path as the shell's `tile` command, extended
`word_structure.ROLE_MAP` with confident-only role assignments (genuinely
ambiguous words left without a role, same "real none" convention as the
original 31). Real coverage: 6/31 (19%) words with real training signal →
99/130 (76%). `graph.json`: 300→399 nodes, 2,667→4,244 edges.

**`grammar_extra.py` — four new hand-coded English mechanics dimensions**
beyond the original 6 syntactic roles, at explicit user direction ("there
is more language mechanics in the English system") after the `supervise.py`
finding: **TENSE** (PAST/PRESENT), **NUMBER** (SINGULAR/PLURAL, "you" left
out — real English ambiguity), **ANIMACY** (ANIMATE/INANIMATE),
**DISCOURSE** (AFFIRM/NEGATE/NEGATOR — finally gives "yes"/"no"/"not" a
real grammatical home). Same frozen-edge mechanism as `word_structure.py`;
uses `grow()` for hub nodes. Wired and queryable, not yet consumed by
`sequence.py`/`decode.py`.

**Four real bugs found and fixed, all verified against the full test
suite after each fix:**

1. **Tile/role-hub collision.** `tile`'s free-node search checked only
   `tiles.json`, not the category/role ranges — the next `tile` call
   would have silently overwritten the NOUN role hub, then the rest.
2. **"space" name collision.** The real English word "space" is also the
   reserved tiles.json key for the space *character* (byte 32) — mining
   logic excluded that key by name from "already have it," so the real
   word got mined as new and silently overwrote the character tile. Fixed
   the exclusion set (now ALL existing keys) and restored the value.
3. **Silent out-of-bounds edges.** `add_fixed_edge()` never validated
   `src`/`dst < graph.n` — a smaller graph given the (now bigger)
   `tiles.json` would silently create an edge to a nonexistent node, only
   crashing later, confusingly, inside `tick()`. Now raises immediately;
   `wire_word_structure()` skips what doesn't fit instead of relying on
   that.
4. **Latent min/max bug in `_enforce_sparsity`'s temperature path,
   exposed (not caused) by the vocabulary growth.**
   `pool_size = min(n, max(k*3, positive_count))` — the function's own
   docstring says the pool should be the generous multiple of k, OR every
   positive value if *fewer* — i.e. `min`, not `max`. It looked correct on
   the old sparse graph (positive_count rarely exceeded k*3, so the two
   operators agreed by coincidence); on the bigger, denser real graph,
   positive_count routinely hit 326/399, ballooning the softmax pool and
   collapsing temperature-based generation to 1/6 role matches across
   10/10 seeds tested. Fixed to `min`; reverified 10/10 seeds back to
   6/6. Worth extra scrutiny on similar `max()`/`min()` code whenever n or
   vocabulary size changes again — this class of bug is invisible at the
   scale it was written and tested against.

Full test suite (`test_graph_sanity.py`, `test_fact_gate.py`,
`test_sequence.py`, `test_integration_stress.py` against the real,
now-expanded `graph.json`) passes after every fix.

## 2026-09-14 — SYNTAX/MORPHOLOGY mechanics, self-expansion, and a real
byte-level trained head that reads and writes back through the graph

**`grammar_extra.py` gained SYNTAX and MORPHOLOGY**, from the merged
`mdbe/language_mechanics_*` worksheets (audited line-by-line first --
zero errors found there, unlike the byte-level `ASCII_Linguistics`
tables, which stay explicitly out of scope: real, verified errors,
built for a transformer/SSM/xLSTM's dense input layer, not ESGR). SYNTAX
is derived FROM `word_structure.ROLE_MAP` (not hand-typed separately --
one word's syntax is a fixed function of its role). Real bug found
wiring this to the live graph: a stale `"space": "NOUN"` entry left over
from the earlier vocabulary-expansion collision fix would have wired the
space CHARACTER into SYNTAX=HEAD; removed, and `wire_grammar_extra()`
now has the same reserved-name guard `wire_word_structure()` already had.

**Self-expansion, the "safe version":** `expand_vocab.auto_expand_vocab()`
is now reusable (not just a one-shot script) and wired into the shell as
`autoexpand [n]` -- frequency alone decides what's added, unattended, no
hand-picked word list. Real bug found on its first real test: 13 of 15
newly mined words collided with the grammar_extra hub nodes, because
those hubs are dynamically grown (no fixed offset like ROLE_OFFSET) and
nothing outside `grammar_extra.py` knew where they live. Fixed with one
shared source of truth, `grammar_extra.hub_node_ids()`, used by both
`autoexpand` and the `tile` command.

**A real, tested byte-level trained head (`head.py`), at explicit user
direction, layered on top of the graph without ever backpropagating into
it:**
- Rows = raw byte value (0x00-0xFF), matching Carbide's own token unit,
  not word-level.
- Per-byte input = a learned embedding concatenated with 32 real, fixed
  columns: the 6 `byte_identity.py` flags (always live) plus the 26
  word-level mechanics columns (ROLE/TENSE/NUMBER/ANIMACY/DISCOURSE/
  SYNTAX/MORPHOLOGY), which only turn on at the exact byte where a real
  tiled word completes -- causal only, never looks ahead.
- Ablation methodology replays Carbide's own (3-seed, `full` vs
  `embedding-only` vs `tags-only`): the mechanics columns gave a real,
  repeatable ~3-point accuracy gain across 3 seeds (35.4%/34.5%/35.1% ->
  38.4%/37.8%/38.5%), same controlled-comparison discipline as
  `carbide/MDBE_MANIFEST.md`.
- Single-token (bigram) head could never beat a plain count table --
  expected: one byte of context can't out-predict counting on the same
  one byte. `NextByteRNN` (a real GRU) was built specifically to test
  whether more context helps, and it does: beats the weak order-1
  baseline easily, and after warm-starting (see below), beat the much
  harder order-2 (previous-2-bytes) baseline for the first time --
  43.67% vs 42.69%.
- **Checkpointing with warm-start** (`head_checkpoint.py`): when the
  graph grows a new fixed column, the byte embedding table, GRU
  hidden-to-hidden weights, and output layer are provably unaffected by
  that (verified: copied byte-for-byte); only the GRU's input weight
  slice for the brand-new columns needs fresh init, since
  `build_tag_table()` always appends new columns after existing ones.
  Measured, not assumed: warm-start's first epoch (38.65%) already beat
  cold-start's third (33.81%).
- **The head teaches the graph back** (`retrain_head.py`,
  `supervise.py`'s existing `supervised_step()`, never a new backprop
  path): only on predictions that are both confident AND actually
  correct against real data -- 8,734 real edges reinforced this way in
  one test run, 0 frozen edges touched, graph node count unchanged.
- **`retrain_head` (shell command) + `uncertainty`**: a person decides
  when to fire a retrain, informed by a real, checkable signal (count of
  currently proposed-but-unconfirmed edges) -- nothing retrains
  automatically.

All of the above verified never touches `graph.w`/`tau`/`confirmed`/
`modulation`/`n`/edge count -- checked explicitly with before/after
tensor snapshots, not just by code inspection. Full existing test suite
still passes throughout.
