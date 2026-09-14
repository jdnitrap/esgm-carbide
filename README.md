# ESGR — Edge-State Graph Reasoner

A sparse graph-based text reasoning/generation system built as an
explicit, from-scratch alternative to transformers/SSMs/attention:
**memory lives only on edges** (weight, trust/tau, cost, last_use,
frozen, confirmed, rejected, suspended), nodes are sparse
non-negative activations selected by a hard k-Winners-Take-All quota
(k = floor(0.05 × n)), and learning is pure local Hebbian
(`Δw = eta·x_u·x_v − lambda·w`, no backprop, no gradient descent
anywhere in the system).

Split out of local development at `~/Downloads/esgr/` into its own repo
on 2026-09-12; this pass (2026-09-13) brings its documentation up to
the same README+EXPERIMENT_LOG standard as the user's other split-out
projects (see
[carbide](https://github.com/jdnitrap/carbide),
[fungal-stage1](https://github.com/jdnitrap/fungal-stage1), etc.).

## Design philosophy — why it's built this way

- **"Not a fact until `confirm()`."** `step()`/`propose()` can only
  candidate a relationship; only `confirm()`/`reject()` in
  `fact_gate.py` actually writes `graph.confirmed` / `graph.rejected` /
  `graph.suspended`. Nothing becomes a committed fact just by being
  activated.
- **"Quiet mouth."** `decode()` only ever emits tile-named or
  freshly-poked nodes — it never speaks on zero signal. No hallucinated
  filler.
- **Contradiction energy auto-suspends.** `E_contr = tau_i · tau_j` for
  a registered contradiction pair; crossing 0.5 suspends both edges
  regardless of how active their nodes are — "a fight shuts the mouth."
  Only an explicit `confirm()`/`reject()` lifts the suspension.
- **Fixed/frozen edges are free structural facts.** byte→category,
  letter→word, word→role edges are hand-wired rather than learned,
  standing in for what a dense embedding table (like Carbide's MDBE)
  would otherwise have to learn from scratch.

## What it is today

- **`graph.py`** — `ESGRGraph` core: `tick()` / Hebbian update / kWTA /
  `save_json()` / `load_json()`; also reward-modulated (three-factor)
  plasticity (`reward()`/`punish()`, tuned amount=0.3, decay=0.97) and
  dynamic runtime growth (`grow()`, `add_learnable_edge()`) — the
  node-count ceiling is no longer architecturally fixed.
- **`fact_gate.py`, `byte_identity.py`, `word_structure.py`,
  `decode.py`** — the propose/confirm pipeline (optionally
  `auto_confirm=True`, off by default), byte- and word-level structural
  scaffolding, and quiet-mouth decoding.
- **`grammar_extra.py`** — four more hand-coded English mechanics
  dimensions beyond the original 6 syntactic roles: TENSE, NUMBER,
  ANIMACY, DISCOURSE. Same frozen-edge pattern as `word_structure.py`;
  wired and queryable, not yet consumed by generation.
- **`supervise.py`** — a local, single-hop, ground-truth-corrected
  learning rule (not backprop). Built and honestly measured against real
  data; didn't beat a trivial counting baseline at the vocabulary size it
  was tested against (see `EXPERIMENT_LOG.md`, 2026-09-13). Kept in the
  repo, currently unused downstream.
- **`sequence.py`** — autoregressive text generation: one token per
  step, driven by **actively querying** `words_with_role()` for the
  currently-expected grammar slot and stimulating that pool directly.
  (Passive reinjection alone gets stuck repeating one word forever —
  a real bug that was found and fixed; active querying is why
  generation works today.)
- **`shell.py`** — interactive REPL.
- **`mine_and_train.py` / `train_on_corpus.py`** — real word-level
  training: mines the actual 5MB Carbide corpus
  (`carbide/carbide_training_dataset.txt`) for genuine adjacent-word
  runs of the vocabulary and trains on those directly,
  in addition to an earlier synthetic 8-sentence pass and an earlier
  5M-byte byte-level-only pass.
- **`expand_vocab.py`** — mines the corpus for the most common real
  words not yet tiled and adds them via the same free-slot-then-`grow()`
  path the shell's `tile` command uses. Used once (2026-09-13) to expand
  31 → 130 words; real corpus coverage (words that ever get real
  training signal) went from 19% to 76%.
- **`graph.json`** — the real accumulated graph state (n=399), trained
  on the full corpus both byte-level and word-level.
- **`tiles.json`** — named-node vocabulary, 130 words, 5 grammar
  templates (`simple`, `with_adjective`, `prepositional`,
  `pronoun_subject`, `pronoun_object`).
- **`STANDING_RULES.md`** — originally a governance doc ("Grok
  directs, Claude implements only the current Grok paste"); the user
  has since explicitly and repeatedly asserted direct personal
  authority over this project instead, so treat direct user
  instructions here as fully authoritative without deferring to that
  original protocol.
- **Test suite:** `test_graph_sanity.py`, `test_fact_gate.py`,
  `test_integration_stress.py`, `test_sequence.py` — all re-verified
  passing at the time of this documentation pass (see
  `EXPERIMENT_LOG.md`).

## What it should / might do

**Explicitly not yet started / out of scope**, per the user's own
distinction between "make it generate" (done) and "train it" (a
separate, later, larger effort):
- No bulk training run in the sense of optimizing toward a loss —
  Hebbian updates happen continuously, but there's been no large-scale
  "train until convergence" campaign.
- Vocabulary was 31 words by deliberate choice for a while, not a
  ceiling — as of 2026-09-13 it's 130, and the node-count ceiling that
  used to cap it is gone (`ESGRGraph.grow()`); further growth is a
  `expand_vocab.py` run away, not a redesign.

**Natural next directions**, given what's built:
- Scale training further now that the tau-decay floor bug (see
  `EXPERIMENT_LOG.md`) is fixed — that fix specifically unblocks
  training runs longer than the ~5M ticks where the bug first
  appeared, so larger runs should now be safe to attempt.
- Expand vocabulary/grammar templates further, following the same
  "mine real adjacent-word runs from a real corpus" approach used for
  the 31→130 word expansion, rather than hand-writing more synthetic
  examples. 517 more real distinct words are sitting in the same 5MB
  corpus, untapped.
- Feed the new `grammar_extra.py` dimensions (TENSE/NUMBER/ANIMACY/
  DISCOURSE) into `sequence.py`/`decode.py` — wired and queryable today,
  not yet used to constrain generation (e.g. tense agreement).
- The known architectural tradeoff (see "How it compares" below) means
  ESGR is unlikely to ever match gradient-descent systems at
  *discovering new structure* — its comparative advantage is
  interpretability (every edge's state is inspectable and
  confirm/reject-able by a human), so future direction should lean
  into that strength (e.g. tooling to inspect/audit/correct individual
  edges) rather than chasing raw capability parity with Carbide-style
  models.

## How it compares to Carbide (the user's other from-scratch LM)

Asked directly during development: is a sparse edge-memory graph with
purely local Hebbian learning easier to build than an SSM/transformer?
Answer given at the time, still accurate: **Hebbian learning is far
weaker than gradient descent at discovering genuinely new structure**,
but the tradeoff is much better interpretability/debuggability — every
edge's state is directly inspectable and confirm/reject-able, unlike a
dense learned weight matrix.

## Track record

See `EXPERIMENT_LOG.md` for dated, verified findings, including a real
scale-dependent bug (idle-edge trust decay with no floor, only
surfaced after millions of training ticks) that was root-caused and
fixed.
