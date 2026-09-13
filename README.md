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
  `save_json()` / `load_json()`.
- **`fact_gate.py`, `byte_identity.py`, `word_structure.py`,
  `decode.py`** — the propose/confirm pipeline, byte- and word-level
  structural scaffolding, and quiet-mouth decoding.
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
  runs of the vocabulary (22,081 found) and trains on those directly,
  in addition to an earlier synthetic 8-sentence pass and an earlier
  5M-byte byte-level-only pass.
- **`graph.json`** — the real accumulated graph state (n=300), trained
  on the full corpus both byte-level and word-level.
- **`tiles.json`** — named-node vocabulary, 31 words, 5 grammar
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
- Vocabulary is still small (31 words) by deliberate choice, not a
  ceiling — hand-curated expansion was chosen over automatic
  large-scale growth so far.

**Natural next directions**, given what's built:
- Scale training further now that the tau-decay floor bug (see
  `EXPERIMENT_LOG.md`) is fixed — that fix specifically unblocks
  training runs longer than the ~5M ticks where the bug first
  appeared, so larger runs should now be safe to attempt.
- Expand vocabulary/grammar templates further, following the same
  "mine real adjacent-word runs from a real corpus" approach that
  worked for the current 31-word vocabulary rather than hand-writing
  more synthetic examples.
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
