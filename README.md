# ESGM-CARBIDE — Edge-State Graph Memory (Carbide-SSM generator variant)

**ESGM (Edge-State Graph Memory)** is a two-component architecture:
an interpretable, from-scratch memory substrate, paired with a
separate, swappable, gradient-trained generator that reads from it.
This repo, **ESGM-CARBIDE**, is a git fork of **ESGM-GRU**
(`jdnitrap/esgr`), intended to pair the identical memory component with
a generator built on Carbide's proven, CPU-optimized SSM patterns
(chunked parallel scan, incremental/cached decoding) instead of a GRU
— so the two can eventually be compared head-to-head on identical
memory behavior.

**Status: fork point only, no Carbide-SSM code yet.** Everything in
this repo right now — including `head.py`'s `NextByteRNN` — is an
unmodified copy of ESGM-GRU at the commit this fork was cut from. The
Generator here is currently still a GRU, not a Carbide-SSM model. This
section will be rewritten to describe the real implementation once
that work happens; until then, do not read this README as a claim that
the swap is done. (This distinction matters: this project already
spent real effort correcting a case where "planned" and "implemented"
got conflated for an xLSTM that was never built — see "Naming" and the
GRU section below. Same discipline applies here.)

**Edge-State Graph Memory** (formerly named "Edge-State Graph
Reasoner" — see "Naming" below for why that changed): a sparse,
from-scratch alternative to a dense embedding table. **Memory lives
only on edges** (weight, trust/tau, cost, last_use, frozen, confirmed,
rejected, suspended), nodes are sparse non-negative activations
selected by a hard k-Winners-Take-All quota (k = floor(0.05 × n)), and
learning is pure local Hebbian (`Δw = eta·x_u·x_v − lambda·w`,
extended with three-factor reward modulation) — no backprop, no
gradient descent anywhere in this component.

**MDBE-Conditioned Autoregressive Generator**: defined by contract,
not by architecture. Any autoregressive sequence model that (a)
accepts a per-step input formed by concatenating a learned
representation with live MDBE-style structural tags read from Edge-
State Graph Memory, and (b) outputs next-byte logits, trained by
ordinary backpropagation entirely external to the memory component,
satisfies this contract. Edge-State Graph Memory is generator-agnostic
by construction — it exposes a live, read-only tag table
(`build_tag_table()` / `word_at_position()`) and has no dependency on
what reads it. This repo's concrete implementation is `head.py`'s
`NextByteRNN`, a single `nn.GRU` layer — **not** an xLSTM (see
"Naming" and `EXPERIMENT_LOG.md` for why GRU was chosen; no xLSTM code
exists anywhere in this repo, only comments noting it as a possible
future swap).

## Naming

This project was originally named "Edge-State Graph **Reasoner**."
That name is retired: what the memory component actually does —
propose/confirm/reject, contradiction-energy suspension — is
consistency-tracking and belief revision (closest real lineage:
Doyle's 1979 Truth Maintenance System), not inference or derivation of
new facts from existing ones. "Reasoner" overclaimed that. It is now
named for what it verifiably is: a **Memory**.

## MDBE — the shared foundation under both components

**Mechanically Defined Byte-Level Embedding (MDBE):** An input
representation matrix for language models where the rows are
permanently mapped to raw character byte values (such as hexadecimal
UTF-8 bytes) and the columns represent explicitly hardcoded,
human-interpretable linguistic and grammatical constraints. The cell
values are fluid, floating-point decimals that quantify the alignment
of each individual character byte with those specific structural
rules. Original to this project and its sibling [carbide](https://github.com/jdnitrap/carbide) — not
derived from external literature.

MDBE is instantiated **twice, in two different forms**, because each
component has a different constitution:
- **In the Generator, nearly unchanged.** `head.py`'s
  `nn.Embedding(256, emb_dim)` concatenated with fixed tag columns
  *is* the MDBE shape — row = byte value, column = dimension, cell =
  the live feature value — dense and differentiable, exactly as the
  definition above describes, floating-point values included.
- **In the Memory, decomposed into graph topology.** `byte_identity.py`
  states directly why: "a dense per-byte vector would violate the
  frozen law ('no hidden dense vector used as memory')." So: **row →
  node** (one per byte value, 0–255), **column → node** (`is_alpha`,
  `is_digit`, etc.), **cell → edge** (`add_fixed_edge(byte, category,
  weight=1.0, trust=1.0)`). A matrix and a bipartite graph with
  weighted edges are the same relation in two representations — this
  is a legitimate re-expression, not an approximation. One thing does
  *not* survive the translation: MDBE's cells are graded, floating-point
  alignment scores; the Memory's edges are binary membership at a fixed
  weight. The Memory's version is a discretized MDBE, not a lossless
  one — only the Generator's version keeps the original graded nature.

## GRU — the inherited Generator implementation (pending replacement)

**Gated Recurrent Unit (GRU)** (Cho et al., 2014): a recurrent neural
network cell that maintains a single hidden state vector and updates
it at every sequence step using two learned gates, rather than a
separate memory cell and three gates the way LSTM does. Given input
`x_t` and previous hidden state `h_{t-1}`:

- **Update gate:** `z_t = sigmoid(W_z·x_t + U_z·h_{t-1})` — how much of
  the old hidden state to keep vs. replace.
- **Reset gate:** `r_t = sigmoid(W_r·x_t + U_r·h_{t-1})` — how much the
  past state should influence the new candidate at all.
- **Candidate state:** `h̃_t = tanh(W_h·x_t + U_h·(r_t ⊙ h_{t-1}))`.
- **New hidden state:** `h_t = (1 − z_t)⊙h_{t-1} + z_t⊙h̃_t`.

**As of this fork's cut point, this repo's Generator (`head.py`'s
`NextByteRNN`) is still this GRU** — inherited unmodified from
ESGM-GRU, not yet replaced. The plan is to swap it for a generator
built on Carbide's `SelectiveSSM` (chunked parallel scan +
incremental/cached decoding, both already proven on this same CPU-only
hardware — see "How it compares" below), because GRU's known
limitation relative to that approach is real: GRU's gates saturate at
±1 (sigmoid/tanh), which degrades gracefully but loses precision over
long sequences, and it offers no algorithmic parallelism advantage the
way a chunked SSM scan does. That said, per this project's own hardware
constraint (CPU-only, no usable GPU), the deciding factor is not
"which is more modern" but which is actually faster and correct on
this hardware — Carbide's chunked scan and incremental decoding are
already measured on this exact hardware; a naive hand-rolled xLSTM
would not be (see "Naming" above for why xLSTM specifically was
rejected). This repo's GRU implementation is one instance of the
MDBE-Conditioned Autoregressive Generator contract, not the contract
itself (see "Architecture" above) — that's precisely what makes
replacing it here possible without touching Edge-State Graph Memory at
all.

## Resources

This architecture combines several well-established ideas from prior
literature; none of the mechanisms below were invented from nothing,
though the specific combination is original to this project.

| Mechanism | Source |
|---|---|
| Hebbian learning (base rule) | Hebb, D.O. (1949), *The Organization of Behavior* |
| Sparse associative memory (overall shape) | Kanerva, P. (1988), *Sparse Distributed Memory*, MIT Press |
| Sparse activation + local synaptic learning | Hawkins, J. & Ahmad, S. (2016), "Why Neurons Have Thousands of Synapses...," *Frontiers in Circuits* |
| k-WTA sparsity quota | Ahmad, S. & Scheinkman, L. (2019), "How Can We Be So Dense?," arXiv |
| Three-factor / reward-modulated plasticity | Frémaux, N. & Gerstner, W. (2016), "Neuromodulated STDP and Theory of Three-Factor Learning Rules," *Frontiers in Neural Circuits* |
| Fact-gating / contradiction suspension | Doyle, J. (1979), "A Truth Maintenance System," *Artificial Intelligence* 12(3) |
| Local-rule vs. backprop ceiling (grounds this project's own honesty about that tradeoff, below) | Lillicrap, T.P. et al. (2020), "Backpropagation and the Brain," *Nature Reviews Neuroscience* |
| Oja's rule (candidate Hebbian variant, not yet used) | Oja, E. (1982), "A Simplified Neuron Model as a Principal Component Analyzer," *J. Mathematical Biology* |
| GRU (this repo's Generator implementation) | Cho, K. et al. (2014), "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," arXiv |
| xLSTM (named alternative, not implemented in this repo) | Beck, M. et al. (2024), "xLSTM: Extended Long Short-Term Memory," arXiv |
| BCM rule (candidate Hebbian variant, not yet used) | Bienenstock, Cooper & Munro (1982), *J. Neuroscience* |
| MDBE | Original to this project (Carbide) — not literature-derived |

Split out of local development at `~/Downloads/esgr/` into its own repo
on 2026-09-12; this pass (2026-09-14) brings its documentation up to
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

## Architecture — two components, not one

ESGM is not a single system; it's two separate pieces bolted together
at a defined boundary, and nothing in this README made that explicit
until this pass, so it's worth stating plainly:

- **Edge-State Graph Memory** (informally, "the brain" — that word is
  a reference, not the name). `graph.py`'s `ESGRGraph`, everything
  under "Design philosophy" above (edge-state memory, confirm/reject,
  contradiction suspension, pure local Hebbian learning) — this is the
  only part of the system that holds memory. It is fully
  edge-inspectable and never trained with backprop.
- **MDBE-Conditioned Autoregressive Generator** (informally, "the
  mouth" or "the head" — again, references, not the name) —
  `head.py` + `generate_bytes.py` in this repo. `head.py` defines
  `NextByteRNN`, a small, separate, gradient-trained next-byte
  prediction model (byte-level, mirroring Carbide's MDBE:
  a learned per-byte embedding concatenated with fixed, never-learned
  category/role/grammar flags read live off the graph). It is
  explicitly **not part of Edge-State Graph Memory and not backprop
  into anything that component is responsible for** — its own file
  header says so directly. `generate_bytes.py` then samples from that
  head autoregressively (its own output fed back in as the next
  input), querying the graph fresh at every generated byte for the
  same causal structural facts (`word_at_position()` /
  `build_tag_table()`).
- **The Memory still supplies the Generator's inputs, live, every
  step** — the tag table the Generator reads from is recomputed from
  current graph state each time, not cached — but **the Memory does
  not yet veto the Generator's output.** `suspend()`/`reject()` change
  what the Memory will report on the next query; they do not currently
  stop `generate_bytes.py` from having already emitted a byte before
  that query happens. Nothing enforces "Memory and Generator must
  agree" today.
- **There are two independent generation paths that currently
  coexist**, and this can read as one system talking with two voices
  if you don't know to expect it: `sequence.py` is the older,
  Memory-only, word-level generator (active-querying `words_with_role()`
  directly, no learned Generator involved at all); `generate_bytes.py`
  is the newer, Generator-driven, byte-level path described above.
  They are not the same code path and do not currently share output
  behavior.

This split exists because it was asked for directly (2026-09-14: "what
if the xlstm produce the generate words and just use the graph for the
relationship") — it is deliberate, not an accident of two unrelated
efforts merging. The Generator side of that request was implemented as
a GRU, not an xLSTM (see "Naming" above); the Memory side is,
by construction, indifferent to which one sits across the boundary —
its only obligation is serving a correct, live tag table, never
knowledge of what architecture consumes it. That's what makes the
ESGM-CARBIDE fork possible without touching this component at all.

## What it is today

**Edge-State Graph Memory:**
- **`graph.py`** — `ESGRGraph` core: `tick()` / Hebbian update / kWTA /
  `save_json()` / `load_json()`; also reward-modulated (three-factor)
  plasticity (`reward()`/`punish()`, tuned amount=0.3, decay=0.97) and
  dynamic runtime growth (`grow()`, `add_learnable_edge()`) — the
  node-count ceiling is no longer architecturally fixed.
- **`fact_gate.py`, `byte_identity.py`, `word_structure.py`,
  `decode.py`** — the propose/confirm pipeline (optionally
  `auto_confirm=True`, off by default), byte- and word-level structural
  scaffolding, and quiet-mouth decoding.
- **`grammar_extra.py`** — six more hand-coded English mechanics
  dimensions beyond the original 6 syntactic roles: TENSE, NUMBER,
  ANIMACY, DISCOURSE, SYNTAX, MORPHOLOGY. Same frozen-edge pattern as
  `word_structure.py`; wired and queryable, and now consumed by
  `head.py` (below) as fixed tag columns — still not consumed by
  `sequence.py`'s own generation logic.
- **`discover_dimension.py`** — lets the graph propose a genuinely new
  category/dimension for itself from real data (distributional
  clustering), instead of only filling in categories a human already
  named. Built; not yet wired into `head.py`'s tag table.
- **`supervise.py`** — a local, single-hop, ground-truth-corrected
  learning rule (not backprop). Built and honestly measured against real
  data; didn't beat a trivial counting baseline at the vocabulary size it
  was tested against (see `EXPERIMENT_LOG.md`, 2026-09-13). Kept in the
  repo, currently unused downstream.
- **`sequence.py`** — the graph-only, word-level generation path: one
  token per step, driven by **actively querying** `words_with_role()`
  for the currently-expected grammar slot and stimulating that pool
  directly. (Passive reinjection alone gets stuck repeating one word
  forever — a real bug that was found and fixed; active querying is why
  this path works at all.) Does not involve `head.py`.
- **`associate_check.py`** — associative-recall check (renamed from
  `jepa_check.py`): context reconstructs a co-stimulated target via
  Hebbian-strengthened edges. Graph-native, no separate predictor
  network — explicitly not JEPA, since JEPA implies a learned,
  gradient-trained predictor.
- **`task_c_contradiction.py`** — standalone script exercising the
  contradiction-energy suspension mechanism directly.

**MDBE-Conditioned Autoregressive Generator (inherited GRU implementation, pending Carbide-SSM replacement):**
- **`head.py`** — `NextByteRNN`: a small, separate, gradient-trained
  next-byte prediction model. Concatenates a learned per-byte embedding
  with fixed, never-learned category/role/grammar flags read live off
  the graph (mirrors Carbide's MDBE pattern). Not part of the graph,
  not backprop into anything Edge-State Graph Memory itself owns.
- **`generate_bytes.py`** — samples from the trained head
  autoregressively (its own output fed back in as the next input),
  querying the graph fresh at every generated byte. See "Architecture"
  above for how this relates to, and currently doesn't defer to, the
  graph's suspend/reject state.
- **`train_head.py` / `train_head_rnn.py`** — train the head on real
  corpus bytes, replaying Carbide's own ablation methodology
  (full vs. embedding-only vs. tags-only) on ESGM's real data rather
  than assuming the Carbide result transfers. Never touch the graph's
  own weights/edges.
- **`head_checkpoint.py`** — persists the trained head's weights
  (`head_checkpoint.pt`), separate from `graph.json`; the graph never
  reads this file. Supports warm-start when the graph grows a new fixed
  column.
- **`retrain_head.py`** — the human-gated retrain trigger (shell's
  `retrain_head` command); never runs on its own.
- **`build_dialogue_corpus.py` / `train_dialogue.py` /
  `test_dialogue.py`** — an attempt to fine-tune the head on
  Q&A-shaped dialogue data. Six fine-tuning rounds all mode-collapsed;
  root-caused to the dialogue corpus being thin, formulaic flashcards
  rather than rich prose like the essay corpus the head otherwise
  trains on (see `EXPERIMENT_LOG.md`, 2026-09-14) — a real, diagnosed
  data-scale ceiling, not a bug.

**Shared / infrastructure:**
- **`shell.py`** — interactive REPL.
- **`mine_and_train.py` / `train_on_corpus.py`** — real word-level
  training: mines the actual 5MB Carbide corpus (from the sibling
  `carbide` repo) for genuine adjacent-word runs of the vocabulary and
  trains on those directly, in addition to an earlier synthetic
  8-sentence pass and an earlier 5M-byte byte-level-only pass.
- **`expand_vocab.py`** — mines the corpus for the most common real
  words not yet tiled and adds them via the same free-slot-then-`grow()`
  path the shell's `tile` command uses. Run repeatedly since
  2026-09-13, most recently taking the vocabulary from 31 words past
  130 to its current size below.
- **`graph.json`** — the real accumulated graph state (currently
  n=649), trained on the full corpus both byte-level and word-level.
- **`tiles.json`** — named-node vocabulary, currently 365 tiles.
- **Test suite:** `test_graph_sanity.py`, `test_fact_gate.py`,
  `test_integration_stress.py`, `test_sequence.py`, `test_dialogue.py`,
  `test_generate_bytes.py`, `test_word_generation.py`,
  `test_word_generation_deep.py` — see `EXPERIMENT_LOG.md` for the
  latest re-verification pass.

`STANDING_RULES.md` (a prior governance doc: "Grok directs, Claude
implements only the current Grok paste") has been removed from the
repo at the user's direction — the project is no longer operating
under that protocol; treat direct user instructions as fully
authoritative.

## What it should / might do

**Explicitly not yet started / out of scope**, per the user's own
distinction between "make it generate" (done) and "train it" (a
separate, later, larger effort):
- No bulk training run in the sense of optimizing toward a loss —
  Hebbian updates happen continuously, but there's been no large-scale
  "train until convergence" campaign.
- Vocabulary was 31 words by deliberate choice for a while, not a
  ceiling — it has since grown past 130 to its current 365 tiles, and
  the node-count ceiling that used to cap it is gone
  (`ESGRGraph.grow()`); further growth is an `expand_vocab.py` run
  away, not a redesign.

**Natural next directions**, given what's built:
- Wire the graph's `suspend()`/`reject()` state into `generate_bytes.py`
  so the mouth can't emit a byte the brain would currently veto — see
  "Architecture" above; this is the one gap where the two organs don't
  yet agree by construction.
- Scale training further now that the tau-decay floor bug (see
  `EXPERIMENT_LOG.md`) is fixed — that fix specifically unblocks
  training runs longer than the ~5M ticks where the bug first
  appeared, so larger runs should now be safe to attempt.
- Expand vocabulary/grammar templates further, following the same
  "mine real adjacent-word runs from a real corpus" approach used for
  the 31→130+ word expansion, rather than hand-writing more synthetic
  examples — re-check how many real distinct corpus words remain
  untapped before picking a target.
- Feed the `grammar_extra.py` dimensions (TENSE/NUMBER/ANIMACY/
  DISCOURSE/SYNTAX/MORPHOLOGY) into `sequence.py`/`decode.py` too —
  already consumed by `head.py`'s tag table, still not used to
  constrain the graph-only generation path.
- If dialogue capability gets picked up again: rewrite
  `dialogue_corpus.txt` with genuinely rich, multi-sentence answers
  instead of more one-line facts in the same thin shape — that lever
  is already exhausted (see `EXPERIMENT_LOG.md`, 2026-09-14).
- The known architectural tradeoff (see "How it compares" below) means
  Edge-State Graph Memory is unlikely to ever match gradient-descent systems at
  *discovering new structure* — its comparative advantage is
  interpretability (every edge's state is inspectable and
  confirm/reject-able by a human), so future direction should lean
  into that strength (e.g. tooling to inspect/audit/correct individual
  edges) rather than chasing raw capability parity with Carbide-style
  models.

## How it compares to Carbide, and to its parent ESGM-GRU

Asked directly during development: is a sparse edge-memory graph with
purely local Hebbian learning easier to build than an SSM/transformer?
Answer given at the time, still accurate: **Hebbian learning is far
weaker than gradient descent at discovering genuinely new structure**,
but the tradeoff is much better interpretability/debuggability — every
edge's state is directly inspectable and confirm/reject-able, unlike a
dense learned weight matrix. That comparison is about [carbide](https://github.com/jdnitrap/carbide)
itself, a separate from-scratch SSM language model — and it's the
project this repo's *planned* Generator borrows its SSM patterns from.

**This repo (ESGM-CARBIDE) is a git fork of ESGM-GRU**
(`jdnitrap/esgr`), sharing Edge-State Graph Memory unchanged. Once the
Generator swap described above actually happens, the two repos will
differ only in Generator implementation — GRU in the parent, a
Carbide-SSM-based generator here — and can be compared head-to-head on
identical memory behavior, without risking the parent's proven
baseline. **Until that swap is implemented, this repo is not yet a
real comparison point** — it's presently identical to its parent.

## Track record

See `EXPERIMENT_LOG.md` for dated, verified findings, including a real
scale-dependent bug (idle-edge trust decay with no floor, only
surfaced after millions of training ticks) that was root-caused and
fixed.
