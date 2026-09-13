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
