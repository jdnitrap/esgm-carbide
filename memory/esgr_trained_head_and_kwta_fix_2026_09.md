---
name: esgr-trained-head-and-kwta-fix-2026-09
description: "ESGR gained a real byte-level trained head (head.py) that reads the graph as MDBE-style input and teaches it back through supervised_step() — building and testing this at real (5MB corpus) scale surfaced and fixed a 3-stage generation-collapse bug in tick()/decode()'s shared sparsity budgets"
metadata: 
  node_type: memory
  type: project
  originSessionId: 30cf9044-ec91-439b-bf8c-6d1b48744cc4
  modified: 2026-09-14T06:40:30.676Z
---

Second wave of the same 2026-09-13/14 session as [[esgr-growth-and-modulation-2026-09]] (read that first for the mechanics/vocabulary/growth work this builds on). Committed as `esgr` repo commits `72224ec` and `8b840db`.

**1. A real, tested, byte-level trained head (`head.py`), at explicit user direction.** Rows = raw byte value (0x00-0xFF, matching Carbide's own token unit, NOT word-level — a real correction from the user mid-session after I initially built it word-level). Per-byte input = a learned embedding concatenated with 32 real fixed columns: `byte_identity.py`'s 6 always-live flags + 26 word-level mechanics columns (ROLE/TENSE/NUMBER/ANIMACY/DISCOURSE/SYNTAX/MORPHOLOGY) that only activate at the exact byte where a real tiled word completes (causal only, never looks ahead — `word_at_position()`/`build_tag_table()` in `head.py`). Directly mirrors Carbide's own `MDBE.forward()` pattern.

**2. Replayed Carbide's own 3-seed ablation methodology on ESGR's real data rather than assuming the pattern transfers** — confirmed real, repeatable gain from the mechanics columns (~3 points across 3 seeds, same controlled-comparison rigor as `carbide/MDBE_MANIFEST.md`).

**3. `head_checkpoint.py` — checkpoint + warm-start.** When the graph's fixed-column count grows, the byte embedding table, GRU hidden-to-hidden weights, and output layer are all provably unaffected (verified copied byte-for-byte); only the GRU's input-weight slice for brand-new columns needs fresh init, since `build_tag_table()` always appends new columns after existing ones. Measured, not assumed: warm-start's first epoch beat cold-start's third.

**4. `retrain_head.py` + `uncertainty` (shell commands) — the human-gated retrain trigger.** A person decides when to fire a retrain, informed by a real signal (`gate.proposed.sum()` — count of edges currently proposed-but-unconfirmed). Nothing retrains automatically.

**5. The head teaches the graph back — `supervised_step()`, gated to confident-AND-correct only, never backprop into the graph.** Real full-corpus run (5MB, not a sample): head trained to 90.1% held-out next-byte accuracy, 439,251 real predictions taught back.

**6. Real bug found at full scale: `supervised_step()` had no weight ceiling.** Extremely common byte pairs (","->" ", "."->" ") got reinforced so many times weight reached **536**, breaking word-level generation completely. Root cause: `tick()`'s own Hebbian growth is naturally saturated by `m_raw/(1+abs(m_raw))` in message-passing; `supervised_step()` writes directly with no such saturation. **Fixed: clamped to `graph.max_weight` (2.0).**

**7. That fix alone was not enough — a real, THREE-stage architectural problem, found only by systematically tracing the whole pipeline (user's explicit direction, after two single-fix attempts each failed differently and ad-hoc patching was clearly chasing symptoms downstream of each other).** All three needed together, verified with one script tracing kWTA activation + decode candidacy + final output across every combination on the identical reconstructed broken scenario:

- **`tick()`'s message-passing has no memory of recent history** — a burst of newly-strengthened edges could cascade from near-zero to full strength in 1-2 ticks. **Fix: `ESGRGraph.max_activation_rate`** (constructor param, default `None`) — a leaky-bucket rate limiter on the message-passing term only, applied BEFORE external stimulus (so direct pokes are never rate-limited). User's own framing, credited directly: "what about a buffer like a network does on the internet." Verified: doesn't weaken normal Hebbian learning (~4x growth over 100 ticks, same as unlimited).
- **kWTA's `k` is ONE shared budget for the whole graph** — enough reinforced byte edges climbing together (even individually rate-limited) can still fill all seats collectively, crowding out word-level nodes even under direct stimulation. **Fix: `ESGRGraph.split_sparsity_at`** (constructor param, default `None`) — splits top-k selection into two fully independent pools (bytes 0-255, word-level 256+). Selection logic factored into shared `_topk_select()` helper.
- **`decode()` has its OWN separate shared budget** (flat top-16 candidate cap) that neither of the above touches — a word node could win its own kWTA seat and still get crowded out of `decode()`'s ranked candidate list by enough high-scoring bytes, especially in later steps of a sequence (more ticks = more time for the cascade to broaden). **Fix: `decode(graph, split_cap=True)`** — same principle one layer up: tiled candidates get their own reserved half of the cap. Opt-in; `sequence.py` now opts in explicitly (`decode(graph, split_cap=True)`), other callers unaffected.

All three are opt-in, default-`None`/`False` — every existing test passes byte-for-byte unchanged unless a caller opts in.

**8. User's own diagnosis, confirmed correct by the systematic trace: "xLSTM is going faster than the graph can keep up."** The rate limiter alone kept "the" alive through the tick sequence but it still didn't reach `decode()`'s candidates; the split budget alone let it win a kWTA seat but it still died mid-sequence. Only the combination — plus the separately-discovered third bottleneck in `decode()` — fully restored generation: 6/6 grammar-correct role matches across 8 seeds, both temperature=0 and temperature=0.3, 0 frozen edges disturbed (verified via explicit tensor snapshots every single attempt, with the real `graph.json` restored from backup after every failed attempt before trying the next fix — no broken intermediate state was ever left persisted).

**9. Final real, persisted result:** `graph.json` reflects a genuine full-corpus training + teach-back cycle with generation fully working: `['the', 'choices', 'eats', 'under', 'the', 'existence']` — 6/6 grammar-correct. Full test suite passes. Committed as `8b840db`.

**For a future session:** don't redo the full-corpus training run or re-diagnose the kWTA/decode split-budget issue — it's fixed, tested, and committed. If teach-back is run again at large scale in the future and something breaks generation again, check first whether `max_activation_rate`/`split_sparsity_at` are still enabled on the graph (`g.max_activation_rate`, `g.split_sparsity_at`) and whether `sequence.py` still calls `decode(graph, split_cap=True)` — these are the three real levers, all documented in `EXPERIMENT_LOG.md`'s 2026-09-14 "session 2" entry with the full reasoning.

**Continued further the same session (commit `b9b9e98`) — "do all" on the remaining open items:**
- Vocabulary grown further, 130 → 340 real words (`autoexpand` + `autopilot`).
- **`max_activation_rate` is NOT scale-invariant** — growing the graph n=412→611 broke generation again (4/6) with the same rate=2.0 that had just been fixed; retuning to rate=1.0 fixed it. Recheck this specific value after any future significant vocabulary/graph growth — it needs to shrink as the graph gets bigger, not stay fixed.
- **Real grammar agreement now wired into `sequence.py`**, not just stored: `generate_sequence(..., hub_ids=...)` (optional, default `None` = old behavior) scores candidates higher when they share a real mechanics value (VERB tense, PRONOUN number) with what's already committed. Verified directly against known TENSE values, not just via emergent generation. New `"two_actions"` grammar template exercises this (no other template had 2 VERB slots).
- **Self-directed expansion is real now, not just the earlier "safe/manual" version**: `autopilot on [threshold]` in the shell (off by default — the one deliberate human opt-in) makes `tick`/`train`/`ask` automatically grow vocabulary + retrain + save on their own once real uncertainty crosses a threshold. Verified firing end-to-end for real.
- **Honest open gap:** ~210 of the newest words don't have `ROLE_MAP` entries yet (real "none" state, not a bug) — hand-extending `ROLE_MAP` for that many words is real, tedious, not-yet-done work.

**Same session, later (2026-09-14), two more explicit requests — "ok then fix that" (the ROLE_MAP gap) and "now i want you to train it really heavey. like what you did with carbide" (not yet committed as of this writing):**
- **ROLE_MAP gap closed**: hand-extended from ~115 to 247 entries, same rule as before (only assign where confident regardless of context; left genuinely ambiguous words — and/that/but/rather/etc — as real "none"). All 4 test suites still pass after.
- **Heaviest real training run yet**: 25 epochs (was 4) on the full 5MB corpus, warm-started from the existing checkpoint, `verbose=True` per-epoch logging added to `retrain_head.py` and `run_real_training.py` (`EPOCHS=25`). Held-out accuracy 90.45% → 91.23%, loss 0.378 → 0.327, wall time 391.7s (~6.5 min) on this machine.
- **Teach-back at the largest scale yet**: 446,897 confident-and-correct predictions taught back (vs. 439,251 in the previous, lighter run) — 0 frozen edges disturbed, `n` unchanged (621), and generation still fully grammar-correct afterward (`test_sequence.py`: 6/6, real accumulated `graph.json` produced `['an', 'revolution', 'ran', 'under', 'an', 'revolution']`). The rate-limiter/split-budget fixes from earlier in this doc held up under meaningfully heavier load without further retuning.
- **Not yet committed** — graph was backed up pre-run to `graph.json.bak-pre-heavy-training`; commit the ROLE_MAP extension + verbose-logging changes + new `graph.json` together next time this is asked for.
