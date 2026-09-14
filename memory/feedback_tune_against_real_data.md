---
name: feedback-tune-against-real-data
description: "User wants new ML/graph hyperparameters validated against real training data before being presented as defaults, not shipped as reasonable-sounding guesses"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 30cf9044-ec91-439b-bf8c-6d1b48744cc4
  modified: 2026-09-14T01:53:26.108Z
---

When adding a new tunable numeric parameter to a learning system (e.g. ESGR's reward-modulated Hebbian amount/decay), don't ship a reasonable-sounding guessed default and call it done — actually replay real training data (the real corpus-mined runs, the real accumulated `graph.json` state, whatever the project's real data is) under a few candidate settings and pick values backed by that evidence, with the reasoning shown.

**Why:** explicit instruction after I added reward/punish amounts (0.5) and a decay rate (0.98) to [[esgr-growth-and-modulation-2026-09]] as untested guesses — user said "tune the reward/punish amounts based on real training data." Doing so surfaced a real problem the guess hadn't: 0.5 nearly saturated the modulation ceiling the instant several real confirmations landed close together (14 edges in the live graph were already primed above the confirm threshold). Tuned to 0.3/0.97 instead, backed by that real replay, not by intuition.

**How to apply:** for this user's from-scratch ML projects (ESGR, Carbide, and similar), when introducing a new hyperparameter, budget time to run it against real project data before presenting a default — treat "what's a sane number" as an empirical question to answer proactively, not something to wait to be asked to check.
