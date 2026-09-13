# ESGR / fungal core — standing rules

Authority: Grok directs. Johnathan carries messages. Claude implements
only what the current Grok paste says.

## Constitution

- Mind = graph. Memory = edges (w, tau, c, last_use, frozen, confirmed,
  rejected, suspended).
- Not a fact until confirm(). Timer/propose may only candidate.
- reject persists across save/load.
- Words are tiles. Nodes are not words unless tiles.json says so.
- No transformer, no attention, no SSM, no KV cache, no next-token
  trainer as the mind.
- Do not edit tick() math unless a Grok paste explicitly allows a
  named one-line change.

## Frozen

- tick math (quota 5%, eta, message bound, outgoing scale)
- MDBE byte 0-255 → category 256-261 frozen wires
- Never reject frozen/MDBE edges

## Allowed without asking

- Fix crashes, typos, HELP_TEXT
- Add commands only if the current Grok paste lists them

## Forbidden without a Grok paste

- New architecture, training loops, chat personality
- Growing n, grow/prune topology
- Softmax, learned unembed, dense persistent hidden state
- "While we're here" features
- ARC, MMLU, Carbide merge, extra MDBE logic

## How to work

- Do the paste in order
- Report exact terminal output to Johnathan
- No essay
- If a test fails, stop and report. Do not retune 12 times

## Current bench (do not regress)

- shell.py commands: load save poke tick propose confirm reject
  say status help tiles words ask quit
- quiet mouth: emit tiles or freshly poked nodes only
- suspend shuts mouth
- graph.json reload restores facts/rejects/suspends
