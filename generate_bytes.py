"""Autoregressive generation directly from the trained byte-level head
(NextByteRNN) -- explicit user direction, 2026-09-14: "what if the
xlstm produce the generate words and just use the graph for the
relationship... since xlstm regains its own weights there should not
reason why it can not generate the words."

Up to now the head only ever ran teacher-forced (fed the REAL corpus
bytes, scored against the REAL next byte) -- its job was "predict, then
teach the graph back." This flips that: the head's own sampled byte
becomes the next input, same as any autoregressive language model. The
graph is no longer deciding the output -- it's queried live, at every
generated position, for the same real structural facts it always
supplies (word_at_position()/build_tag_table() are already causal-only,
never look ahead, so they're already streaming-safe with zero changes).

NOT part of the graph, NOT backprop into anything ESGR is responsible
for -- same boundary as head.py itself. The graph stays fully
edge-inspectable; only this bolt-on head produces free-form output.
"""
import torch
from head import build_tag_table, word_at_position
from word_structure import ROLE_OFFSET, ROLE_NAMES


def _word_is_vetoed(graph, hub_ids, word_node):
    """True if the brain currently disputes this word's structural
    facts -- an explicit reject(), or a live contradiction-suspend --
    on its role edge or any grammar_extra dimension edge. A word with
    NO structural edge at all (never proposed, e.g. a spelling the
    mouth invented) is NOT vetoed -- silence isn't an objection, only
    an explicit reject or an active fight is. This is the missing join
    named in the README's Architecture section: suspend()/reject()
    used to only change what the NEXT tag-table query would report;
    this makes them also stop a byte that's already been emitted from
    standing as if the brain co-signed it."""
    for role_node in range(ROLE_OFFSET, ROLE_OFFSET + len(ROLE_NAMES)):
        e = graph.find_edge(word_node, role_node)
        if e is not None and (bool(graph.rejected[e]) or bool(graph.suspended[e])):
            return True
    for hubs in hub_ids.values():
        for hub_node in hubs.values():
            e = graph.find_edge(word_node, hub_node)
            if e is not None and (bool(graph.rejected[e]) or bool(graph.suspended[e])):
                return True
    return False


def generate_bytes(graph, model, hub_ids, tiles, seed_text, max_len=200,
                    temperature=0.7, stop_at_newline=True, seed=None):
    """Samples up to max_len further bytes after seed_text, one byte at
    a time, feeding the model's own previous output back in. Recomputes
    the full tag table from the graph every step (simple, correct,
    reuses the exact tested build_tag_table() rather than a hand-rolled
    incremental version) -- cheap at this scale (a few hundred bytes).

    temperature<=0: greedy argmax (deterministic).
    temperature>0: real multinomial sampling over softmax(logits/temp).
    stop_at_newline: stop the first time a real \\n byte is produced.

    After each byte, checks whether it just completed a real tiled word
    the brain currently vetoes (_word_is_vetoed above) -- if so, that
    word is stripped back out of the output and generation stops there,
    same as a hard stop, rather than shipping a word the graph actively
    disputes as if it were endorsed.

    Returns (text: str, byte_list: list[int], stopped_early: bool).
    """
    if seed is not None:
        torch.manual_seed(seed)
    seq = list(seed_text.encode("ascii"))
    model.eval()
    stopped_early = False
    word_to_node = {k.lower(): v for k, v in tiles.items() if k not in ("A", "space", "newline")}
    tiles_rev = {v: k for k, v in tiles.items()}
    max_word_len = max((len(w) for w in word_to_node), default=0)
    with torch.no_grad():
        for _ in range(max_len):
            tag_seq = build_tag_table(graph, hub_ids, tiles, bytes(seq))
            byte_tensor = torch.tensor([seq], dtype=torch.long)
            tag_tensor = tag_seq.unsqueeze(0)
            logits, _ = model(byte_tensor, tag_tensor)
            last_logits = logits[0, -1]
            if temperature <= 0:
                next_byte = int(last_logits.argmax())
            else:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_byte = int(torch.multinomial(probs, 1))
            seq.append(next_byte)
            word_node = word_at_position(seq, len(seq) - 1, word_to_node, max_word_len)
            if word_node is not None and _word_is_vetoed(graph, hub_ids, word_node):
                word_len = len(tiles_rev[word_node])
                seq = seq[:-word_len]
                stopped_early = True
                break
            if stop_at_newline and next_byte == 10:
                stopped_early = True
                break
    text = bytes(seq).decode("ascii", errors="replace")
    return text, seq, stopped_early
