"""A small, separate, bolt-on next-BYTE prediction head -- explicitly
NOT part of the graph and NOT backprop into anything ESGR itself is
responsible for. Byte-level, matching Carbide's own token unit exactly
(rows = raw byte value 0x00-0xFF), not word-level -- real user
correction, 2026-09-14: the embedding table's rows are the hexadecimal
byte value, same unit ESGR's own byte_identity.py already uses.

Design mirrors Carbide's MDBE.forward(): concatenate a LEARNED
per-byte embedding with the SAME never-learned, hand-given category
flags byte_identity.py already wires as frozen edges (is_alpha,
is_digit, is_upper, is_punct, is_space, utf8_lead) -- read live from
the graph, not duplicated from the buggy mdbe/ASCII_Linguistics.csv
spreadsheet (see EXPERIMENT_LOG.md: that table's Vowel/Consonant,
Numeral, and Punctuation columns were found to be wrong or scrambled;
byte_identity.py's own 6 columns were checked and are correct).

Carbide validated this concat-embedding-with-fixed-flags pattern with a
real 3-seed ablation (full beats no_constraints beats plain_embedding
on loss -- carbide/MDBE_MANIFEST.md, 2026-09-11); train_head.py replays
the same three-way ablation on ESGR's own real byte data rather than
assuming the result transfers.
"""
import torch
import torch.nn as nn
from byte_identity import CATEGORY_NAMES, CATEGORY_OFFSET, byte_categories
from word_structure import get_role, ROLE_NAMES
from grammar_extra import (get_value, TENSE_NAMES, NUMBER_NAMES, ANIMACY_NAMES,
                            DISCOURSE_NAMES, SYNTAX_NAMES, MORPHOLOGY_NAMES)

N_BYTES = 256
BYTE_COLUMNS_DIM = len(CATEGORY_NAMES)  # is_alpha, is_digit, is_upper, is_punct, is_space, utf8_lead

# The rest of the fixed columns: real word-level language mechanics
# (2026-09-14, at explicit user direction) -- ROLE plus everything
# grammar_extra.py added (TENSE/NUMBER/ANIMACY/DISCOURSE/SYNTAX/
# MORPHOLOGY). A single byte doesn't have a ROLE or TENSE on its own;
# these only turn on for the specific byte where a real tiled word
# completes (see word_at_position()/build_tag_table() below) -- every
# other byte gets the real "none" state across all of them, honestly,
# not a guess.
WORD_DIMS = [
    ("ROLE", ROLE_NAMES, None),
    ("TENSE", TENSE_NAMES, "TENSE"),
    ("NUMBER", NUMBER_NAMES, "NUMBER"),
    ("ANIMACY", ANIMACY_NAMES, "ANIMACY"),
    ("DISCOURSE", DISCOURSE_NAMES, "DISCOURSE"),
    ("SYNTAX", SYNTAX_NAMES, "SYNTAX"),
    ("MORPHOLOGY", MORPHOLOGY_NAMES, "MORPHOLOGY"),
]
WORD_TAG_DIM = sum(len(names) + 1 for _, names, _ in WORD_DIMS)  # +1 = a real "none" bucket per dimension

# 2026-09-14: a TURN column, at explicit user direction ("the graph
# needs more columns for more dimensions") after heavy dialogue
# fine-tuning overfit on too little data. Same "deterministic fact
# given for free, never learned" category as byte_identity.py's
# is_alpha/is_punct/etc -- computed by a pure forward scan for the
# literal "Q:"/"A:" markers (see turn_tags() below), not stored as
# graph edges: unlike ROLE/TENSE, a turn state isn't a property of a
# specific known word or byte value, it's a property of POSITION in
# one particular byte stream, so there's no natural graph node to hang
# it on -- same reasoning as why word_at_position() is a pure function
# over byte_seq rather than a graph lookup.
TURN_NAMES = ["QUESTION", "ANSWER"]
TURN_TAG_DIM = len(TURN_NAMES) + 1  # +1 = "none" bucket, real state outside any Q:/A: pair

# 2026-09-14, same day, second addition: user pushed back that TURN
# alone isn't "all of prompt mechanics" -- correct. This is the other
# well-evidenced piece: WHICH KIND of question is being asked/answered,
# detected from the real opening-word patterns actually used in
# dialogue_corpus.txt (What/Who/Where/When/Why/How/Which = WH-question,
# Can/Do/Does/Is/Are/Will/Would = yes-no question, Tell/Explain/Describe
# = imperative request). Directly targets the specific measured failure
# (test_dialogue.py: answers are grammatically fine but topically
# unrelated to the question -- "what is a black hole" answered with a
# fact about the mind-body connection) by giving the model, for free,
# the ONE signal that connects "what kind of question" to "what kind of
# answer is expected" -- the same role ROLE/TENSE play for grammar.
#
# Deliberately NOT included: sentiment, politeness/register, or a full
# declarative/imperative/exclamatory sentence-type beyond what's above
# -- none of those are exercised by the real data (dialogue_corpus.txt
# has no exclamations, no register variation), so adding them now would
# be speculative, not evidence-based -- the same discipline that held
# for SYNTAX/MORPHOLOGY (validated with a real ablation before being
# trusted) applies here: build what the real data justifies, not
# everything imaginable.
QUESTION_FORM_NAMES = ["WH_WHAT", "WH_WHO", "WH_WHERE", "WH_WHEN", "WH_WHY",
                        "WH_HOW", "WH_WHICH", "YES_NO", "IMPERATIVE"]
QUESTION_FORM_FIRST_WORD = {
    "what": "WH_WHAT", "who": "WH_WHO", "where": "WH_WHERE", "when": "WH_WHEN",
    "why": "WH_WHY", "how": "WH_HOW", "which": "WH_WHICH",
    "can": "YES_NO", "do": "YES_NO", "does": "YES_NO", "is": "YES_NO",
    "are": "YES_NO", "will": "YES_NO", "would": "YES_NO",
    "tell": "IMPERATIVE", "explain": "IMPERATIVE", "describe": "IMPERATIVE",
}
QUESTION_FORM_TAG_DIM = len(QUESTION_FORM_NAMES) + 1  # +1 = "none" bucket

N_COLUMNS = BYTE_COLUMNS_DIM + WORD_TAG_DIM + TURN_TAG_DIM + QUESTION_FORM_TAG_DIM


def byte_columns(graph, byte_val):
    """The real, fixed 6-column vector for one byte, read LIVE from the
    graph's own frozen byte->category edges (not recomputed from
    byte_categories() directly) -- so if a specific edge were ever
    rejected/altered this reflects the graph's actual live state, not
    just the classification rule in isolation. Read-only."""
    vec = torch.zeros(BYTE_COLUMNS_DIM)
    for i in range(BYTE_COLUMNS_DIM):
        e = graph.find_edge(byte_val, CATEGORY_OFFSET + i)
        if e is not None and bool(graph.confirmed[e]):
            vec[i] = 1.0
    return vec


def byte_columns_batch(graph):
    """All 256 rows at once, real graph state -- used to build a full
    (256, 6) table once per epoch instead of 256 separate lookups."""
    return torch.stack([byte_columns(graph, b) for b in range(N_BYTES)])


def word_tag_features(graph, hub_ids, word_node):
    """Real 26-dim word-level mechanics vector for one word node --
    one-hot per dimension, real 'none' bucket where a dimension
    genuinely doesn't apply (e.g. "you" has no NUMBER). Read-only."""
    vec = torch.zeros(WORD_TAG_DIM)
    offset = 0
    for dim_name, names, hub_key in WORD_DIMS:
        value = get_role(graph, word_node) if dim_name == "ROLE" else get_value(graph, hub_ids, hub_key, word_node)
        if value is None:
            vec[offset + len(names)] = 1.0
        else:
            vec[offset + names.index(value)] = 1.0
        offset += len(names) + 1
    return vec


def word_at_position(byte_seq, i, word_to_node, max_word_len):
    """Does the byte sequence, read backward from position i (inclusive),
    spell out a real tiled word with a genuine left boundary (start of
    sequence, or a non-alphabetic byte right before it)? Longest match
    wins. Causal only -- never looks past position i: a real streaming
    system doesn't know yet whether "cat" is about to become "cats" or
    "category", so it commits to its best reading of what's been seen
    so far, same as everything else in this repo does."""
    for length in range(min(i + 1, max_word_len), 0, -1):
        start = i - length + 1
        if start > 0:
            prev = byte_seq[start - 1]
            if (65 <= prev <= 90) or (97 <= prev <= 122):
                continue  # mid-word -- not a real left boundary
        try:
            candidate = bytes(byte_seq[start:i + 1]).decode("ascii").lower()
        except UnicodeDecodeError:
            continue
        node = word_to_node.get(candidate)
        if node is not None:
            return node
    return None


def turn_tags(byte_seq):
    """One state per byte position (0=none, 1=question, 2=answer), by a
    single deterministic left-to-right scan for the literal "Q:" and
    "A:" markers, resetting to "none" at a blank line ("\\n\\n", the
    real separator between pairs in dialogue_corpus.txt). Causal only
    -- each position's state depends only on bytes at or before it, so
    this is exactly as streaming-safe as word_at_position(). Text with
    no Q:/A: markers at all (e.g. the plain essay corpus) stays "none"
    throughout -- the real, honest default, not a guess."""
    text = bytes(byte_seq)
    n = len(text)
    state = 0
    out = [0] * n
    for i in range(n):
        pair = text[i:i + 2]
        if pair == b"Q:":
            state = 1
        elif pair == b"A:":
            state = 2
        elif pair == b"\n\n":
            state = 0
        out[i] = state
    return out


def question_form_tags(byte_seq):
    """One state per byte position: which QUESTION_FORM_NAMES category
    (or None -> the "none" bucket) the CURRENT pair's question opened
    with, detected from the real first word after each "Q:" marker.
    Held through BOTH the question and its answer -- reset only at the
    blank line between pairs, not at "A:" -- so the signal is present
    exactly where it's actually needed: while the answer is being
    generated, not just while the question is being read.

    Causal: depends only on bytes at or before each position. Positions
    inside the still-being-typed first word are the real "none" state
    until that word is complete (a delimiter is seen) -- same
    "commit to the best reading seen so far" rule as word_at_position().
    """
    text = bytes(byte_seq)
    n = len(text)
    out = [None] * n
    state = None
    collecting = False
    word = ""
    for i in range(n):
        b = text[i]
        ch = chr(b) if b < 128 else ""
        if text[i:i + 2] == b"Q:":
            state = None
            collecting = True
            word = ""
        elif text[i:i + 2] == b"\n\n":
            state = None
            collecting = False
            word = ""
        elif collecting:
            if ch.isalpha():
                word += ch.lower()
            elif word:
                state = QUESTION_FORM_FIRST_WORD.get(word)
                collecting = False
        out[i] = state
    return out


def build_tag_table(graph, hub_ids, tiles, byte_seq):
    """Full (len(byte_seq), N_COLUMNS) tag tensor for one raw byte
    sequence: the 6 byte-level columns are always present; the 26
    word-level columns are present only at the exact byte where a real
    tiled word completes, real "none" state everywhere else; the 3
    turn columns (QUESTION/ANSWER/none) and 10 question-form columns
    are present at every position, computed by turn_tags() and
    question_form_tags()."""
    word_to_node = {k.lower(): v for k, v in tiles.items() if k not in ("A", "space", "newline")}
    max_len = max((len(w) for w in word_to_node), default=0)
    byte_tab = byte_columns_batch(graph)

    n = len(byte_seq)
    out = torch.zeros(n, N_COLUMNS)
    out[:, :BYTE_COLUMNS_DIM] = byte_tab[list(byte_seq)]
    word_off = BYTE_COLUMNS_DIM
    none_offsets = []
    off = word_off
    for _, names, _ in WORD_DIMS:
        none_offsets.append(off + len(names))
        off += len(names) + 1
    out[:, none_offsets] = 1.0  # default: real "none" everywhere, overwritten below where a word completes

    for i in range(n):
        word_node = word_at_position(byte_seq, i, word_to_node, max_len)
        if word_node is not None:
            out[i, word_off:word_off + WORD_TAG_DIM] = word_tag_features(graph, hub_ids, word_node)

    turn_off = word_off + WORD_TAG_DIM
    states = turn_tags(byte_seq)
    for i, s in enumerate(states):
        out[i, turn_off + s] = 1.0  # s=0 -> none bucket, s=1 -> QUESTION, s=2 -> ANSWER

    qf_off = turn_off + TURN_TAG_DIM
    qf_none_idx = len(QUESTION_FORM_NAMES)
    qf_states = question_form_tags(byte_seq)
    for i, form in enumerate(qf_states):
        idx = QUESTION_FORM_NAMES.index(form) if form is not None else qf_none_idx
        out[i, qf_off + idx] = 1.0
    return out


class NextByteHead(nn.Module):
    """Single-token bigram classifier over raw bytes -- kept for the
    ablation baseline (see EXPERIMENT_LOG.md, 2026-09-14): with only the
    current byte as input, no architecture can beat a plain 256x256
    count table, because there's no more context to extract from than
    counting already uses. Superseded by NextByteRNN below for anything
    claiming a real advantage over counting."""

    def __init__(self, emb_dim=16, hidden=64, use_embedding=True, use_tags=True):
        super().__init__()
        assert use_embedding or use_tags, "head needs at least one input source"
        self.use_embedding = use_embedding
        self.use_tags = use_tags
        in_dim = (emb_dim if use_embedding else 0) + (N_COLUMNS if use_tags else 0)
        if use_embedding:
            self.embedding = nn.Embedding(N_BYTES, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_BYTES),
        )

    def forward(self, byte_idx, tag_vec):
        parts = []
        if self.use_embedding:
            parts.append(self.embedding(byte_idx))
        if self.use_tags:
            parts.append(tag_vec)
        return self.net(torch.cat(parts, dim=-1))


class NextByteRNN(nn.Module):
    """Real sequence model: a GRU hidden state carries forward several
    bytes of context, the one thing a bigram table (and NextByteHead
    above) structurally cannot do. Per-step input is still the same
    MDBE-style concat (learned embedding + byte_identity.py's real
    fixed columns, re-read from the graph at the START of each
    sequence -- not re-injected mid-sequence the way Carbide's
    Block.constraint_proj does per-layer, since this has only one
    recurrent layer, not a stack; a deeper version would need that).
    GRU chosen over a hand-rolled xLSTM for the same reason NextWordHead
    used a plain LSTM/GRU originally: correctness of a well-tested
    torch.nn implementation over a custom architecture at this scale --
    swap in a real xLSTM later if this scale stops being small."""

    def __init__(self, emb_dim=16, hidden=64, use_embedding=True, use_tags=True):
        super().__init__()
        assert use_embedding or use_tags, "head needs at least one input source"
        self.use_embedding = use_embedding
        self.use_tags = use_tags
        in_dim = (emb_dim if use_embedding else 0) + (N_COLUMNS if use_tags else 0)
        if use_embedding:
            self.embedding = nn.Embedding(N_BYTES, emb_dim)
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, N_BYTES)

    def _step_input(self, byte_seq, tag_seq):
        parts = []
        if self.use_embedding:
            parts.append(self.embedding(byte_seq))
        if self.use_tags:
            parts.append(tag_seq)
        return torch.cat(parts, dim=-1)

    def forward(self, byte_seq, tag_seq, hidden=None):
        """byte_seq, tag_seq: (batch, seq_len[, ...]). Returns logits
        for EVERY position (batch, seq_len, N_BYTES) -- next-byte
        prediction at each step, teacher-forced during training -- plus
        the final hidden state, so a caller can carry it across chunks."""
        x = self._step_input(byte_seq, tag_seq)
        out, hidden = self.gru(x, hidden)
        return self.out(out), hidden
