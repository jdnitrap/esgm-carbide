"""The user's MDBE idea (from the Carbide project), translated into
ESGR's sparse-node/fixed-wire form instead of a dense embedding vector
— a dense per-byte vector would violate the frozen law ("no hidden
dense vector used as memory"), but the underlying IDEA — give the
system deterministic structural facts for free, never learned, never
forgotten, instead of making it re-derive them from data — maps
directly onto ESGR's own stated topology design: "Fixed 'wires' plus
grow/prune."

Node layout (first 262 node indices are reserved, deterministic, not
random topology):
  0-255   one node per raw byte value (0x00-0xFF) — byte-identity nodes
  256     is_alpha
  257     is_digit
  258     is_upper
  259     is_punct
  260     is_space
  261     utf8_lead

Each byte node gets a FIXED, frozen edge (weight=1.0, trust=1.0, never
touched by Hebbian update) to every category node it belongs to — same
classification rules as Carbide's mdbe_constraints(), reused verbatim
so the two systems agree on what "is_alpha" etc. mean for a given byte.
"""
from graph import ESGRGraph

N_BYTE_NODES = 256
CATEGORY_NAMES = ["is_alpha", "is_digit", "is_upper", "is_punct", "is_space", "utf8_lead"]
CATEGORY_OFFSET = N_BYTE_NODES  # category nodes are 256..261
N_RESERVED = N_BYTE_NODES + len(CATEGORY_NAMES)  # 262


def byte_categories(b: int):
    """Identical classification rules to Carbide's mdbe_constraints(),
    for one byte value — reused so both systems agree."""
    cats = []
    if (65 <= b <= 90) or (97 <= b <= 122):
        cats.append(0)  # is_alpha
    if 48 <= b <= 57:
        cats.append(1)  # is_digit
    if 65 <= b <= 90:
        cats.append(2)  # is_upper
    if (33 <= b <= 47) or (58 <= b <= 64) or (91 <= b <= 96) or (123 <= b <= 126):
        cats.append(3)  # is_punct
    if b in (32, 9, 10, 13):
        cats.append(4)  # is_space
    if (0 <= b < 0x80) or (0xC0 <= b <= 0xFF):
        cats.append(5)  # utf8_lead
    return cats


def wire_byte_identity(graph: ESGRGraph):
    """Adds the fixed byte-identity -> category wiring to an existing
    graph. Requires graph.n >= N_RESERVED (262) so those node indices
    exist. Returns the number of fixed edges added."""
    assert graph.n >= N_RESERVED, f"graph needs >= {N_RESERVED} nodes for byte-identity wiring"
    count = 0
    for b in range(N_BYTE_NODES):
        for cat in byte_categories(b):
            graph.add_fixed_edge(b, CATEGORY_OFFSET + cat, weight=1.0, trust=1.0)
            count += 1
    return count
