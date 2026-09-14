"""Shell only. No natural language parser, no training. Reads one
command per line from stdin, dispatches to existing graph/fact_gate/
decode functions — this file adds no new logic to tick(), propose(),
or decode() themselves, just a thin command dispatcher over them.
"""
import sys
import json
import torch
from graph import ESGRGraph
from fact_gate import FactGate
from byte_identity import wire_byte_identity, CATEGORY_OFFSET, CATEGORY_NAMES
from word_structure import wire_word_structure, ROLE_OFFSET, ROLE_NAMES
from expand_vocab import auto_expand_vocab
from grammar_extra import hub_node_ids
from retrain_head import retrain_head, uncertainty_signal
from decode import decode
from sequence import generate_sequence, DEFAULT_GRAMMAR, GRAMMARS

DEFAULT_PATH = "graph.json"
STIM_VALUE = 3.0

with open("tiles.json") as f:
    TILES = json.load(f)
TILES_REV = {v: k for k, v in TILES.items()}


def resolve(token):
    """name-or-int. Returns None (caller prints "no tile") if a name
    isn't in tiles.json — does not touch tick()/propose()/decode()."""
    try:
        return int(token)
    except ValueError:
        return TILES.get(token)


def print_say(result, poked_since_load):
    """Quiet mouth: keep only bytes that are named tiles OR were poked
    since the last load. Suspend still wins — nothing to filter, an
    empty list stays empty."""
    if result["status"] == "suspend":
        print(result)
        return
    kept = [b for b in result["bytes"] if b in TILES_REV or b in poked_since_load]
    status = result["status"] if kept else "silence"
    filtered = {"bytes": kept, "status": status}
    print(filtered)
    names = []
    for b in kept:
        if b in TILES_REV:
            print(TILES_REV[b])
            names.append(TILES_REV[b])
    if names:
        print(" ".join(names))


def energy_snapshot(graph):
    """Read-only energy calc on CURRENT x/w/tau — no Hebbian step, no
    call to tick(). E_drift=0.0 by definition (drift is a measure of
    CHANGE; no step means no drift). Used only when status has no
    last_info yet, so it never has to print None."""
    e_fire = float((graph.x > 0).sum())
    active = graph.x > 0
    used_proxy = active[graph.src] & active[graph.dst]
    e_wire = float(graph.c[used_proxy].sum())
    e_contr = 0.0
    for (i, j) in graph.contradiction_pairs:
        if bool(graph.confirmed[i]) and bool(graph.confirmed[j]):
            e_contr += float(graph.tau[i] * graph.tau[j])
    return {"E_drift": 0.0, "E_wire": e_wire, "E_fire": e_fire, "E_contr": e_contr}


HELP_TEXT = """load [path]
save [path]
poke <name-or-int> [n_ticks=4]
tick [n=1]
propose <name-or-int>
confirm <name-or-int> <name-or-int>
reject <name-or-int> <name-or-int>
auto [on|off]  (with no arg, prints current state; on = sustained-trust candidates confirm themselves)
autoexpand [n=20]  (mines the corpus for the n most common real words not yet tiled, adds them unattended)
uncertainty  (how many proposed-but-unconfirmed edges exist right now -- a real signal for when retrain_head might be worth running)
retrain_head [epochs=3]  (warm-starts from the saved checkpoint if one exists, trains on real corpus bytes, saves)
say
status
tiles
words
ask <words...>
tile <name>
gen [grammar_name] <prompt words...>  (grammars: simple, with_adjective, prepositional, pronoun_subject, pronoun_object)
train [reps=200]
help
quit"""


def make_default_graph():
    torch.manual_seed(0)
    g = ESGRGraph(n_nodes=300, mean_out_degree=8, active_fraction=0.05, seed=0)
    wire_byte_identity(g)
    wire_word_structure(g)
    return g


def main():
    try:
        graph = ESGRGraph.load_json(DEFAULT_PATH)
    except FileNotFoundError:
        graph = make_default_graph()
    gate = FactGate(graph, json_path=DEFAULT_PATH)
    last_info = None
    poked_since_load = set()

    def report_auto_confirms(new_indices):
        # Silent when auto_confirm is off, so existing manual-confirm
        # workflows see no new output at all. Checks graph.confirmed
        # directly rather than assuming every new proposal got auto-
        # confirmed — a proposal below auto_confirm_min_tau is left for
        # a human, and stays silent here too.
        if not gate.auto_confirm or not new_indices:
            return
        for i in new_indices:
            if not bool(graph.confirmed[i]):
                continue
            s, d = int(graph.src[i]), int(graph.dst[i])
            print(f"auto-confirmed: {TILES_REV.get(s, s)} -> {TILES_REV.get(d, d)}")

    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        cmd, args = parts[0], parts[1:]

        if cmd == "load":
            path = args[0] if args else DEFAULT_PATH
            graph = ESGRGraph.load_json(path)
            gate = FactGate(graph, json_path=path)
            poked_since_load = set()
            print(f"loaded {path}")

        elif cmd == "save":
            path = args[0] if args else DEFAULT_PATH
            graph.save_json(path)
            print(f"saved {path}")

        elif cmd == "poke":
            node = resolve(args[0])
            if node is None:
                print("no tile")
                continue
            n_ticks = int(args[1]) if len(args) > 1 else 4
            poked_since_load.add(node)
            stim = torch.zeros(graph.n)
            stim[node] = STIM_VALUE
            for _ in range(n_ticks):
                last_info = graph.tick(external_input=stim)
                report_auto_confirms(gate.step())
            print_say(decode(graph), poked_since_load)

        elif cmd == "tick":
            n = int(args[0]) if args else 1
            for _ in range(n):
                last_info = graph.tick(external_input=None)
                report_auto_confirms(gate.step())
            print(f"ticked {n}")

        elif cmd == "auto":
            if not args:
                print("on" if gate.auto_confirm else "off")
            elif args[0] == "on":
                gate.auto_confirm = True
                print("auto-confirm on — sustained-trust candidates confirm themselves, no human step")
            elif args[0] == "off":
                gate.auto_confirm = False
                print("auto-confirm off")
            else:
                print("usage: auto <on|off>")

        elif cmd == "autoexpand":
            n = int(args[0]) if args else 20
            stats = auto_expand_vocab(graph, TILES, n_words=n)
            TILES_REV.clear()
            TILES_REV.update({v: k for k, v in TILES.items()})
            shown = ", ".join(stats["mined"][:10]) + ("..." if len(stats["mined"]) > 10 else "")
            print(f"auto-expanded: mined {len(stats['mined'])} words ({shown})")
            print(f"grew graph {stats['grown']} times, wired {stats['letter_edges']} letter "
                  f"+ {stats['role_edges']} role edges -- remember to `save` to persist the graph")

        elif cmd == "uncertainty":
            n = uncertainty_signal(graph, gate)
            print(f"proposed-but-unconfirmed edges: {n}")

        elif cmd == "retrain_head":
            n_epochs = int(args[0]) if args else 3
            print("retraining (this reads the real corpus and trains for real, may take a moment)...")
            stats = retrain_head(graph, n_epochs=n_epochs)
            if stats["had_checkpoint"]:
                grow_note = (f", warm-started {stats['old_dim']}->{stats['new_dim']} cols"
                             if stats["warm_started"] else "")
                print(f"loaded existing checkpoint{grow_note}")
                print(f"held-out accuracy before: {stats['before_acc']:.4f}")
            else:
                print("no existing checkpoint -- started fresh")
            print(f"held-out accuracy after {stats['epochs']} epochs: {stats['after_acc']:.4f}")
            print("checkpoint saved -- graph itself untouched by this (head training never writes to the graph)")

        elif cmd == "propose":
            src = resolve(args[0])
            if src is None:
                print("no tile")
                continue
            print(graph.propose(src))

        elif cmd == "confirm":
            src, dst = resolve(args[0]), resolve(args[1])
            if src is None or dst is None:
                print("no tile")
                continue
            i = gate.confirm(src, dst)
            print(f"confirmed edge {i}")

        elif cmd == "reject":
            src, dst = resolve(args[0]), resolve(args[1])
            if src is None or dst is None:
                print("no tile")
                continue
            existing = graph.find_edge(src, dst)
            if existing is not None and bool(graph.frozen[existing]):
                print("frozen")
                continue
            i = gate.reject(src, dst)
            print(f"rejected edge {i}")

        elif cmd == "say":
            print_say(decode(graph), poked_since_load)

        elif cmd == "status":
            n_facts = int(graph.confirmed.sum())
            n_suspended = int(graph.suspended.sum())
            active_frac = float((graph.x > 0).float().sum() / graph.n)
            e = last_info if last_info is not None else energy_snapshot(graph)
            print(f"n={graph.n} n_facts={n_facts} n_suspended={n_suspended} "
                  f"active_frac={active_frac:.4f} "
                  f"E_drift={e.get('E_drift')} E_wire={e.get('E_wire')} "
                  f"E_fire={e.get('E_fire')} E_contr={e.get('E_contr')}")

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "tile":
            raw = args[0]
            name = raw if raw in ("I", "A") else raw.lower()
            if name in TILES:
                print(f"already {name} {TILES[name]}")
            else:
                # Must exclude the category and role node ranges too, not
                # just existing tiles.json values — real bug found by
                # testing: the old version checked TILES.values() only,
                # so the very next tile call would silently overwrite the
                # NOUN role hub (275), then VERB, ARTICLE, PRONOUN,
                # PREPOSITION, ADJECTIVE in turn, before ever reaching a
                # genuinely free node.
                used = set(TILES.values())
                used.update(range(CATEGORY_OFFSET, CATEGORY_OFFSET + len(CATEGORY_NAMES)))
                used.update(range(ROLE_OFFSET, ROLE_OFFSET + len(ROLE_NAMES)))
                used.update(hub_node_ids())  # grammar_extra hubs -- dynamically allocated, real bug found by testing
                node = 274
                while node in used:
                    node += 1
                if node >= graph.n:
                    # graph is full -- grow it a real node instead of
                    # refusing. Fixes the hard n_nodes ceiling: the graph
                    # can now make itself bigger when it runs out of room.
                    old_n = graph.n
                    node = graph.grow(1)[0]
                    print(f"graph was full (n={old_n}) -- grew to n={graph.n}, new node {node}")
                TILES[name] = node
                TILES_REV[node] = name
                with open("tiles.json", "w") as f:
                    json.dump(TILES, f, indent=2)
                wire_word_structure(graph)  # attach the new word's letters-in/role-out edges
                print(f"tiled {name} {node} -- remember to `save` to persist the grown graph")

        elif cmd == "tiles":
            print(TILES)

        elif cmd == "words":
            word_tiles = {k: v for k, v in TILES.items() if k not in ("A", "space", "newline")}
            print(word_tiles)

        elif cmd == "ask":
            lower_tiles = {k.lower(): v for k, v in TILES.items()}
            known_nodes = []
            for w in [a.lower() for a in args]:
                if w not in lower_tiles:
                    print(f"no tile: {w}")
                    continue
                known_nodes.append(lower_tiles[w])
            if known_nodes:
                stim = torch.zeros(graph.n)
                for node in known_nodes:
                    stim[node] = STIM_VALUE
                    poked_since_load.add(node)
                for _ in range(4):
                    last_info = graph.tick(external_input=stim)
                    report_auto_confirms(gate.step())
                print_say(decode(graph), poked_since_load)
                fired = [name for name, node in TILES.items() if graph.x[node] > 0]
                print("fired:", " ".join(fired))

        elif cmd == "gen":
            grammar_name = None
            prompt_tokens = list(args)
            if prompt_tokens and prompt_tokens[0] in GRAMMARS:
                grammar_name = prompt_tokens.pop(0)
            if not prompt_tokens:
                prompt_tokens = ["the"]
            grammar = GRAMMARS.get(grammar_name, DEFAULT_GRAMMAR)
            lower_tiles = {k.lower(): v for k, v in TILES.items()}
            prompt_nodes = [lower_tiles[w.lower()] for w in prompt_tokens if w.lower() in lower_tiles]
            if not prompt_nodes:
                print("no tile")
                continue
            seq, trace = generate_sequence(graph, prompt_nodes, grammar=grammar, max_len=len(grammar))
            names = [TILES_REV.get(n, n) for n in seq]
            role_matches = sum(1 for t in trace if str(t.get("reason", "")).startswith("role_match"))
            print(" ".join(str(n) for n in names))
            print(f"(grammar={grammar_name or 'prepositional (default)'}, role_matches={role_matches}/{len(trace)})")

        elif cmd == "train":
            reps = int(args[0]) if args else 200
            sentences = [
                ["the", "cat", "sat", "on", "the", "mat"],
                ["the", "dog", "sat", "on", "the", "mat"],
                ["is", "the", "cat", "on", "the", "mat"],
                ["is", "the", "dog", "on", "the", "mat"],
                ["yes", "the", "cat", "is", "on", "the", "mat"],
                ["no", "the", "dog", "is", "not", "on", "the", "mat"],
                ["you", "is", "not", "a", "cat"],
                ["I", "is", "not", "a", "dog"],
            ]
            lower_tiles = {k.lower(): v for k, v in TILES.items()}
            sentence_nodes = [[lower_tiles[w.lower()] for w in s] for s in sentences]
            for r in range(reps):
                words = sentence_nodes[r % len(sentence_nodes)]
                stim = torch.zeros(graph.n)
                for node in words:
                    stim[node] = STIM_VALUE
                    poked_since_load.add(node)
                for _ in range(4):
                    last_info = graph.tick(external_input=stim)
                    report_auto_confirms(gate.step())
            print(f"trained {reps} reps over {len(sentences)} sentences")

        elif cmd == "quit":
            break

        else:
            print("no")


if __name__ == "__main__":
    main()
