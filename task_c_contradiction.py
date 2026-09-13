import torch
from graph import ESGRGraph
from fact_gate import FactGate

torch.manual_seed(0)
g = ESGRGraph(n_nodes=256, mean_out_degree=12, active_fraction=0.05, seed=0)
gate = FactGate(g, trust_threshold=0.8, min_consecutive_ticks=5, json_path="task_c_graph.json")

C = None
for node in range(g.n):
    edges = (g.src == node).nonzero().flatten().tolist()
    if len(edges) >= 2:
        C = node
        e_A, e_notA = edges[0], edges[1]
        A, notA = int(g.dst[e_A]), int(g.dst[e_notA])
        break

g.register_contradiction(e_A, e_notA)
gate.confirm(C, A)

print(f"{'tick':>4s} {'E_contr':>8s} {'tau_A':>7s} {'tau_notA':>9s} {'clip_hits':>10s} {'nan':>5s} {'suspended':>10s}")
suspend_tick = None
for t in range(50):
    info = g.tick(external_input=None)
    if t == 5:
        gate.confirm(C, notA)
    if info["newly_suspended"] and suspend_tick is None:
        suspend_tick = t
    print(f"{t:>4d} {info['E_contr']:>8.4f} {g.tau[e_A].item():>7.4f} {g.tau[e_notA].item():>9.4f} "
          f"{info['clip_hits']:>10d} {str(info['nan']):>5s} {str(bool(g.suspended[e_A])):>10s}")

print(f"\nsuspend_tick_index={suspend_tick}")

g.save_json("task_c_graph.json")
