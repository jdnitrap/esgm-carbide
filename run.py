"""Item 6 of the frozen minimal checklist: print E each tick. This ties
together everything built so far (graph.py, fact_gate.py) into the
smallest runnable program — nothing more. Not a chatbot, not an
ARC-AGI attempt. A ticking graph that obeys the law.
"""
from graph import ESGRGraph
from fact_gate import FactGate
import torch

g = ESGRGraph(n_nodes=1024, mean_out_degree=12, active_fraction=0.05, seed=0)
gate = FactGate(g, trust_threshold=0.8, min_consecutive_ticks=5, json_path="graph.json")

stimulus = torch.zeros(g.n)
stimulus[:40] = 2.0

for t in range(50):
    info = g.tick(external_input=stimulus)
    hits = gate.step()
    hit_note = f"  +{len(hits)} new fact(s)" if hits else ""
    print(f"tick {t:3d}  E_drift={info['E_drift']:.4f}  E_wire={info['E_wire']:.4f}  "
          f"E_fire={info['E_fire']:.0f}  active={info['active_fraction']:.1%}  "
          f"mean_trust={info['mean_trust']:.3f}{hit_note}")

print(f"\nTotal confirmed facts: {gate.n_confirmed()} (persisted in graph.json)")
