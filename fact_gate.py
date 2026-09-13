"""Timer proposes only. confirm(src,dst)/reject(src,dst) are the only
writers of confirmed/rejected state, which now lives directly on the
graph (graph.confirmed / graph.rejected) — single source of truth,
persisted via graph.save_json(), not a separate parallel copy.
"""
import torch


class FactGate:
    def __init__(self, graph, trust_threshold=0.8, min_consecutive_ticks=5, json_path="graph.json"):
        self.graph = graph
        self.trust_threshold = trust_threshold
        self.min_consecutive_ticks = min_consecutive_ticks
        self.json_path = json_path
        n_edges = graph.src.shape[0]
        self.consecutive_high_trust = torch.zeros(n_edges, dtype=torch.long)
        self.proposed = torch.zeros(n_edges, dtype=torch.bool)

    def step(self):
        """Timer proposes candidates only — never writes confirmed/rejected."""
        high_trust = self.graph.tau >= self.trust_threshold
        self.consecutive_high_trust = torch.where(
            high_trust, self.consecutive_high_trust + 1, torch.zeros_like(self.consecutive_high_trust)
        )
        newly_proposed_mask = (
            (self.consecutive_high_trust >= self.min_consecutive_ticks)
            & (~self.proposed) & (~self.graph.confirmed) & (~self.graph.rejected)
        )
        new_indices = newly_proposed_mask.nonzero().flatten().tolist()
        self.proposed[newly_proposed_mask] = True
        return new_indices

    def confirm(self, src, dst):
        """Explicit yes. Also the only thing (besides reject) that can
        lift a suspend — a suspended edge stays suspended through pure
        tick() decay forever otherwise."""
        i = self.graph.find_edge(src, dst)
        if i is None:
            raise ValueError(f"no edge ({src},{dst}) in graph")
        self.graph.confirmed[i] = True
        self.graph.rejected[i] = False
        self.graph.suspended[i] = False
        self.graph.tau[i] = 1.0
        self.graph.save_json(self.json_path)
        return i

    def reject(self, src, dst):
        i = self.graph.find_edge(src, dst)
        if i is None:
            raise ValueError(f"no edge ({src},{dst}) in graph")
        self.graph.rejected[i] = True
        self.graph.confirmed[i] = False
        self.graph.suspended[i] = False
        self.graph.tau[i] = 0.0
        self.graph.save_json(self.json_path)
        return i

    def n_confirmed(self):
        return int(self.graph.confirmed.sum())
