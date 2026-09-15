"""Timer proposes; confirm(src,dst)/reject(src,dst) are the normal
writers of confirmed/rejected state, which lives directly on the graph
(graph.confirmed / graph.rejected) — single source of truth, persisted
via graph.save_json(), not a separate parallel copy. With auto_confirm=True,
step() itself also writes confirmed state on a newly-proposed edge, with
no human call — off by default, so every existing caller is unaffected.
"""
import torch


class FactGate:
    def __init__(self, graph, trust_threshold=0.8, min_consecutive_ticks=5, json_path="graph.json",
                 auto_confirm=False, auto_confirm_min_tau=0.95):
        self.graph = graph
        self.trust_threshold = trust_threshold
        self.min_consecutive_ticks = min_consecutive_ticks
        self.json_path = json_path
        self.auto_confirm = auto_confirm
        # Deliberately separate from trust_threshold: trust_threshold gates
        # when a candidate is even proposed for a human to look at (0.8 is
        # plenty for that). auto_confirm_min_tau is the higher bar an edge
        # must ALSO clear before the system is allowed to confirm it with
        # no human involved — a candidate that clears trust_threshold but
        # not this floor still gets proposed, just not auto-confirmed, so
        # a person can still review it via confirm()/reject().
        self.auto_confirm_min_tau = auto_confirm_min_tau
        n_edges = graph.src.shape[0]
        self.consecutive_high_trust = torch.zeros(n_edges, dtype=torch.long)
        self.proposed = torch.zeros(n_edges, dtype=torch.bool)

    def resync(self):
        """Call after graph.grow() / add_learnable_edge() / add_fixed_edge()
        append new edges -- this gate's own tensors were sized once at
        __init__ to graph.src.shape[0] and never grow with the graph on
        their own. Found by testing: new edges from growth were silently
        outside confirm/reject coverage, since step() indexes
        consecutive_high_trust/proposed by edge id and a stale, shorter
        tensor either errors or -- worse -- silently misaligns once
        edge ids it was never sized for exist. Append-only, same
        invariant as the graph's own tensors: existing entries never
        move."""
        n_edges = self.graph.src.shape[0]
        n_missing = n_edges - self.consecutive_high_trust.shape[0]
        if n_missing <= 0:
            return
        self.consecutive_high_trust = torch.cat(
            [self.consecutive_high_trust, torch.zeros(n_missing, dtype=torch.long)]
        )
        self.proposed = torch.cat([self.proposed, torch.zeros(n_missing, dtype=torch.bool)])

    def step(self):
        """Timer proposes candidates. If auto_confirm is set, a newly
        proposed candidate whose CURRENT tau also clears
        auto_confirm_min_tau is committed immediately in the same call —
        same write confirm() does, just triggered by sustained trust
        instead of an explicit human call. A candidate proposed but still
        below that floor is left proposed, not confirmed, for a human to
        decide. With auto_confirm off (the default), behavior is
        unchanged: propose only, nothing written."""
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
        if self.auto_confirm and new_indices:
            auto_now = [i for i in new_indices if self.graph.tau[i].item() >= self.auto_confirm_min_tau]
            for i in auto_now:
                self._set_confirmed(i)
            if auto_now:
                self.graph.save_json(self.json_path)
        return new_indices

    def _set_confirmed(self, i):
        self.graph.confirmed[i] = True
        self.graph.rejected[i] = False
        self.graph.suspended[i] = False
        self.graph.tau[i] = 1.0
        # Reward-modulated Hebbian: a confirmed edge's two endpoints get
        # a temporary learning boost, so nearby co-firing that resembles
        # something already confirmed is trusted more than co-firing
        # that's never led anywhere -- the first real signal-vs-noise
        # distinction the update rule has.
        self.graph.reward([int(self.graph.src[i]), int(self.graph.dst[i])])

    def confirm(self, src, dst):
        """Explicit yes. Also the only thing (besides reject) that can
        lift a suspend — a suspended edge stays suspended through pure
        tick() decay forever otherwise."""
        i = self.graph.find_edge(src, dst)
        if i is None:
            raise ValueError(f"no edge ({src},{dst}) in graph")
        self._set_confirmed(i)
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
        self.graph.punish([src, dst])
        self.graph.save_json(self.json_path)
        return i

    def n_confirmed(self):
        return int(self.graph.confirmed.sum())
