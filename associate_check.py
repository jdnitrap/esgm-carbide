"""Renamed from jepa_check.py. This is pairing/associative recall
(context reconstructs a co-stimulated target via Hebbian-strengthened
edges), not JEPA — JEPA is a learned, gradient-trained predictor over
embeddings; nothing here is gradient-trained. Do not call E_pred=0 at
n=256 a JEPA result.

Graph-native, no separate predictor network, no dense hidden state —
the association mechanism IS the graph's own edges, nothing but
tick() dynamics.

Method:
  1. TRAIN: repeatedly co-stimulate a CONTEXT region and a TARGET region
     together, letting ordinary Hebbian consolidation (Step 1, already
     verified) form context->target associations on the edges between
     them.
  2. RECORD the true target activation pattern under full (unmasked)
     co-stimulation — this is the "ground truth embedding."
  3. TEST: stimulate ONLY the context region (target region masked —
     given no direct external drive at all), and read out whatever
     activation the target region reaches purely through existing
     edges from context. This is the "predicted embedding."
  4. E_pred = MSE(predicted, true) — logged as an energy term, exactly
     matching the spec's "E_pred: predict the next region of the graph
     (representation), not the next token."

No softmax over anything, no token vocabulary, no attention — this
whole file operates purely on node activation vectors.
"""
import torch
from graph import ESGRGraph

torch.manual_seed(0)


def run_jepa_check(n_nodes=1024, context_size=40, target_size=40,
                    train_ticks=300, test_ticks=10, seed=0):
    g = ESGRGraph(n_nodes=n_nodes, mean_out_degree=12, active_fraction=0.05, seed=seed)

    context_nodes = torch.arange(0, context_size)
    target_nodes = torch.arange(context_size, context_size + target_size)

    stimulus_full = torch.zeros(n_nodes)
    stimulus_full[context_nodes] = 2.0
    stimulus_full[target_nodes] = 2.0

    # 1. TRAIN: co-stimulate context + target so Hebbian consolidation
    # can form real context->target edges/associations (Step 1's
    # already-verified attractor-consolidation mechanism, reused here).
    energy_log = []
    for t in range(train_ticks):
        info = g.tick(external_input=stimulus_full)
        energy_log.append(info)

    # 2. RECORD ground truth: target's activation pattern under full
    # (unmasked) co-stimulation, once consolidated.
    info = g.tick(external_input=stimulus_full)
    true_target = g.x[target_nodes].clone()

    # 3. TEST: mask the target — stimulate ONLY context, read out
    # whatever the target region reaches purely through existing edges.
    stimulus_context_only = torch.zeros(n_nodes)
    stimulus_context_only[context_nodes] = 2.0
    predicted_target = None
    for t in range(test_ticks):
        g.tick(external_input=stimulus_context_only)
        predicted_target = g.x[target_nodes].clone()

    e_pred = ((predicted_target - true_target) ** 2).mean().item()
    true_active = (true_target > 0).sum().item()
    pred_active = (predicted_target > 0).sum().item()
    overlap = ((true_target > 0) & (predicted_target > 0)).sum().item()

    return {
        "e_pred": e_pred,
        "true_active_count": true_active,
        "predicted_active_count": pred_active,
        "overlap_count": overlap,
        "overlap_frac_of_true": overlap / true_active if true_active > 0 else 0.0,
    }


if __name__ == "__main__":
    print("=== JEPA-style masked-region prediction check ===\n")
    print("First attempt (n=1024, active budget ~51 nodes graph-wide) produced a")
    print("near-trivial test: only 1 of 40 target nodes was ever active — checked")
    print("honestly rather than reported as a clean win. Re-run with a smaller graph")
    print("so the active budget (5% cap, unchanged) is concentrated enough to be a")
    print("real test: n=256, active budget ~12 nodes, context+target = 80/256 (~31%")
    print("of the graph) instead of 80/1024 (~8%).\n")

    print("Untrained baseline (0 co-stimulation ticks — random topology only):")
    result_untrained = run_jepa_check(n_nodes=256, train_ticks=0)
    for k, v in result_untrained.items():
        print(f"  {k}: {v}")

    print("\nAfter training (300 co-stimulation ticks, Hebbian consolidation):")
    result_trained = run_jepa_check(n_nodes=256, train_ticks=300)
    for k, v in result_trained.items():
        print(f"  {k}: {v}")

    print(f"\nE_pred: untrained={result_untrained['e_pred']:.4f} -> "
          f"trained={result_trained['e_pred']:.4f}")
    print(f"target-overlap fraction: untrained={result_untrained['overlap_frac_of_true']:.1%} -> "
          f"trained={result_trained['overlap_frac_of_true']:.1%}")
    print("\nIf training genuinely helps prediction, E_pred should drop and overlap")
    print("fraction should rise relative to the untrained baseline — checked, not assumed.")
