"""Persists a trained NextByteRNN's weights, separate from graph.json --
the graph never reads this file, it's the head's own saved state living
alongside the graph it was trained against (see EXPERIMENT_LOG.md,
2026-09-14 "combination" discussion).

Warm-start, not cold restart, when the graph grows a new fixed column
(e.g. a 7th grammar_extra dimension added later): the byte embedding
table, the GRU's hidden-to-hidden weights, and the output layer are all
completely independent of how many tag columns exist -- only the GRU's
INPUT weight matrix changes shape, and only because build_tag_table()
always appends new columns after the existing ones (verified in
head.py: byte columns are always [:BYTE_COLUMNS_DIM], word-mechanics
columns always appended after), so the old columns' learned weights
sit at a known, stable position in the new, wider matrix and can be
copied in directly. Only the brand-new columns' weight slice starts
fresh.
"""
import torch
from head import NextByteRNN, N_COLUMNS


def save_head_checkpoint(model, path, emb_dim, hidden, tag_dim):
    torch.save({
        "state_dict": model.state_dict(),
        "tag_dim": tag_dim,
        "emb_dim": emb_dim,
        "hidden": hidden,
        "use_embedding": model.use_embedding,
        "use_tags": model.use_tags,
    }, path)


def load_head_checkpoint(path, new_tag_dim=None):
    """Returns (model, warm_started: bool, old_tag_dim, new_tag_dim).
    If new_tag_dim is None or matches the checkpoint's own tag_dim,
    loads directly, no warm-start needed. Otherwise builds a fresh
    model at new_tag_dim and copies in everything that's still valid."""
    ckpt = torch.load(path, weights_only=True)
    old_dim = ckpt["tag_dim"]
    target_dim = new_tag_dim if new_tag_dim is not None else old_dim

    model = NextByteRNN(emb_dim=ckpt["emb_dim"], hidden=ckpt["hidden"],
                         use_embedding=ckpt["use_embedding"], use_tags=ckpt["use_tags"])

    if target_dim == old_dim:
        model.load_state_dict(ckpt["state_dict"])
        return model, False, old_dim, target_dim

    old_sd = ckpt["state_dict"]
    new_sd = model.state_dict()
    emb_dim = ckpt["emb_dim"] if ckpt["use_embedding"] else 0

    for key, old_tensor in old_sd.items():
        if key not in new_sd or new_sd[key].shape == old_tensor.shape:
            # Unaffected by tag_dim growing: embedding table, GRU hidden-
            # to-hidden weights/biases, output layer -- copy verbatim.
            new_sd[key] = old_tensor
            continue
        if key == "gru.weight_ih_l0":
            # Shape (3*hidden, emb_dim+tag_dim) -- columns are laid out
            # [embedding part | tag columns], and build_tag_table() only
            # ever APPENDS new tag columns after the old ones, so the
            # old columns occupy the exact same [0 : emb_dim+old_dim)
            # slice in the new, wider matrix. New columns' slice keeps
            # its fresh random init from `model`'s own construction.
            new_sd[key][:, :emb_dim + old_dim] = old_tensor
        # any other shape-mismatched key: leave at fresh init (none
        # expected at this model's size, but fail safe rather than crash)

    model.load_state_dict(new_sd)
    return model, True, old_dim, target_dim
