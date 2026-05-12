import torch
import math

#################################################################################
#                               Mask Scheduling                                #
#################################################################################
def build_masks(model, batch_size, device, seed: int = 0):
    max_seq_length = model.cls_token_num + model.block_size
    causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device))
    proximity_mask = causal_mask.clone()
    model.setup_proximity_mask(
        proximity_mask[-model.block_size:, -model.block_size:],
        model.block_size,
    )
    union_mask = causal_mask | proximity_mask
    # Edges to ADD when annealing causal -> union (proximity may include non-causal edges)
    delta = proximity_mask & ~causal_mask
    delta_indices = [torch.nonzero(delta[q], as_tuple=False).view(-1) for q in range(max_seq_length)]
    # Edges to REMOVE when annealing union -> proximity (causal edges not in proximity)
    removal = causal_mask & ~proximity_mask
    removal_indices = [torch.nonzero(removal[q], as_tuple=False).view(-1) for q in range(max_seq_length)]

    # Shuffle edge order once to make annealing smoother.
    # NOTE: do NOT use the global RNG state (keeps data RNG stable).
    if seed is None:
        seed = 0
    try:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
    except Exception:
        g = None
    for q in range(max_seq_length):
        idx = delta_indices[q]
        if idx.numel() > 1:
            perm = torch.randperm(idx.numel(), generator=g, device=idx.device)
            delta_indices[q] = idx[perm]
        ridx = removal_indices[q]
        if ridx.numel() > 1:
            perm = torch.randperm(ridx.numel(), generator=g, device=ridx.device)
            removal_indices[q] = ridx[perm]
    causal_mask = causal_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    proximity_mask = proximity_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    union_mask = union_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return causal_mask, proximity_mask, union_mask, delta_indices, removal_indices


def pick_mask(
    mask_causal,
    mask_proximity,
    mask_union,
    schedule,
    step,
    anneal_steps,
    rng,
    removal_indices=None,
):
    if schedule == "static_proximity":
        return mask_proximity, 1.0, None
    if schedule == "shrink":
        if anneal_steps <= 0:
            return mask_proximity, 1.0, None
        p = min(1.0, step / anneal_steps)
        mask = mask_union.clone()
        if removal_indices is not None:
            for q, idx in enumerate(removal_indices):
                if idx.numel() == 0:
                    continue
                k = int(math.ceil(p * idx.numel()))
                if k <= 0:
                    continue
                mask[:, q, idx[:k]] = False
        return mask, p, None
    if schedule == "curriculum":
        # Phase 1: causal mask (leverage AR init), Phase 2: proximity mask
        if anneal_steps <= 0 or step >= anneal_steps:
            return mask_proximity, 1.0, None
        return mask_causal, 0.0, None
    raise ValueError(f"Unsupported mask schedule: {schedule}. Use 'static_proximity', 'shrink', or 'curriculum'.")
