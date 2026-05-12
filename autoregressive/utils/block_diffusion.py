import torch
import math
from typing import Tuple, Optional, Dict, Callable

MASK_TOKEN_ID = 16384


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedules
# ─────────────────────────────────────────────────────────────────────────────

def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine mask-rate schedule.

    Maps diffusion time ``t`` in [0, 1] to a masking probability:
      * t = 0 → 0  (clean data, nothing masked)
      * t = 1 → 1  (fully masked)

    Follows the schedule ``mask_rate = 1 - cos(t * π / 2)`` so that the
    derivative is zero at both endpoints (smooth ramp-up / ramp-down).
    """
    return 1.0 - torch.cos(t * (math.pi / 2))


def linear_schedule(t: torch.Tensor) -> torch.Tensor:
    """Linear mask-rate schedule: ``mask_rate = t``."""
    return t.clone() if t.requires_grad else t.float()


NOISE_SCHEDULES: Dict[str, Callable] = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}


# ─────────────────────────────────────────────────────────────────────────────
# Block partition strategies
# ─────────────────────────────────────────────────────────────────────────────

def get_block_assignments(seq_len: int, grid_size: int, strategy: str) -> torch.LongTensor:
    """Assign every token position to a block index.

    Strategies
    ----------
    mdlm   : 1 block — the entire sequence is one block (pure discrete
              diffusion with fully bidirectional attention).
    row     : ``grid_size`` blocks — each row of the 2-D grid is one block,
              ordered top-to-bottom.
    sub4x4  : ``(grid_size / 4) ** 2`` blocks — each 4×4 spatial sub-block
              is one block, ordered in raster-scan (left→right, top→bottom).

    Parameters
    ----------
    seq_len : int
        Total number of image tokens (e.g. 256 for 16×16).
    grid_size : int
        Side length of the square token grid (e.g. 16).
    strategy : str
        One of ``{"mdlm", "row", "sub4x4"}``.

    Returns
    -------
    torch.LongTensor of shape ``[seq_len]`` with values in ``[0, num_blocks)``.
    """
    assert grid_size * grid_size == seq_len, (
        f"grid_size²={grid_size ** 2} != seq_len={seq_len}"
    )

    if strategy == "mdlm":
        return torch.zeros(seq_len, dtype=torch.long)

    if strategy == "row":
        return torch.arange(seq_len, dtype=torch.long) // grid_size

    if strategy == "sub4x4":
        sub = 4
        assert grid_size % sub == 0, (
            f"grid_size ({grid_size}) must be divisible by sub-block size ({sub})"
        )
        nsub = grid_size // sub
        rows = torch.arange(seq_len) // grid_size
        cols = torch.arange(seq_len) % grid_size
        return (rows // sub * nsub + cols // sub).long()

    raise ValueError(
        f"Unknown block strategy '{strategy}'. Choose from: mdlm, row, sub4x4"
    )


def get_num_blocks(strategy: str, grid_size: int) -> int:
    """Return the total number of blocks for a given strategy."""
    if strategy == "mdlm":
        return 1
    if strategy == "row":
        return grid_size
    if strategy == "sub4x4":
        return (grid_size // 4) ** 2
    raise ValueError(f"Unknown block strategy '{strategy}'")


def get_block_token_indices(
    block_assignments: torch.LongTensor,
    block_idx: int,
) -> torch.LongTensor:
    """Return the token-position indices that belong to *block_idx*.

    Useful during sampling to identify which positions in the sequence
    correspond to the block currently being denoised.
    """
    return (block_assignments == block_idx).nonzero(as_tuple=False).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Block-causal attention mask
# ─────────────────────────────────────────────────────────────────────────────

def build_block_causal_mask(
    seq_len: int,
    block_assignments: torch.LongTensor,
    cls_token_num: int = 1,
) -> torch.BoolTensor:
    """Build a block-causal attention mask.

    Within the same block the mask is **bidirectional** (every token attends to
    every other token in its block).  Across blocks the mask is **causal**: a
    token can attend to all tokens in earlier blocks but not to tokens in later
    blocks.

    Convention
    ----------
    ``True`` = attend, ``False`` = masked.  This matches the bool-mask
    convention of ``torch.nn.functional.scaled_dot_product_attention``.

    The first ``cls_token_num`` positions (class / condition tokens) are
    globally visible: every position can attend to them.  The cls positions
    themselves only attend to other cls positions (they do **not** attend to
    image tokens, consistent with the FlashAR baseline's prefix design).

    Parameters
    ----------
    seq_len : int
        Number of image tokens (not including cls prefix).
    block_assignments : torch.LongTensor, shape ``[seq_len]``
        Per-token block index from :func:`get_block_assignments`.
    cls_token_num : int
        Number of prefix condition tokens (default 1).

    Returns
    -------
    torch.BoolTensor, shape ``[cls_token_num + seq_len, cls_token_num + seq_len]``
    """
    total = cls_token_num + seq_len
    mask = torch.zeros(total, total, dtype=torch.bool)

    # Cls tokens: everyone → cls (broadcast conditioning), cls → cls only.
    mask[:, :cls_token_num] = True
    mask[:cls_token_num, :cls_token_num] = True

    # Image region: position i attends to j iff block[j] <= block[i].
    ba = block_assignments
    img_mask = ba.unsqueeze(0) <= ba.unsqueeze(1)  # [seq, seq]
    mask[cls_token_num:, cls_token_num:] = img_mask

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Noise application (absorbing diffusion)
# ─────────────────────────────────────────────────────────────────────────────

def apply_absorbing_noise(
    tokens: torch.LongTensor,
    mask_rate: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """Replace tokens with *mask_token_id* independently at per-sample rate.

    Each token is independently replaced with the absorbing-state
    ``mask_token_id`` with probability ``mask_rate[b]`` for sample ``b``.

    Parameters
    ----------
    tokens : LongTensor ``[B, seq_len]``
        Clean token ids.
    mask_rate : FloatTensor ``[B]``
        Per-sample mask probability in ``[0, 1]``.
    mask_token_id : int
        Replacement id (default :data:`MASK_TOKEN_ID`).

    Returns
    -------
    noisy_tokens : LongTensor ``[B, seq_len]``
    is_masked : BoolTensor ``[B, seq_len]``
    """
    rand = torch.rand_like(tokens.float())
    is_masked = rand < mask_rate.unsqueeze(1)
    noisy = tokens.clone()
    noisy[is_masked] = mask_token_id
    return noisy, is_masked


# ─────────────────────────────────────────────────────────────────────────────
# Sampling utilities
# ─────────────────────────────────────────────────────────────────────────────

def unmask_schedule(
    total_tokens: int,
    total_steps: int,
    schedule: str = "cosine",
) -> torch.LongTensor:
    """Pre-compute how many tokens to have *unmasked* after each sampling step.

    At step 0 nothing is unmasked; at step ``total_steps`` all tokens are
    unmasked.  The returned tensor ``n[i]`` is the cumulative number of tokens
    that should be revealed after step ``i`` (1-indexed).

    The number to **newly unmask** at step ``i`` is ``n[i] - n[i-1]``.

    Parameters
    ----------
    total_tokens : int
        Size of the block being denoised (e.g. 16 for a row block).
    total_steps : int
        Number of denoising iterations for this block.
    schedule : str
        ``"cosine"`` or ``"linear"``.

    Returns
    -------
    torch.LongTensor of shape ``[total_steps]`` (1-indexed step counts).
    """
    sched_fn = NOISE_SCHEDULES.get(schedule)
    if sched_fn is None:
        raise ValueError(f"Unknown schedule '{schedule}', choose from {list(NOISE_SCHEDULES)}")

    # t goes from 1 → 0 over the sampling steps (denoising direction).
    # At each step s (1-indexed), t_s = 1 - s / total_steps.
    # mask_rate = sched_fn(t_s), so ratio_unmasked = 1 - mask_rate.
    steps = torch.arange(1, total_steps + 1, dtype=torch.float64)
    t = 1.0 - steps / total_steps
    mask_rates = sched_fn(t)
    n_unmasked = ((1.0 - mask_rates) * total_tokens).round().long()
    # Ensure monotonically non-decreasing and ends at total_tokens.
    n_unmasked = torch.clamp(n_unmasked, min=0, max=total_tokens)
    for i in range(1, len(n_unmasked)):
        n_unmasked[i] = max(n_unmasked[i], n_unmasked[i - 1])
    n_unmasked[-1] = total_tokens
    return n_unmasked.long()


def topk_confidence_unmask(
    logits: torch.Tensor,
    noisy_tokens: torch.LongTensor,
    n_unmask: int,
    mask_token_id: int = MASK_TOKEN_ID,
    temperature: float = 1.0,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Select and sample the *n_unmask* most-confident masked positions.

    Only positions that are currently ``mask_token_id`` are candidates.
    Among those, we pick the ``n_unmask`` positions with the highest predicted
    probability (after temperature scaling and sampling).

    Parameters
    ----------
    logits : FloatTensor ``[B, seq_len, vocab_size]``
        Raw model output logits for a single block (or entire sequence).
    noisy_tokens : LongTensor ``[B, seq_len]``
        Current (partially denoised) token ids.
    n_unmask : int
        Number of tokens to unmask at this step.
    mask_token_id : int
        Id used for masked positions.
    temperature : float
        Sampling temperature (1.0 = standard categorical sampling).

    Returns
    -------
    sampled_tokens : LongTensor ``[B, seq_len]``
        Updated tokens with ``n_unmask`` positions newly revealed.
    confidence : FloatTensor ``[B, seq_len]``
        Log-probability of the sampled token at each position (``-inf`` for
        positions that were not masked).
    """
    B, S, V = logits.shape
    is_masked = (noisy_tokens == mask_token_id)  # [B, S]

    # Sample from logits
    scaled = logits / max(temperature, 1e-8)
    probs = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs.view(-1, V), num_samples=1).view(B, S)
    log_probs = torch.log_softmax(scaled, dim=-1)
    sampled_log_prob = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # [B, S]

    # Among masked positions, pick top-n_unmask by confidence.
    sampled_log_prob[~is_masked] = -float("inf")
    if n_unmask <= 0:
        return noisy_tokens.clone(), sampled_log_prob

    n_unmask = min(n_unmask, int(is_masked.sum(dim=-1).min().item()))
    if n_unmask <= 0:
        return noisy_tokens.clone(), sampled_log_prob

    _, topk_idx = sampled_log_prob.topk(n_unmask, dim=-1)  # [B, n_unmask]

    result = noisy_tokens.clone()
    result.scatter_(1, topk_idx, sampled.gather(1, topk_idx))
    return result, sampled_log_prob
