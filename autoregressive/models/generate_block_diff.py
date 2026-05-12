"""Block Diffusion sampling with proper DDPM reverse process.

Generates one block at a time.  Within each block, runs T denoising steps
using the absorbing-state reverse kernel (not MaskGIT confidence ranking).

For block *b*, previous blocks 1..b-1 are already clean; the current block
starts fully masked and is progressively denoised.
"""
import torch
from torch.nn import functional as F

from autoregressive.utils.block_diffusion import (
    MASK_TOKEN_ID,
    get_block_assignments,
    build_block_causal_mask,
)


def _gumbel_sample(logits):
    """Sample from categorical via Gumbel-max (numerically stable)."""
    noise = torch.rand_like(logits).clamp(min=1e-10)
    gumbel = -(-noise.log()).log()
    return (logits + gumbel).argmax(dim=-1)


@torch.no_grad()
def generate_block_diff(
    model,
    cond,
    max_new_tokens,
    block_strategy="mdlm",
    denoise_steps=16,
    cfg_scale=1.0,
    cfg_interval=-1,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    sample_logits=True,
):
    """DDPM reverse-process generation for block diffusion.

    Interface compatible with ``run_fid_eval``'s *generate_fn*.
    """
    device = cond.device
    B = cond.shape[0]
    seq_len = max_new_tokens
    grid_size = int(seq_len ** 0.5)
    assert grid_size * grid_size == seq_len

    block_assignments = get_block_assignments(
        seq_len, grid_size, block_strategy
    ).to(device)
    attn_mask = build_block_causal_mask(
        seq_len, block_assignments, cls_token_num=model.cls_token_num
    ).to(device)

    num_blocks = int(block_assignments.max().item()) + 1

    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond

    tokens = torch.full(
        (B, seq_len), MASK_TOKEN_ID, dtype=torch.long, device=device
    )

    dt = 1.0 / denoise_steps

    for block_idx in range(num_blocks):
        block_pos = (block_assignments == block_idx)  # [seq_len]

        for step in range(denoise_steps):
            t_val = 1.0 - step * dt  # noise level: 1 → dt
            s_val = t_val - dt        # target noise level after this step

            n_batch = 2 * B if cfg_scale > 1.0 else B
            t_input = torch.full(
                (n_batch, seq_len), s_val, device=device
            )
            t_input[:, block_pos] = t_val

            inp = tokens.clone()
            if cfg_scale > 1.0:
                inp = inp.repeat(2, 1)

            logits, _ = model(inp, cond_combined, t_input, attn_mask)

            if cfg_scale > 1.0:
                cond_logits, uncond_logits = logits.chunk(2, dim=0)
                logits = uncond_logits + cfg_scale * (
                    cond_logits - uncond_logits
                )

            logits_block = logits[:, block_pos, :]  # [B, block_size, V]
            logits_block = logits_block / max(temperature, 1e-5)

            if top_p < 1.0:
                sorted_logits, sorted_idx = logits_block.sort(
                    dim=-1, descending=True
                )
                cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum_probs > top_p
                remove[..., 0] = False
                sorted_logits[remove] = -float("inf")
                logits_block.scatter_(-1, sorted_idx, sorted_logits)

            tokens_block = tokens[:, block_pos]  # [B, block_size]
            is_mask = (tokens_block == MASK_TOKEN_ID)

            sampled = _gumbel_sample(logits_block)

            if step == denoise_steps - 1:
                tokens_block[is_mask] = sampled[is_mask]
            else:
                # DDPM reverse kernel for absorbing diffusion:
                #   P(stay masked) = s / t
                #   P(unmask to x)  = (1 - s/t) * p_x0(x)
                unmask_prob = 1.0 - s_val / t_val
                flip = torch.rand_like(tokens_block.float()) < unmask_prob
                reveal = is_mask & flip
                tokens_block[reveal] = sampled[reveal]

            tokens[:, block_pos] = tokens_block

    tokens = tokens.clamp(0, model.vocab_size - 1)
    return tokens
