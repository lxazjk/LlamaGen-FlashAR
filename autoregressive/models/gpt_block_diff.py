from dataclasses import dataclass
from typing import Optional, List
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from autoregressive.models.gpt import (
    ModelArgs,
    LabelEmbedder,
    CaptionEmbedder,
    RMSNorm,
    FeedForward,
    TransformerBlock,
    precompute_freqs_cis_2d,
    find_multiple,
)
from autoregressive.utils.block_diffusion import MASK_TOKEN_ID


class TimestepEmbedder(nn.Module):
    """Sinusoidal + MLP timestep embedder (zero-init final layer).

    Accepts either a per-sample ``t: [B]`` (broadcast to all positions) or a
    per-position ``t: [B, S]`` (each block can have its own noise level).
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] or [B, S] float in [0,1] -> [B, 1, dim] or [B, S, dim]"""
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.unsqueeze(-1).float() * freqs  # [B, S, half]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)  # [B, S, dim]


class BlockDiffTransformer(nn.Module):
    """Transformer for block diffusion (absorbing-state discrete diffusion).

    Differences from the FlashAR ``Transformer``:
      * Single output head (no medusa R/B dual heads).
      * Token embedding table expanded by 1 for the ``[MASK]`` token.
      * Timestep conditioning via additive sinusoidal embedding.
      * Accepts an explicit attention mask (block-causal or fully bidirectional).
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num

        if self.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(
                config.num_classes, config.dim, config.class_dropout_prob
            )
        elif self.model_type == "t2i":
            self.cls_embedding = CaptionEmbedder(
                config.caption_dim, config.dim, config.class_dropout_prob
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.tok_embeddings = nn.Embedding(config.vocab_size + 1, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        self.t_embedder = TimestepEmbedder(config.dim)

        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.n_layer)
        ]
        self.layers = nn.ModuleList(
            [TransformerBlock(config, dpr[i]) for i in range(config.n_layer)]
        )

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.grid_size = grid_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size,
            config.dim // config.n_head,
            config.rope_base,
            config.cls_token_num,
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.constant_(self.output.weight, 0)
        nn.init.zeros_(self.t_embedder.mlp[-1].weight)
        nn.init.zeros_(self.t_embedder.mlp[-1].bias)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        is_masked: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            t: [B] (broadcast) or [B, seq_len] (per-position noise level).
            loss_weights: [B, seq_len] per-position ELBO weight (1/t).
                          If None, falls back to uniform weighting.
        """
        cond_emb = self.cls_embedding(cond_idx, train=self.training)[
            :, : self.cls_token_num
        ]
        tok_emb = self.tok_embeddings(idx)
        tok_emb = tok_emb + self.t_embedder(t)

        h = torch.cat([cond_emb, tok_emb], dim=1)
        h = self.tok_dropout(h)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[: h.shape[1]]

        mask = attn_mask.to(h.device)
        if mask.dim() == 2:
            mask = mask[None, None]

        for layer in self.layers:
            h = layer(h, freqs_cis, None, mask)

        h = self.norm(h)
        logits = self.output(h[:, self.cls_token_num :]).float()

        loss = None
        if targets is not None and is_masked is not None:
            flat_logits = logits.reshape(-1, self.vocab_size)
            flat_targets = targets.reshape(-1)
            flat_mask = is_masked.reshape(-1).float()
            ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            if loss_weights is not None:
                w = (loss_weights.reshape(-1) * flat_mask)
            else:
                w = flat_mask
            loss = (ce * w).sum() / flat_mask.sum().clamp_min(1.0)

        return logits, loss

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


#############################################################################
#                             Model Configs                                 #
#############################################################################
def GPT_B_BD(**kwargs):
    return BlockDiffTransformer(
        ModelArgs(n_layer=12, n_head=12, dim=768, medusa_attention_num=0, **kwargs)
    )

def GPT_M_BD(**kwargs):
    return BlockDiffTransformer(
        ModelArgs(n_layer=18, n_head=16, dim=1024, medusa_attention_num=0, **kwargs)
    )

def GPT_L_BD(**kwargs):
    return BlockDiffTransformer(
        ModelArgs(n_layer=24, n_head=16, dim=1024, medusa_attention_num=0, **kwargs)
    )

def GPT_XL_BD(**kwargs):
    return BlockDiffTransformer(
        ModelArgs(n_layer=36, n_head=20, dim=1280, medusa_attention_num=0, **kwargs)
    )

def GPT_XXL_BD(**kwargs):
    return BlockDiffTransformer(
        ModelArgs(n_layer=48, n_head=24, dim=1536, medusa_attention_num=0, **kwargs)
    )

BlockDiff_models = {
    "GPT-B": GPT_B_BD,
    "GPT-M": GPT_M_BD,
    "GPT-L": GPT_L_BD,
    "GPT-XL": GPT_XL_BD,
    "GPT-XXL": GPT_XXL_BD,
}
