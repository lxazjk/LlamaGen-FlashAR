# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
import contextlib
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
import math


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    medusa_attention_num: int = 1
    # Backbone layer index where the vertical branch starts.
    # <0 keeps legacy behavior: branch after the final backbone layer.
    vertical_start_layer: int = -1

    # Right(head)/Below(head) mixing
    # If enabled, learn a mixing coefficient alpha in (0,1):
    #   logits = alpha * logitsR + (1-alpha) * logitsB
    # This replaces the fixed 0.5/0.5 average.
    hv_mix: bool = False
    hv_mix_init: float = 0.5

    # Per-position learnable gate: MLP(proj(cat(logitsR, logitsB))) -> sigmoid.
    # Gate input is the SAME logit pair being mixed -> train/inference consistent.
    hv_gate: bool = False
    hv_gate_proj_dim: int = 0  # 0 = auto (max(32, vocab_size // 512))


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        if self.k_cache.dtype != k_val.dtype or self.k_cache.device != k_val.device:
            self.k_cache = self.k_cache.to(device=k_val.device, dtype=k_val.dtype)
        if self.v_cache.dtype != v_val.dtype or self.v_cache.device != v_val.device:
            self.v_cache = self.v_cache.to(device=v_val.device, dtype=v_val.dtype)
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        # NOTE: during training we call the model with input_pos=None.
        # If a previous sampling/eval call initialized KV cache, keep training safe by
        # ignoring KV cache unless input_pos is explicitly provided.
        if self.kv_cache is not None and input_pos is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.medusa_attention_num = config.medusa_attention_num
        requested_vertical_start = int(getattr(config, "vertical_start_layer", -1))
        if requested_vertical_start < 0:
            requested_vertical_start = self.n_layer
        self.vertical_start_layer = max(0, min(self.n_layer, requested_vertical_start))

        # Learnable mixing weight between right (horizontal) and below (vertical) logits.
        # Stored as logit to keep alpha in (0,1).
        self.hv_mix_logit = None
        if getattr(config, "hv_mix", False):
            p = float(getattr(config, "hv_mix_init", 0.5))
            p = min(max(p, 1e-4), 1.0 - 1e-4)
            init_logit = math.log(p / (1.0 - p))
            self.hv_mix_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        # Per-position gate MLP: input is cat(logitsR, logitsB) for the SAME target position.
        # This ensures train/inference consistency (no position mismatch).
        self.hv_gate_mlp = None
        if getattr(config, "hv_gate", False):
            _vocab = config.vocab_size
            _proj = int(getattr(config, "hv_gate_proj_dim", 0)) or max(32, _vocab // 512)
            self.hv_gate_mlp = nn.Sequential(
                nn.Linear(2 * _vocab, _proj, bias=False),
                nn.SiLU(),
                nn.Linear(_proj, 1, bias=True),
            )
            # Init last layer to 0 so initial gate = sigmoid(0) = 0.5
            nn.init.zeros_(self.hv_gate_mlp[-1].bias)
            nn.init.zeros_(self.hv_gate_mlp[-1].weight)

        # Optional training-time schedule control (set by training loop).
        # When set, the effective right-head weight is:
        #   w = blend * target + (1-blend) * sigmoid(hv_mix_logit)
        # so you can start with a desired bias (e.g., vertical head dominates) and
        # gradually hand over to the learnable weight.
        self.hv_mix_target = None
        self.hv_mix_blend = None

        # Split loss: compute separate CE for each head instead of on mixed logits.
        self.split_loss = False
        self.split_loss_lambda = 0.5   # weight for R head loss; B gets (1 - lambda)
        self.col0_boost = 0.0          # extra weight for B head loss on column-0 tokens

        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer + config.medusa_attention_num)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer + config.medusa_attention_num):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # medusa output layer
        self.medusa_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.medusa_output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.grid_size = grid_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)
        if self.hv_gate_mlp is not None:
            gate_std = 1.0 / math.sqrt(2.0 * float(self.vocab_size))
            nn.init.normal_(self.hv_gate_mlp[0].weight, mean=0.0, std=gate_std)
            nn.init.zeros_(self.hv_gate_mlp[-1].bias)
            nn.init.zeros_(self.hv_gate_mlp[-1].weight)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    @staticmethod
    def _binary_gate_entropy(prob: torch.Tensor) -> torch.Tensor:
        prob = prob.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
        return entropy / math.log(2.0)

    def _hv_gate(self, logits_h: torch.Tensor, logits_v: torch.Tensor) -> torch.Tensor:
        features = torch.cat([logits_h.detach().float(), logits_v.detach().float()], dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=30.0, neginf=-30.0)
        mean = features.mean(dim=-1, keepdim=True)
        centered = features - mean
        rms = centered.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
        features = (centered / rms).clamp(-6.0, 6.0)
        autocast_ctx = torch.cuda.amp.autocast(enabled=False) if features.is_cuda else contextlib.nullcontext()
        with autocast_ctx:
            gate_logits = self.hv_gate_mlp(features.to(dtype=self.hv_gate_mlp[0].weight.dtype))
        return torch.sigmoid(gate_logits).to(dtype=logits_h.dtype)

    def _run_backbone_and_vertical(
        self,
        hidden: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ):
        horizontal_hidden = hidden
        vertical_source = None
        for layer_num in range(self.n_layer):
            if layer_num == self.vertical_start_layer:
                vertical_source = horizontal_hidden
            horizontal_hidden = self.layers[layer_num](horizontal_hidden, freqs_cis, input_pos, mask)
        if vertical_source is None:
            vertical_source = horizontal_hidden
        vertical_hidden = vertical_source
        for layer_num in range(self.n_layer, len(self.layers)):
            vertical_hidden = self.layers[layer_num](vertical_hidden, freqs_cis, input_pos, mask)
        return horizontal_hidden, vertical_hidden
    
    def setup_proximity_mask(self, mask, block_size):
        mask[:, :] = 0 # zero out visual attention mask
        cur_token, previous_token = [], []
        H, W = int(block_size ** 0.5), int(block_size ** 0.5) # height and width of the token map
        for c in range(H + W - 1): # c is the Manhattan distance from the initial token x0
            cur_token = []
            for h in range(H):
                w = c - h # obtain coordinates (h, w) on the same slash
                if 0 <= w < W:
                    token_id = (h * W + w)
                    cur_token.append(token_id)
                    previous_token.append(token_id)
            for id in cur_token:
                mask[id, previous_token] = 1

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.config.block_size
        proximity_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        low = self.cls_token_num
        high = self.cls_token_num + self.config.block_size
        self.setup_proximity_mask(
            proximity_mask[low: high, low: high], 
            self.config.block_size,
        )
        self.proximity_mask = proximity_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        accept_first_last: Optional[bool] = False,
        input_pos:  Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
            mask = mask[:, None].to(h.device)
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.proximity_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis
        
        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        h, medusa_h = self._run_backbone_and_vertical(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logitsR = self.output(h).float() # logitsR represents the logits of the token on the right
        # medusa output layers
        medusa_h = self.medusa_norm(medusa_h)
        logitsB = self.medusa_output(medusa_h).float() # LogitsB represents the logits of the token below

        # Right(head) weight: scalar fallback (hv_gate_mlp computes per-position after rolls).
        if self.hv_mix_logit is None:
            right_w = 0.5
        elif self.hv_gate_mlp is not None:
            right_w = 0.5  # will be overridden per-position after rolls
        else:
            learned_w = torch.sigmoid(self.hv_mix_logit).to(device=logitsR.device, dtype=logitsR.dtype)
            if self.training and (self.hv_mix_target is not None) and (self.hv_mix_blend is not None):
                t = torch.as_tensor(self.hv_mix_target, device=logitsR.device, dtype=logitsR.dtype)
                b = torch.as_tensor(self.hv_mix_blend, device=logitsR.device, dtype=logitsR.dtype)
                b = torch.clamp(b, 0.0, 1.0)
                right_w = b * t + (1.0 - b) * learned_w
            else:
                right_w = learned_w

        gate_stats = {}
        if idx is not None and cond_idx is not None: # training
            logitsR = logitsR[:, self.cls_token_num - 1:]
            logitsB = logitsB[:, self.cls_token_num - 1:]

            bsz, _, emb_size = logitsR.shape
            gs = self.grid_size
            logitsR_cond = logitsR[:, 0, :]  # raw R logit at corner (cls pos)
            logitsB_cond = logitsB[:, 0, :]  # raw B logit at corner (cls pos)
            # Corner: gate from corner logits
            if self.hv_gate_mlp is not None:
                rw_corner = self._hv_gate(logitsR_cond, logitsB_cond)  # [B, 1]
            else:
                rw_corner = right_w
            cond_logits = logitsR_cond * rw_corner + logitsB_cond * (1 - rw_corner)
            # Roll to align logits with their target positions
            logitsR = logitsR[:, 1:, :] \
                        .reshape(bsz, gs, gs, emb_size) \
                        .roll(shifts=1, dims=2)
            logitsB = logitsB[:, 1:, :] \
                        .reshape(bsz, gs, gs, emb_size) \
                        .roll(shifts=1, dims=1)
            # Interior gate: computed AFTER roll, from same-target logitsR and logitsB
            # -> train/inference consistent (gate always sees the two predictions for the same token)
            if self.hv_gate_mlp is not None:
                _int_R = logitsR[:, 1:, 1:, :].reshape(bsz, -1, emb_size)
                _int_B = logitsB[:, 1:, 1:, :].reshape(bsz, -1, emb_size)
                rw_interior = self._hv_gate(_int_R, _int_B).reshape(bsz, gs - 1, gs - 1, 1)  # [B, gs-1, gs-1, 1]
            else:
                rw_interior = right_w
            logits = torch.zeros_like(logitsB)
            logits[:, 0, 0, :] = cond_logits
            logits[:, 0, 1:, :] = logitsR[:, 0, 1:, :]
            logits[:, 1:, 0, :] = logitsB[:, 1:, 0, :]
            logits[:, 1:, 1:, :] = (logitsR[:, 1:, 1:, :] * rw_interior
                                    + logitsB[:, 1:, 1:, :] * (1 - rw_interior))
            logits = logits.reshape(bsz, -1, emb_size).contiguous()

            if self.hv_gate_mlp is not None:
                corner_mean = rw_corner.mean().reshape(())
                if gs > 1:
                    gate_mean = rw_interior.mean().reshape(())
                    gate_entropy = self._binary_gate_entropy(rw_interior).mean().reshape(())
                    gate_collapse = (1.0 - gate_entropy).reshape(())
                else:
                    gate_mean = corner_mean
                    gate_entropy = corner_mean.new_ones(())
                    gate_collapse = corner_mean.new_zeros(())
            else:
                base_gate = torch.as_tensor(right_w, device=logitsR.device, dtype=logitsR.dtype).reshape(())
                gate_mean = base_gate
                corner_mean = base_gate
                gate_entropy = base_gate.new_ones(())
                gate_collapse = base_gate.new_zeros(())
            gate_stats = {
                "hv_gate_h": gate_mean,
                "hv_gate_v": (1.0 - gate_mean).reshape(()),
                "hv_gate_h_corner": corner_mean,
                "hv_gate_v_corner": (1.0 - corner_mean).reshape(()),
                "hv_gate_entropy": gate_entropy,
                "loss_gate_collapse": gate_collapse,
            }
        else:
            if cond_idx is not None: # prefill in inference
                logitsR = logitsR[:, -1:, :]
                logitsB = logitsB[:, -1:, :]
                if self.hv_gate_mlp is not None:
                    _rw = self._hv_gate(logitsR, logitsB)  # [B, 1, 1]
                else:
                    _rw = right_w
                logits = logitsR * _rw + logitsB * (1 - _rw) # left top token prediction
            else: # inference
                first = logitsR[:, 0, :]
                last = logitsB[:, -1, :]
                if self.hv_gate_mlp is not None:
                    # Gate from SAME pair being mixed: logitsR[i+1] and logitsB[i] for each target
                    # Consistent with training where gate uses rolled logitsR[r,c] & logitsB[r,c]
                    _rw_mid = self._hv_gate(logitsR[:, 1:, :], logitsB[:, :-1, :])  # [B, seq-1, 1]
                else:
                    _rw_mid = right_w
                middle = logitsR[:, 1:, :] * _rw_mid + logitsB[:, :-1, :] * (1 - _rw_mid)
                if accept_first_last:
                    logits = torch.cat([first[:, None, :], middle, last[:, None, :]], dim=1)
                else:
                    logits = middle

        # if we are given some desired targets also calculate the loss
        loss = None
        if self.split_loss and targets is not None and idx is not None:
            gs = self.grid_size
            targets_2d = targets.reshape(bsz, gs, gs)
            ignore_index = -100
            target_valid = (targets_2d != ignore_index).to(dtype=logitsR.dtype)

            # Per-head logits in 2D grid (fill only valid positions)
            logitsR_2d = torch.zeros(bsz, gs, gs, emb_size, device=logitsR.device, dtype=logitsR.dtype)
            logitsR_2d[:, 0, 0, :] = logitsR_cond
            logitsR_2d[:, 0, 1:, :] = logitsR[:, 0, 1:, :]
            logitsR_2d[:, 1:, 1:, :] = logitsR[:, 1:, 1:, :]

            logitsB_2d = torch.zeros(bsz, gs, gs, emb_size, device=logitsB.device, dtype=logitsB.dtype)
            logitsB_2d[:, 0, 0, :] = logitsB_cond
            logitsB_2d[:, 1:, 0, :] = logitsB[:, 1:, 0, :]
            logitsB_2d[:, 1:, 1:, :] = logitsB[:, 1:, 1:, :]

            # Per-position CE (reduction='none')
            flat_t = targets_2d.reshape(-1)
            ceR = F.cross_entropy(
                logitsR_2d.reshape(-1, emb_size),
                flat_t,
                reduction='none',
                ignore_index=ignore_index,
            ).reshape(bsz, gs, gs)
            ceB = F.cross_entropy(
                logitsB_2d.reshape(-1, emb_size),
                flat_t,
                reduction='none',
                ignore_index=ignore_index,
            ).reshape(bsz, gs, gs)

            # Validity masks: R invalid at col0 rows 1+, B invalid at row0 cols 1+
            validR = torch.ones(gs, gs, device=ceR.device, dtype=ceR.dtype)
            validR[1:, 0] = 0
            validB = torch.ones(gs, gs, device=ceB.device, dtype=ceB.dtype)
            validB[0, 1:] = 0

            weightR = validR * target_valid
            weightB = validB * target_valid
            denomR = weightR.sum().clamp_min(1.0)
            denomB = weightB.sum().clamp_min(1.0)
            lossR = (ceR * weightR).sum() / denomR
            lossB = (ceB * weightB).sum() / denomB

            lam = self.split_loss_lambda
            loss = lam * lossR + (1 - lam) * lossB

            # Col0 boost: extra penalty for B head on first-column tokens
            if self.col0_boost > 0:
                col0_mask = torch.zeros(gs, gs, device=ceB.device, dtype=ceB.dtype)
                col0_mask[1:, 0] = 1
                col0_weight = col0_mask * target_valid
                col0_denom = col0_weight.sum().clamp_min(1.0)
                loss_col0 = (ceB * col0_weight).sum() / col0_denom
                loss = loss + self.col0_boost * loss_col0
        elif valid is not None:
            flat_targets = targets.view(-1)
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                flat_targets,
                reduction='none',
                ignore_index=-100,
            )
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1).to(dtype=loss_all.dtype)
            valid_all = valid_all * (flat_targets != -100).to(dtype=loss_all.dtype)
            loss = (loss_all * valid_all).sum() / valid_all.sum().clamp_min(1.0)
        elif targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        if return_stats:
            return logits, loss, gate_stats
        return logits, loss

    @torch.no_grad()
    def get_hv_right_weight(self) -> float:
        """Return current right(head) weight alpha in [0,1]. For hv_gate_mlp, returns NaN (per-position)."""
        if self.hv_gate_mlp is not None:
            return float("nan")  # per-position gate, no single scalar
        if self.hv_mix_logit is None:
            return 0.5
        return float(torch.sigmoid(self.hv_mix_logit).item())

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) 

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) 

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs))

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) 

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))

def GPT_M(**kwargs):
    return Transformer(ModelArgs(n_layer=18, n_head=16, dim=1024, **kwargs))

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) 
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-M': GPT_M, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}
