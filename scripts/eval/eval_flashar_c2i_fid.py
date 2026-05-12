import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from autoregressive.models.gpt import GPT_models as FLASHAR_GPT_models
from autoregressive.models.generate import generate
from autoregressive.train.train_c2i_flashar import extract_state_dict, normalize_state_dict, configure_vertical_branch
from autoregressive.utils.fid_eval import run_fid_eval
from tokenizer.tokenizer_image.vq_model import VQ_models


class Logger:
    def __init__(self, rank):
        self.rank = rank
    def info(self, msg):
        if self.rank == 0:
            print(msg, flush=True)
    def exception(self, msg):
        if self.rank == 0:
            print(msg, flush=True)


def get_attr(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


def main(cli):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    dist.init_process_group(cli.dist_backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logger = Logger(rank)

    ckpt = torch.load(cli.gpt_ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args") if isinstance(ckpt, dict) else None
    state = normalize_state_dict(extract_state_dict(ckpt))

    gpt_model = cli.gpt_model or get_attr(ckpt_args, "gpt_model", "GPT-XL")
    image_size = cli.image_size or get_attr(ckpt_args, "image_size", 256)
    downsample_size = cli.downsample_size or get_attr(ckpt_args, "downsample_size", 16)
    latent_size = image_size // downsample_size
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cli.precision]

    eval_args = SimpleNamespace(
        image_size=image_size,
        image_size_eval=cli.image_size_eval or image_size,
        downsample_size=downsample_size,
        num_classes=get_attr(ckpt_args, "num_classes", 1000),
        codebook_size=get_attr(ckpt_args, "codebook_size", 16384),
        codebook_embed_dim=get_attr(ckpt_args, "codebook_embed_dim", 8),
        cls_token_num=get_attr(ckpt_args, "cls_token_num", 1),
        gpt_type=get_attr(ckpt_args, "gpt_type", "c2i"),
        hv_mix=get_attr(ckpt_args, "hv_mix", False),
        hv_mix_init=get_attr(ckpt_args, "hv_mix_init", 0.5),
        hv_gate=get_attr(ckpt_args, "hv_gate", False),
        medusa_attention_num=get_attr(ckpt_args, "medusa_attention_num", 1),
        vertical_start_layer=get_attr(ckpt_args, "vertical_start_layer", -1),
        vertical_start_last_minus_depth=get_attr(ckpt_args, "vertical_start_last_minus_depth", False),
        fid_ref=cli.fid_ref,
        fid_num_samples=cli.num_samples,
        fid_batch_size=cli.batch_size,
        fid_sample_dir=cli.sample_dir,
        fid_cfg_scale=cli.cfg_scale,
        fid_cfg_interval=cli.cfg_interval,
        fid_temperature=cli.temperature,
        fid_top_k=cli.top_k,
        fid_top_p=cli.top_p,
        fid_skip_evaluator=False,
    )

    torch.manual_seed(cli.seed + rank)
    logger.info(f"Loading FlashAR {gpt_model}, image_size={image_size}, eval={eval_args.image_size_eval}, cfg={cli.cfg_scale}")
    model = FLASHAR_GPT_models[gpt_model](
        vocab_size=eval_args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=eval_args.num_classes,
        cls_token_num=eval_args.cls_token_num,
        model_type=eval_args.gpt_type,
        hv_mix=eval_args.hv_mix,
        hv_mix_init=eval_args.hv_mix_init,
        hv_gate=eval_args.hv_gate,
        medusa_attention_num=eval_args.medusa_attention_num,
        vertical_start_layer=eval_args.vertical_start_layer,
    ).to(device=device, dtype=precision)
    configure_vertical_branch(model, eval_args, logger if rank == 0 else None)
    incompatible = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded ckpt: missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")
    model.eval()
    del ckpt, state

    vq_model = VQ_models[cli.vq_model](
        codebook_size=eval_args.codebook_size,
        codebook_embed_dim=eval_args.codebook_embed_dim,
    ).to(device)
    vq_ckpt = torch.load(cli.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_ckpt["model"])
    vq_model.eval()
    del vq_ckpt

    os.makedirs(cli.sample_dir, exist_ok=True)
    npz_path, txt_path, metrics = run_fid_eval(
        eval_args,
        model,
        vq_model,
        device,
        step=cli.step,
        logger=logger,
        generate_fn=generate,
        epoch=None,
        sample_dir=os.path.join(cli.sample_dir, "latest"),
        keep_last_samples=True,
        rank=rank,
        world_size=world_size,
        barrier=dist.barrier,
    )
    if rank == 0:
        payload = {"cfg_scale": cli.cfg_scale, "ckpt": cli.gpt_ckpt, "npz_path": npz_path, "txt_path": txt_path}
        payload.update(metrics)
        with open(os.path.join(cli.sample_dir, "metrics.json"), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpt-ckpt", required=True)
    p.add_argument("--gpt-model", default=None)
    p.add_argument("--vq-ckpt", default="./pretrained_models/vq_ds16_c2i.pt")
    p.add_argument("--vq-model", default="VQ-16")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--image-size-eval", type=int, default=None)
    p.add_argument("--downsample-size", type=int, default=None)
    p.add_argument("--fid-ref", required=True)
    p.add_argument("--sample-dir", required=True)
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--cfg-interval", type=float, default=-1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--precision", default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--dist-backend", default="gloo", choices=["gloo", "nccl"])
    main(p.parse_args())
