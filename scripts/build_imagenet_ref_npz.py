#!/usr/bin/env python3
import argparse
import os
import sys
import tarfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resampling(name: str):
    return getattr(Image, "Resampling", Image).__dict__[name]


def center_crop_arr(pil_image: Image.Image, image_size: int) -> np.ndarray:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=_resampling("BOX"))

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=_resampling("BICUBIC"))
    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def list_tar_members(val_tar: str) -> List[tarfile.TarInfo]:
    with tarfile.open(val_tar, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg", ".png"))]
    return sorted(members, key=lambda m: m.name)


def list_image_files(val_dir: str) -> List[Path]:
    root = Path(val_dir)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
    return sorted(files)


def iter_tar_images(val_tar: str, members: Sequence[tarfile.TarInfo], image_size: int) -> Iterable[np.ndarray]:
    with tarfile.open(val_tar, "r") as tar:
        for member in members:
            fp = tar.extractfile(member)
            if fp is None:
                continue
            with fp:
                with Image.open(fp) as pil_image:
                    yield center_crop_arr(pil_image, image_size)


def iter_dir_images(files: Sequence[Path], image_size: int) -> Iterable[np.ndarray]:
    for path in files:
        with Image.open(path) as pil_image:
            yield center_crop_arr(pil_image, image_size)


def batched_images(
    images: Iterable[np.ndarray],
    batch_size: int,
    arr_memmap: Optional[np.memmap],
    arr_num: int,
    stats_limit: Optional[int],
) -> Iterable[np.ndarray]:
    batch = []
    seen = 0
    for image in images:
        if stats_limit is not None and seen >= stats_limit:
            break
        if arr_memmap is not None and seen < arr_num:
            arr_memmap[seen] = image
        batch.append(image)
        seen += 1
        if len(batch) == batch_size:
            print(f"processed {seen} images", flush=True)
            yield np.stack(batch, axis=0)
            batch = []
    if batch:
        print(f"processed {seen} images", flush=True)
        yield np.stack(batch, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ImageNet reference npz for evaluations/c2i/evaluator.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--val-tar", type=str, help="Path to ILSVRC2012_img_val.tar")
    source.add_argument("--val-dir", type=str, help="Path to an extracted ImageNet val image directory")
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "pretrained_models" / "VIRTUAL_imagenet384_labeled.npz"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--arr-num", type=int, default=10000, help="Number of real images to store as arr_0 for precision/recall")
    parser.add_argument("--stats-limit", type=int, default=0, help="Use only this many images for stats; 0 means all images")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tmp-dir", type=str, default=str(REPO_ROOT / "tmp"))
    parser.add_argument("--cpu", action="store_true", help="Hide GPUs from TensorFlow before evaluator import")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    sys.path.insert(0, str(REPO_ROOT))
    import tensorflow._api.v2.compat.v1 as tf
    from evaluations.c2i.evaluator import Evaluator

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    arr_path = tmp_dir / f"{output.stem}.arr_0.{args.image_size}.mmap"

    if args.val_tar:
        items = list_tar_members(args.val_tar)
        image_iter = iter_tar_images(args.val_tar, items, args.image_size)
        source_desc = args.val_tar
    else:
        items = list_image_files(args.val_dir)
        image_iter = iter_dir_images(items, args.image_size)
        source_desc = args.val_dir

    if not items:
        raise RuntimeError(f"No images found in {source_desc}")

    stats_limit = args.stats_limit if args.stats_limit and args.stats_limit > 0 else None
    stats_num = min(len(items), stats_limit) if stats_limit is not None else len(items)
    arr_num = min(args.arr_num, stats_num)
    print(f"source={source_desc}", flush=True)
    print(f"images_for_stats={stats_num}, arr_0_images={arr_num}, image_size={args.image_size}", flush=True)
    print(f"arr_0_memmap={arr_path}", flush=True)

    arr_memmap = np.memmap(arr_path, mode="w+", dtype=np.uint8, shape=(arr_num, args.image_size, args.image_size, 3))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))
    print("warming up TensorFlow...", flush=True)
    evaluator.warmup()

    print("computing ImageNet reference activations...", flush=True)
    batches = batched_images(image_iter, args.batch_size, arr_memmap, arr_num, stats_limit)
    pool_acts, spatial_acts = evaluator.compute_activations(batches)
    arr_memmap.flush()

    print("computing statistics...", flush=True)
    pool_stats = evaluator.compute_statistics(pool_acts)
    spatial_stats = evaluator.compute_statistics(spatial_acts)

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    print(f"writing {tmp_output}...", flush=True)
    with open(tmp_output, "wb") as output_file:
        np.savez(
            output_file,
            mu=pool_stats.mu,
            sigma=pool_stats.sigma,
            mu_s=spatial_stats.mu,
            sigma_s=spatial_stats.sigma,
            arr_0=np.asarray(arr_memmap),
        )
    os.replace(tmp_output, output)
    print(f"done: {output}", flush=True)


if __name__ == "__main__":
    main()
