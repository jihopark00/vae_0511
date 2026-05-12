"""
DDP training script for ViT-VAE.

Launch (single node, N GPUs):
    cd /home/ljeadec31/opt/ssl/levae && conda activate flow
    torchrun --standalone --nproc_per_node=N train_vae.py \\
        --exps_dir exps --exp_name exp001 --config configs/vae.yaml \\
        --data_path /datasets/imagenet/train \\
        --last_ckpt_every 1000 --ckpt_every 10000 \\
        --log_every 50 --visualize_every 1000 \\
        --num_workers 8 --seed 42 \\
        [--resume_last] \\
        [--wandb --wandb_key ... --wandb_entity ... --wandb_project ...]

Resume:
    torchrun --standalone --nproc_per_node=N train_vae.py \\
        --exps_dir exps --exp_name exp001 \\
        --data_path /datasets/imagenet/train --resume_last
"""

import argparse
import contextlib
import hashlib
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models  # noqa: E402
from lejepa.multivariate import SlicingUnivariateTest  # noqa: E402
from lejepa.univariate import EppsPulley  # noqa: E402


# ─── Distributed / general helpers ────────────────────────────────────────────


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def dist_setup() -> Tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    return local_rank, rank, world_size


def dist_cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def deep_dict_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_dict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(deep_dict_equal(x, y) for x, y in zip(a, b))
    return a == b


def atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def wandb_run_id_from_name(exp_name: str) -> str:
    return hashlib.md5(exp_name.encode()).hexdigest()[:16]


# ─── Config ───────────────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(d: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def _yaml_normalize(d: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(d, sort_keys=False))


def resolve_config(args: argparse.Namespace, exp_dir: Path) -> dict:
    saved_path = exp_dir / "config.yaml"
    given = load_yaml(Path(args.config)) if args.config else None

    # Main writes the saved config first (if needed), then everyone reads.
    # Without this barrier, non-main ranks race against main's write and may
    # see saved as None even when main is about to create it.
    if given is not None and is_main_process() and not saved_path.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(given, saved_path)
    if dist.is_initialized():
        dist.barrier()

    saved = load_yaml(saved_path) if saved_path.exists() else None

    if given is not None and saved is None:
        return _yaml_normalize(given)

    if given is not None and saved is not None:
        given_n = _yaml_normalize(given)
        if not deep_dict_equal(given_n, saved):
            raise RuntimeError(
                f"Config mismatch: --config={args.config} differs from "
                f"existing {saved_path}. Remove --config to resume, or fix the "
                f"config to match the saved one."
            )
        return saved

    if given is None and saved is not None:
        return saved

    raise RuntimeError(
        f"No config provided and no saved config at {saved_path}. "
        f"Pass --config for a fresh run."
    )


def validate_config(cfg: dict, world_size: int) -> dict:
    gbs = cfg["global_batch_size"]
    accum = cfg["train_config"].get("accum_steps", 1)
    if gbs % (accum * world_size) != 0:
        raise ValueError(
            f"global_batch_size={gbs} not divisible by "
            f"accum_steps*world_size={accum*world_size}"
        )
    cfg["_micro_batch_size"] = gbs // (accum * world_size)

    if cfg["data_config"]["img_size"] != cfg["vae_config"]["img_size"]:
        raise ValueError(
            f"data_config.img_size={cfg['data_config']['img_size']} != "
            f"vae_config.img_size={cfg['vae_config']['img_size']}"
        )

    if cfg["precision"] not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {cfg['precision']}")

    return cfg


# ─── Builders ─────────────────────────────────────────────────────────────────


def build_dataloader(
    args: argparse.Namespace, cfg: dict, rank: int, world_size: int
) -> Tuple[Any, DistributedSampler, DataLoader]:
    img_size = cfg["data_config"]["img_size"]
    aug = cfg["data_config"].get("augmentation", "none")
    hflip = cfg["data_config"].get("hflip", False)

    if aug == "none" or aug is None:
        ops = [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
    elif aug == "random_resized_crop":
        ops = [transforms.RandomResizedCrop(img_size)]
    elif aug == "random_crop":
        ops = [
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size),
        ]
    else:
        raise ValueError(
            f"unknown data_config.augmentation={aug!r}; "
            f"expected one of: 'none', 'random_resized_crop', 'random_crop'"
        )
    if hflip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    tfm = transforms.Compose(ops)
    dataset = datasets.ImageFolder(args.data_path, transform=tfm)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["_micro_batch_size"],
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return dataset, sampler, loader


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    cls = getattr(models, cfg["vae_class"])
    model = cls(**cfg["vae_config"])
    model.to(device)
    return model


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    tc = cfg["train_config"]
    if tc.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError(f"only adamw supported, got {tc.get('optimizer')}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc.get("weight_decay", 0.0),
        betas=tuple(tc.get("betas", [0.9, 0.95])),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict
) -> torch.optim.lr_scheduler.LambdaLR:
    tc = cfg["train_config"]
    max_steps = tc["max_steps"]
    warmup = tc.get("warmup_steps", 0)
    min_lr = tc.get("min_lr", 0.0)
    base_lr = tc["lr"]
    kind = tc.get("scheduler", "cosine").lower()

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        if kind == "constant":
            return 1.0
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + (base_lr - min_lr) * cos) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_sigreg_fn(cfg: dict, device: torch.device) -> Optional[nn.Module]:
    sr = cfg["loss_config"].get("sigreg_loss", {"use": False})
    if not sr.get("use", False):
        return None
    uni = EppsPulley(n_points=sr.get("n_points", 17))
    fn = SlicingUnivariateTest(univariate_test=uni, num_slices=sr["num_slices"])
    fn.to(device)
    return fn


# ─── Loss ─────────────────────────────────────────────────────────────────────


def compute_loss(
    model_unwrapped: nn.Module,
    ddp_model: nn.Module,
    x: torch.Tensor,
    cfg: dict,
    sigreg_fn: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    lc = cfg["loss_config"]
    out = ddp_model(x, return_dict=True)
    x_normalized = (x - model_unwrapped.img_mean) / model_unwrapped.img_std
    x_normalized_recon = out["x_normalized_recon"]
    z, mu, logvar = out["z"], out["mu"], out["logvar"]

    recon_cfg = lc.get("recon_loss", {"type": "l1", "weight": 1.0})
    if recon_cfg["type"] == "l1":
        recon_loss = F.l1_loss(x_normalized_recon, x_normalized)
    elif recon_cfg["type"] == "l2":
        recon_loss = F.mse_loss(x_normalized_recon, x_normalized)
    else:
        raise ValueError(f"unknown recon_loss.type={recon_cfg['type']}")

    total = recon_loss * recon_cfg.get("weight", 1.0)
    logs: Dict[str, torch.Tensor] = {"recon_loss": recon_loss.detach()}

    if logvar is not None:
        logs["std_mean"] = torch.exp(0.5 * logvar.detach()).mean()

    kld_cfg = lc.get("kld_loss", {"use": False})
    if kld_cfg.get("use", False):
        assert model_unwrapped.variational, "KLD requires variational=True"
        target_mu = kld_cfg.get("target_mu", 0.0)
        target_std = kld_cfg.get("target_std", 1.0)
        var_gt = target_std ** 2
        logvar_gt = 2.0 * math.log(target_std)
        mu_f = mu.flatten(1)
        lv_f = torch.clamp(logvar.flatten(1), -30.0, 20.0)
        kld = 0.5 * torch.sum(
            (mu_f - target_mu) ** 2 / var_gt
            + torch.exp(lv_f) / var_gt
            - 1.0
            - lv_f
            + logvar_gt,
            dim=1,
        ).mean()
        total = total + kld * kld_cfg["weight"]
        logs["kld_loss"] = kld.detach()

    sr_cfg = lc.get("sigreg_loss", {"use": False})
    if sr_cfg.get("use", False) and sigreg_fn is not None:
        sigreg = sigreg_fn(z.flatten(1))
        total = total + sigreg * sr_cfg["weight"]
        logs["sigreg_loss"] = sigreg.detach()

    logs["loss"] = total.detach()
    return total, logs


# ─── Checkpoint I/O ───────────────────────────────────────────────────────────


def save_ckpt(
    path: Path,
    *,
    step: int,
    model_unwrapped: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    cfg: dict,
    epoch: int,
) -> None:
    if not is_main_process():
        return
    payload = {
        "step": step,
        "epoch": epoch,
        "model": model_unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "py_rng": random.getstate(),
        "config": cfg,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, path)


def load_ckpt(
    path: Path,
    *,
    model_unwrapped: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    map_location: Any,
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model_unwrapped.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["torch_rng"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    random.setstate(ckpt["py_rng"])
    return int(ckpt["step"]), int(ckpt.get("epoch", 0))


# ─── Visualization ────────────────────────────────────────────────────────────


@torch.no_grad()
def pca_visualize_z(z_batch: torch.Tensor, H_p: int, W_p: int) -> np.ndarray:
    B = z_batch.shape[0]
    z_np = z_batch.float().cpu().numpy()
    imgs = np.empty((B, H_p, W_p, 3), dtype=np.float32)
    for i in range(B):
        proj = PCA(n_components=3).fit_transform(z_np[i])
        lo = proj.min(0, keepdims=True)
        hi = proj.max(0, keepdims=True)
        proj = (proj - lo) / (hi - lo + 1e-8)
        imgs[i] = proj.reshape(H_p, W_p, 3)
    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
    nrow = int(math.ceil(math.sqrt(B)))
    grid = make_grid(imgs_t, nrow=nrow, padding=2)
    grid_np = grid.numpy().transpose(1, 2, 0)
    return (grid_np * 255).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def make_recon_grid(x: torch.Tensor, x_recon: torch.Tensor, n: int) -> torch.Tensor:
    n = min(n, x.size(0))
    pairs = torch.stack([x[:n], x_recon.clamp(0, 1)[:n]], dim=1)
    pairs = pairs.reshape(2 * n, *x.shape[1:])
    nrow = int(math.ceil(math.sqrt(2 * n)))
    return make_grid(pairs, nrow=nrow)


def log_visuals(
    args: argparse.Namespace,
    exp_dir: Path,
    step: int,
    x: torch.Tensor,
    out_dict: dict,
    model_unwrapped: nn.Module,
    wandb_run: Any,
) -> None:
    H_p, W_p = model_unwrapped.num_patches_h, model_unwrapped.num_patches_w
    n = min(args.visualize_batch, x.size(0))
    z_batch = out_dict["z"][:n]
    pca_img = pca_visualize_z(z_batch, H_p, W_p)

    grid = make_recon_grid(
        x.detach().cpu().float(),
        out_dict["x_recon"].detach().cpu().float(),
        n=n,
    )
    grid_np = grid.numpy().transpose(1, 2, 0)
    grid_uint = (grid_np * 255).clip(0, 255).astype(np.uint8)

    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {
                "viz/z_pca": wandb.Image(pca_img),
                "viz/recon_grid": wandb.Image(grid_uint),
            },
            step=step,
        )

    if args.save_visualizations_locally:
        viz_dir = exp_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pca_img).save(viz_dir / f"step_{step:08d}_zpca.png")
        Image.fromarray(grid_uint).save(viz_dir / f"step_{step:08d}_recon.png")


# ─── Training loop ────────────────────────────────────────────────────────────


def train_loop(
    args: argparse.Namespace,
    cfg: dict,
    ddp_model: DDP,
    model_unwrapped: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    sigreg_fn: Optional[nn.Module],
    loader: DataLoader,
    sampler: DistributedSampler,
    ckpt_dir: Path,
    exp_dir: Path,
    start_step: int,
    start_epoch: int,
    device: torch.device,
    wandb_run: Any,
) -> Tuple[int, int]:
    tc = cfg["train_config"]
    max_steps = tc["max_steps"]
    accum_steps = tc.get("accum_steps", 1)
    grad_clip = tc.get("grad_clip", 1.0)
    precision = cfg["precision"]
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    amp_enabled = precision == "bf16"

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    samples_per_step_global = cfg["_micro_batch_size"] * accum_steps * world_size

    ddp_model.train()
    step = start_step
    epoch = start_epoch
    sampler.set_epoch(epoch)
    data_iter = iter(loader)

    t_log = time.time()
    samples_since_log = 0
    steps_since_log = 0

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_log: Dict[str, float] = {}
        last_x: Optional[torch.Tensor] = None

        for micro_i in range(accum_steps):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device, non_blocking=True)

            sync_ctx = (
                ddp_model.no_sync()
                if micro_i < accum_steps - 1
                else contextlib.nullcontext()
            )
            with sync_ctx:
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    loss, logs = compute_loss(
                        model_unwrapped, ddp_model, x, cfg, sigreg_fn
                    )
                (loss / accum_steps).backward()

            for k, v in logs.items():
                accum_log[k] = accum_log.get(k, 0.0) + float(v) / accum_steps

            if micro_i == accum_steps - 1:
                last_x = x.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        samples_since_log += samples_per_step_global
        steps_since_log += 1

        # logging
        if step % args.log_every == 0 and is_main_process():
            dt = time.time() - t_log
            sps = samples_since_log / max(1e-6, dt)
            steps_per_sec = steps_since_log / max(1e-6, dt)
            lr_now = optimizer.param_groups[0]["lr"]
            parts = [
                f"step {step:>7d}/{max_steps}",
                f"lr={lr_now:.2e}",
                f"gn={float(grad_norm):.3f}",
                f"sps={sps:.1f}",
                f"steps/s={steps_per_sec:.2f}",
            ]
            parts += [f"{k}={v:.4f}" for k, v in accum_log.items()]
            print("  ".join(parts), flush=True)
            if wandb_run is not None:
                wandb_payload = {
                    "step": step,
                    "lr": lr_now,
                    "grad_norm": float(grad_norm),
                    "samples_per_sec": sps,
                    "steps_per_sec": steps_per_sec,
                }
                for k, v in accum_log.items():
                    wandb_payload[f"loss/{k}"] = v
                wandb_run.log(wandb_payload, step=step)
            t_log = time.time()
            samples_since_log = 0
            steps_since_log = 0

        # visualization
        if step % args.visualize_every == 0 and is_main_process() and last_x is not None:
            ddp_model.eval()
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out_viz = model_unwrapped(last_x, return_dict=True)
            log_visuals(args, exp_dir, step, last_x, out_viz, model_unwrapped, wandb_run)
            ddp_model.train()

        # checkpointing
        if step % args.last_ckpt_every == 0:
            save_ckpt(
                ckpt_dir / "last.pt",
                step=step,
                model_unwrapped=model_unwrapped,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
            )
            if dist.is_initialized():
                dist.barrier()
        if step % args.ckpt_every == 0:
            save_ckpt(
                ckpt_dir / f"step_{step:08d}.pt",
                step=step,
                model_unwrapped=model_unwrapped,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
            )
            if dist.is_initialized():
                dist.barrier()

    return step, epoch


# ─── CLI / main ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exps_dir", type=str, required=True)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--resume_last", action="store_true")
    p.add_argument("--last_ckpt_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=10000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--visualize_every", type=int, default=1000)
    p.add_argument("--visualize_batch", type=int, default=8)
    p.add_argument("--save_visualizations_locally", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_key", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    local_rank, rank, world_size = dist_setup()
    device = torch.device(f"cuda:{local_rank}")

    exp_dir = Path(args.exps_dir) / args.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    if is_main_process():
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    cfg = resolve_config(args, exp_dir)
    cfg = validate_config(cfg, world_size)

    if is_main_process():
        print(
            f"[config] global_batch_size={cfg['global_batch_size']}  "
            f"micro_batch_size={cfg['_micro_batch_size']}  "
            f"accum_steps={cfg['train_config'].get('accum_steps', 1)}  "
            f"world_size={world_size}  precision={cfg['precision']}",
            flush=True,
        )

    seed_everything(args.seed, rank)

    dataset, sampler, loader = build_dataloader(args, cfg, rank, world_size)
    model = build_model(cfg, device)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    optimizer = build_optimizer(ddp_model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    sigreg_fn = build_sigreg_fn(cfg, device)

    start_step = 0
    start_epoch = 0
    if args.resume_last:
        last = ckpt_dir / "last.pt"
        if last.exists():
            start_step, start_epoch = load_ckpt(
                last,
                model_unwrapped=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location={"cuda:0": f"cuda:{local_rank}"},
            )
            # load_ckpt restored RNG state from the rank-0 checkpoint, leaving
            # all ranks identical. Re-seed per-rank (incorporating step so a
            # resume doesn't replay the same sequence as the original run).
            seed_everything(args.seed + start_step, rank)
            if is_main_process():
                print(f"[resume] resumed from {last} at step {start_step}", flush=True)
        else:
            if is_main_process():
                print(f"[resume] no last.pt at {last}, starting fresh", flush=True)
        dist.barrier()

    # Seed sigreg's projection counter so post-resume forwards don't reuse the
    # pre-resume seeds. global_step advances once per forward (i.e. once per
    # micro-batch), not once per train step, so scale by accum_steps to match
    # what a continuous run would have at this train step. MAX-reduced across
    # ranks each forward, so setting it identically here is sufficient.
    if sigreg_fn is not None:
        accum_steps = cfg["train_config"].get("accum_steps", 1)
        sigreg_fn.global_step.fill_(start_step * accum_steps)

    wandb_run = None
    if args.wandb and is_main_process():
        import wandb

        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.exp_name,
            # id=wandb_run_id_from_name(args.exp_name),
            resume="allow",
            config=cfg,
        )

    final_step, final_epoch = train_loop(
        args,
        cfg,
        ddp_model,
        model,
        optimizer,
        scheduler,
        sigreg_fn,
        loader,
        sampler,
        ckpt_dir,
        exp_dir,
        start_step,
        start_epoch,
        device,
        wandb_run,
    )

    save_ckpt(
        ckpt_dir / "last.pt",
        step=final_step,
        model_unwrapped=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        epoch=final_epoch,
    )
    if dist.is_initialized():
        dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()

    dist_cleanup()


if __name__ == "__main__":
    main(parse_args())
