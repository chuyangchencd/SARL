"""Train and evaluate probes on the SARL tasks.

Usage::

    python -m probe.run --backbone rawfeat_logmel_binaural \\
        --tasks azimuth elevation distance event --data_root /path/to/data

Freezes the backbone, trains one probe head per task (linear, or an MLP with
``--use_mlp_head``), and writes a JSON of per-task scores. Source and room tasks
must be run separately.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import batch_process, make_loader
from metrics import build_metrics
from models.registry import build_backbone
from probe.model import NaivePredictor, Probe
from tasks import Binning, Family, Metric, TaskSpec, resolve_tasks


# =============================================================================
# Targets and loss
# =============================================================================

def soft_target(spec: TaskSpec, raw: torch.Tensor, sigma_bins: float) -> torch.Tensor:
    """Gaussian soft label over bin centers for a continuous task, shape ``[B, C]``.

    The kernel width is ``sigma_bins`` bins; samples with non-finite labels get a
    uniform target.
    """
    centers = spec.bin_centers().to(raw.device, raw.dtype)          # [C]
    C = spec.n_classes
    sigma = (spec.value_max - spec.value_min) / C * sigma_bins
    diff = centers.unsqueeze(0) - raw.unsqueeze(1)                  # [B, C]
    if spec.binning is Binning.CIRCULAR:
        diff = diff.abs() % 360.0
        diff = torch.minimum(diff, 360.0 - diff)
    else:
        diff = diff.abs()
    valid = torch.isfinite(raw).unsqueeze(1)
    w = torch.exp(-(diff ** 2) / (2 * sigma ** 2 + 1e-12))
    w = torch.where(valid, w, torch.zeros_like(w))
    total = w.sum(dim=1, keepdim=True)
    soft = torch.where(total > 1e-8, w / total.clamp(min=1e-12),
                       torch.full_like(w, 1.0 / C))
    soft = soft + 1e-8
    return soft / soft.sum(dim=1, keepdim=True)


def hard_target(spec: TaskSpec, raw: torch.Tensor) -> torch.Tensor:
    """Nearest-bin / class index target ``[B]`` (long)."""
    idx = [spec.value_to_index(float(v)) if torch.isfinite(v) else 0 for v in raw]
    return torch.tensor(idx, dtype=torch.long, device=raw.device)


def task_loss(spec: TaskSpec, logits: torch.Tensor, raw: torch.Tensor,
              soft_labels: bool, sigma_bins: float) -> torch.Tensor:
    """Cross-entropy loss for one task (soft targets for continuous tasks)."""
    if spec.metric is Metric.NORM_MAE and soft_labels:
        return F.cross_entropy(logits, soft_target(spec, raw, sigma_bins))
    return F.cross_entropy(logits, hard_target(spec, raw))


# =============================================================================
# Train / evaluate
# =============================================================================

@torch.no_grad()
def validation_losses(probe: Probe, loader: DataLoader, specs: Sequence[TaskSpec],
                      audio_format: str, soft_labels: bool, sigma_bins: float,
                      device: str) -> Dict[str, float]:
    """Mean per-task cross-entropy loss over the validation set."""
    probe.eval()
    sums = {s.name: 0.0 for s in specs}
    n = 0
    for src, rir, labels, noise in loader:
        src, rir = src.to(device), rir.to(device)
        noise = noise.to(device) if noise is not None else None
        x = batch_process(src, rir, audio_format, noise)
        out = probe(x)
        for s in specs:
            sums[s.name] += float(
                task_loss(s, out[s.name], labels[s.name].to(device), soft_labels, sigma_bins))
        n += 1
    return {name: total / max(1, n) for name, total in sums.items()}


def train(probe: Probe, train_loader: DataLoader, val_loader: DataLoader,
          specs: Sequence[TaskSpec], audio_format: str, epochs: int, lr: float,
          weight_decay: float, grad_clip: float, soft_labels: bool,
          sigma_bins: float, device: str) -> Dict[str, int]:
    """Train all heads jointly; keep each head's lowest-validation-loss epoch.

    The heads share one forward pass, but their best epochs differ, so each head is
    checkpointed independently by its own lowest validation loss. The probe is left
    holding each head's best state.
    """
    opt = torch.optim.AdamW(probe.head_parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_loss = {s.name: float("inf") for s in specs}
    best_state = {s.name: copy.deepcopy(probe.heads[s.name].state_dict()) for s in specs}
    best_epoch = {s.name: 0 for s in specs}

    for epoch in range(epochs):
        probe.train()
        train_loader.dataset.set_epoch(epoch)
        running = 0.0
        for src, rir, labels, noise in train_loader:
            src, rir = src.to(device), rir.to(device)
            noise = noise.to(device) if noise is not None else None
            x = batch_process(src, rir, audio_format, noise)
            out = probe(x)
            loss = sum(task_loss(s, out[s.name], labels[s.name].to(device),
                                 soft_labels, sigma_bins) for s in specs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Per-head gradient clipping (each head clipped independently).
            for head in probe.heads.values():
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            opt.step()
            running += float(loss)
        sched.step()

        val = validation_losses(probe, val_loader, specs, audio_format,
                                soft_labels, sigma_bins, device)
        for s in specs:
            if val[s.name] < best_loss[s.name]:
                best_loss[s.name] = val[s.name]
                best_state[s.name] = {k: v.detach().cpu().clone()
                                      for k, v in probe.heads[s.name].state_dict().items()}
                best_epoch[s.name] = epoch + 1
        val_str = "  ".join(f"{s.name}={val[s.name]:.3f}" for s in specs)
        print(f"  epoch {epoch + 1}/{epochs}  train_loss={running / max(1, len(train_loader)):.4f}"
              f"  val_loss[{val_str}]")

    # Restore each head to its best-validation epoch.
    for s in specs:
        probe.heads[s.name].load_state_dict(best_state[s.name])
    print(f"best epochs per head: {best_epoch}")
    return best_epoch


@torch.no_grad()
def evaluate(model, loader: DataLoader, specs: Sequence[TaskSpec],
             audio_format: str, device: str) -> Dict[str, Dict[str, float]]:
    model.eval()
    metrics = build_metrics([s.name for s in specs])
    for src, rir, labels, noise in loader:
        src, rir = src.to(device), rir.to(device)
        noise = noise.to(device) if noise is not None else None
        x = batch_process(src, rir, audio_format, noise)
        out = model(x)
        for s in specs:
            metrics[s.name].update(out[s.name], labels[s.name])
    return {name: m.compute() for name, m in metrics.items()}


# =============================================================================
# Entry point
# =============================================================================

def save_checkpoint(probe: Probe, path: Path, backbone: str,
                    tasks: Sequence[str], seed: int) -> None:
    """Save the trained heads (not the frozen backbone) plus identifying metadata."""
    torch.save({
        "backbone": backbone,
        "tasks": list(tasks),
        "seed": seed,
        "heads": {name: head.state_dict() for name, head in probe.heads.items()},
    }, path)


def load_checkpoint(probe: Probe, path: Path, tasks: Sequence[str], device: str) -> None:
    """Load head weights saved by :func:`save_checkpoint` into ``probe``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing = [t for t in tasks if t not in ckpt.get("heads", {})]
    if missing:
        raise SystemExit(f"Checkpoint {path} has no head(s) for {missing}.")
    for name in tasks:
        probe.heads[name].load_state_dict(ckpt["heads"][name])


def build_model(name: str, tasks: Sequence[str], use_mlp_head: bool, seed: int):
    """Return ``(model, audio_format, sample_rate)``.

    ``naive_random`` builds the training-free normalization baseline; any other name
    builds a frozen backbone from the registry wrapped in a probe head.
    """
    if name.startswith("naive_"):
        if name != "naive_random":
            raise ValueError(f"the only naive baseline is 'naive_random', got {name!r}")
        return NaivePredictor(tasks, seed=seed), None, None
    backbone = build_backbone(name)
    probe = Probe(backbone, tasks, use_mlp_head=use_mlp_head)
    return probe, backbone.audio_format, backbone.sample_rate


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate SARL probes.")
    p.add_argument("--backbone", required=True,
                   help="Registered backbone name, or naive_random.")
    p.add_argument("--tasks", nargs="+", required=True,
                   help="Tasks to probe (one family: source or room).")
    p.add_argument("--data_root", required=True, help="Dataset root directory.")
    p.add_argument("--audio_format", default=None,
                   help="Override audio format (required for naive_random).")
    p.add_argument("--sample_rate", type=int, default=None, help="Override sample rate.")
    p.add_argument("--duration", type=float, default=10.0, help="Clip length in seconds.")
    p.add_argument("--samples_train", type=int, default=15000)
    p.add_argument("--samples_val", type=int, default=3000)
    p.add_argument("--samples_test", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--sigma_bins", type=float, default=2.0,
                   help="Soft-label Gaussian width in bins.")
    p.add_argument("--no_soft_labels", action="store_true",
                   help="Use hard bin targets instead of Gaussian soft labels.")
    p.add_argument("--use_mlp_head", action="store_true")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training and load trained heads from --ckpt.")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint path (default: <out_dir>/<backbone>_<tasks>_seed<seed>.ckpt).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    specs: List[TaskSpec] = resolve_tasks(args.tasks)
    family = specs[0].family

    model, fmt, sr = build_model(args.backbone, args.tasks, args.use_mlp_head, args.seed)
    audio_format = args.audio_format or fmt
    sample_rate = args.sample_rate or sr
    if audio_format is None or sample_rate is None:
        raise SystemExit("--audio_format and --sample_rate are required for naive_random.")
    model.to(args.device)

    trainable = getattr(model, "trainable", True)
    print(f"backbone={args.backbone}  family={family.value}  tasks={','.join(args.tasks)}")
    print(f"audio_format={audio_format}  sample_rate={sample_rate}  trainable={trainable}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.backbone}_{'-'.join(args.tasks)}_seed{args.seed}"
    ckpt_path = Path(args.ckpt) if args.ckpt else out_dir / f"{stem}.ckpt"

    if trainable and args.eval_only:
        load_checkpoint(model, ckpt_path, args.tasks, args.device)
        print(f"loaded heads from {ckpt_path}")
    elif trainable:
        # Loader seeds: train=seed, val=seed+1 (test=seed+2 below).
        train_loader = make_loader(
            args.data_root, "train", args.tasks, audio_format, args.duration,
            sample_rate, args.batch_size, args.seed, args.samples_train,
            num_workers=args.num_workers, shuffle=True)
        val_loader = make_loader(
            args.data_root, "val", args.tasks, audio_format, args.duration,
            sample_rate, args.batch_size, args.seed + 1, args.samples_val,
            num_workers=args.num_workers, shuffle=False)
        t0 = time.time()
        train(model, train_loader, val_loader, specs, audio_format, args.epochs,
              args.lr, args.weight_decay, args.grad_clip, not args.no_soft_labels,
              args.sigma_bins, args.device)
        print(f"trained in {time.time() - t0:.1f}s")
        save_checkpoint(model, ckpt_path, args.backbone, args.tasks, args.seed)
        print(f"saved heads to {ckpt_path}")

    test_loader = make_loader(
        args.data_root, "test", args.tasks, audio_format, args.duration,
        sample_rate, args.batch_size, args.seed + 2, args.samples_test,
        num_workers=args.num_workers, shuffle=False)
    scores = evaluate(model, test_loader, specs, audio_format, args.device)

    # Report.
    print("\n=== test scores ===")
    for name in args.tasks:
        fields = "  ".join(f"{k}={v:.4f}" for k, v in scores[name].items())
        print(f"  {name:<10} {fields}")

    result = {
        "backbone": args.backbone,
        "family": family.value,
        "tasks": list(args.tasks),
        "seed": args.seed,
        "scores": scores,
    }
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
