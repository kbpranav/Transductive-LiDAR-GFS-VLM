"""
Test-time adaptation baselines for a frozen PTv3 nuScenes-lidarseg model,
evaluated on the same 6k unlabeled val scans / 69% frozen-baseline used by
transductive_inference.py.

Unlike transductive_inference.py (fully frozen, per-scan, model state never
touched), the two methods here DO mutate model state -- BatchNorm running
stats and/or a small parameter subset -- using the val scans as the
unlabeled *target* set to adapt to, then re-running inference on those same
scans. To guarantee the on-disk checkpoint is never modified and the two
methods never contaminate each other, every experiment below reloads its own
fresh model straight from the checkpoint (transductive_inference.load_model
only ever reads the .pth; nothing here writes to it).

This backbone mixes nn.BatchNorm1d (stem + grid-pool/unpool projections) and
nn.LayerNorm (attention blocks) -- see
pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py -- and
also has Dropout/DropPath inside the attention blocks (drop_path=0.3 by
default). That matters for both methods below:

  1. bn_adapt_inference -- BatchNorm running-stat adaptation. No gradients,
     no weight updates. Only BatchNorm1d submodules are flipped to
     .train() (the rest of the model, including dropout/droppath, stays in
     .eval() so no extra stochasticity leaks into the stat collection);
     each BatchNorm1d's momentum is set to None first so PyTorch accumulates
     a cumulative moving average over every scan seen instead of the
     default recency-weighted EMA (momentum=0.1 would mostly forget
     everything but the last few hundred single-scan batches).

  2. entropy_min_inference -- Tent-style (Wang et al., 2021) entropy
     minimization. Only the affine (weight/bias) parameters of norm layers
     (BatchNorm1d + LayerNorm) get requires_grad=True; every other
     parameter stays frozen -- updating the full backbone with entropy loss
     alone (no labels) is known to collapse predictions into whichever
     class trivially minimizes entropy. The model is kept in .eval() mode
     throughout (BN uses its original running stats, dropout/droppath are
     off), so this experiment isolates the effect of entropy-driven affine
     tuning from the BN-stat effect measured separately in
     bn_adapt_inference above.

Both methods run as two clean phases -- adapt over the full target pool,
then a separate eval pass over the same pool -- rather than the original
Tent paper's fully online single-batch-then-predict loop; that matches the
adapt-then-re-run-inference structure requested for this comparison.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pointcept.utils.config import Config
from pointcept.utils.env import set_seed
from pointcept.utils.misc import intersection_and_union_gpu
from pointcept.datasets import build_dataset, collate_fn

from transductive_inference import load_model


@dataclass
class AdaptConfig:
    """CLI-exposed knobs for the test-time adaptation experiments."""

    config_file: str
    checkpoint: str
    method: str  # "bn_adapt", "entropy_min", or "both"
    bn_num_passes: int = 1
    entropy_lr: float = 1e-3
    entropy_num_epochs: int = 1
    ignore_index: int | None = None  # None -> resolved from cfg.data.ignore_index
    num_classes: int | None = None  # None -> resolved from cfg.data.num_base_novels
    device: str = "cuda"
    seed: int = 42
    limit: int | None = None
    skip_baseline_pass: bool = False


def parse_args() -> AdaptConfig:
    parser = argparse.ArgumentParser(
        description="BN-adaptation / entropy-minimization test-time adaptation "
        "for a frozen PTv3 nuScenes-lidarseg model."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/nuscenes/semseg-pt-v3m1-0-gfsregistrain_k5.py",
        help="Pointcept config defining model + data.val.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint, e.g. "
        "exp/nuscenes/pt_v3_nuscenes_registrain_k5_newnovel3/model/"
        "regis2_model_best.pth",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bn_adapt", "entropy_min", "both"],
        default="both",
        help="Which adaptation experiment to run.",
    )
    parser.add_argument(
        "--bn-num-passes",
        type=int,
        default=1,
        help="Number of full passes over the target scans for BN stat adaptation.",
    )
    parser.add_argument(
        "--entropy-lr", type=float, default=1e-3, help="Adam LR for entropy minimization."
    )
    parser.add_argument(
        "--entropy-num-epochs",
        type=int,
        default=1,
        help="Number of full passes over the target scans for entropy minimization.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=None,
        help="Override ignore index (default: taken from config, normally -1).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override number of classes (default: taken from config, normally 16).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N val scans (fast iteration).",
    )
    parser.add_argument(
        "--skip-baseline-pass",
        action="store_true",
        help="Skip the frozen 'before' eval pass and only report the post-adaptation "
        "mIoU. Useful once you've already confirmed the baseline matches the known "
        "69%% number and don't want to re-pay that pass on every trial.",
    )
    args = parser.parse_args()

    return AdaptConfig(
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        method=args.method,
        bn_num_passes=args.bn_num_passes,
        entropy_lr=args.entropy_lr,
        entropy_num_epochs=args.entropy_num_epochs,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
        device=args.device,
        seed=args.seed,
        limit=args.limit,
        skip_baseline_pass=args.skip_baseline_pass,
    )


# ---------------------------------------------------------------------------
# Shared forward / eval helpers
# ---------------------------------------------------------------------------


def _to_device(batch: dict, device: str) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def _eval_scan(model: torch.nn.Module, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """One frozen forward pass -> (predicted labels, GT segment), inverse-mapped
    back to raw points if the val pipeline provides an "inverse" index (mirrors
    refine_scan in transductive_inference.py)."""
    input_dict = _to_device(batch, device)
    seg_logits = model(input_dict)["seg_logits"]
    labels = seg_logits.argmax(dim=-1)
    segment = input_dict["segment"]

    if "inverse" in batch:
        inverse = batch["inverse"].to(device)
        labels = labels[inverse]
        segment = batch["origin_segment"].to(device)

    return labels, segment


def run_eval_loop(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    ignore_index: int,
    limit: int | None,
    device: str,
    log_prefix: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Full pass over the val loader computing per-class IoU / mIoU. Puts the
    model in eval mode first (safe no-op if it's already there)."""
    model.eval()
    inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=device)
    n = len(loader) if limit is None else min(limit, len(loader))

    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        labels, segment = _eval_scan(model, batch, device)
        it, un, _ = intersection_and_union_gpu(labels.clone(), segment.clone(), num_classes, ignore_index)
        inter += it
        union += un
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"[{log_prefix}] Eval: [{i + 1}/{n}]")

    iou = (inter / union.clamp_min(1e-10)).cpu().numpy()
    valid = union.cpu().numpy() > 0
    miou = float(np.mean(iou[valid])) if valid.any() else float("nan")
    return miou, iou, valid


# ---------------------------------------------------------------------------
# 1) BN adaptation
# ---------------------------------------------------------------------------


def bn_adapt_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    num_passes: int,
    limit: int | None,
) -> torch.nn.Module:
    """Mutate `model`'s BatchNorm1d running stats in place using the target
    scans. No gradients, no weight updates -- only running_mean/running_var
    move.

    Only BatchNorm1d submodules are switched to .train(); everything else
    (including the attention blocks' Dropout/DropPath) stays in .eval() so
    the stat collection isn't polluted by dropout noise.
    """
    model.eval()
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert bn_layers, "No BatchNorm1d layers found -- nothing for BN adaptation to touch."
    for bn in bn_layers:
        bn.momentum = None  # cumulative moving average instead of the default recency-weighted EMA
        bn.train()

    n = len(loader) if limit is None else min(limit, len(loader))
    with torch.no_grad():
        for p in range(num_passes):
            for i, batch in enumerate(loader):
                if limit is not None and i >= limit:
                    break
                model(_to_device(batch, device))
                if (i + 1) % 50 == 0 or (i + 1) == n:
                    print(f"[bn_adapt] stat pass {p + 1}/{num_passes}: [{i + 1}/{n}]")

    model.eval()  # BatchNorm1d.eval() too -- now uses the freshly-adapted running stats
    return model


# ---------------------------------------------------------------------------
# 2) Entropy minimization (Tent-style, affine params only)
# ---------------------------------------------------------------------------


def entropy_min_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    lr: float,
    num_epochs: int,
    limit: int | None,
) -> torch.nn.Module:
    """Mutate `model`'s norm-layer affine params in place via entropy-loss
    gradient steps on the target scans. Every other parameter stays frozen
    to avoid the well-known collapse mode of full-backbone entropy
    minimization without labels.
    """
    norm_types = (torch.nn.BatchNorm1d, torch.nn.LayerNorm)
    for p in model.parameters():
        p.requires_grad = False

    affine_params = []
    for m in model.modules():
        if isinstance(m, norm_types):
            if getattr(m, "weight", None) is not None:
                m.weight.requires_grad = True
                affine_params.append(m.weight)
            if getattr(m, "bias", None) is not None:
                m.bias.requires_grad = True
                affine_params.append(m.bias)
    assert affine_params, "No norm-layer affine params found -- nothing for entropy minimization to update."

    optimizer = torch.optim.Adam(affine_params, lr=lr)
    model.eval()  # BN keeps its original running stats; dropout/droppath stay off

    n = len(loader) if limit is None else min(limit, len(loader))
    for e in range(num_epochs):
        for i, batch in enumerate(loader):
            if limit is not None and i >= limit:
                break

            seg_logits = model(_to_device(batch, device))["seg_logits"]
            log_prob = F_.log_softmax(seg_logits.float(), dim=-1)
            entropy = -(log_prob.exp() * log_prob).sum(dim=-1).mean()

            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()

            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(
                    f"[entropy_min] epoch {e + 1}/{num_epochs}: [{i + 1}/{n}] "
                    f"entropy={entropy.item():.4f}"
                )

    for p in affine_params:
        p.requires_grad = False
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main: run the requested method(s), report before/after mIoU
# ---------------------------------------------------------------------------


def report(
    method: str,
    miou_before: float | None,
    iou_before: np.ndarray | None,
    valid_before: np.ndarray | None,
    miou_after: float,
    iou_after: np.ndarray,
    valid_after: np.ndarray,
    num_classes: int,
    class_names: list[str],
) -> None:
    """miou_before/iou_before/valid_before are None when --skip-baseline-pass
    was passed -- in that case we only report the post-adaptation numbers,
    with no before/delta columns."""
    if miou_before is None:
        print(f"\n===== [{method}] Overall mIoU: AFTER={miou_after:.4f} "
              f"(baseline pass skipped, no delta) =====")

        rows = [
            (class_names[c], iou_after[c])
            for c in range(num_classes)
            if valid_after[c]
        ]
        rows.sort(key=lambda r: r[1])  # ascending: worst classes first

        print(f"\n{'class':<22}{'IoU after':>12}")
        print("-" * 34)
        for name, after in rows:
            print(f"{name:<22}{after:>12.4f}")
        return

    valid = valid_before | valid_after
    print(
        f"\n===== [{method}] Overall mIoU: BEFORE={miou_before:.4f}  AFTER={miou_after:.4f}  "
        f"DELTA={miou_after - miou_before:+.4f} ====="
    )

    rows = []
    for c in range(num_classes):
        if not valid[c]:
            continue
        delta = iou_after[c] - iou_before[c]
        rows.append((class_names[c], iou_before[c], iou_after[c], delta))
    rows.sort(key=lambda r: r[3])  # ascending: worst regressions first

    print(f"\n{'class':<22}{'IoU before':>12}{'IoU after':>12}{'delta':>10}")
    print("-" * 56)
    for name, before, after, delta in rows:
        print(f"{name:<22}{before:>12.4f}{after:>12.4f}{delta:>+10.4f}")


def main() -> None:
    cli_cfg = parse_args()
    set_seed(cli_cfg.seed)

    device = cli_cfg.device
    assert torch.cuda.is_available() or device == "cpu", (
        "CUDA requested but not available; pass --device cpu explicitly if intended."
    )

    cfg = Config.fromfile(cli_cfg.config_file)

    num_classes = (
        cli_cfg.num_classes if cli_cfg.num_classes is not None else cfg.data.num_base_novels
    )
    ignore_index = (
        cli_cfg.ignore_index if cli_cfg.ignore_index is not None else cfg.data.ignore_index
    )
    class_names = list(cfg.data.names)
    assert len(class_names) == num_classes, (
        f"cfg.data.names has {len(class_names)} entries but num_classes={num_classes}"
    )

    val_dataset = build_dataset(cfg.data.val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # process each scan independently, matching transductive_inference.py
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    n_scans = len(val_loader) if cli_cfg.limit is None else min(cli_cfg.limit, len(val_loader))
    print(f"[main] method={cli_cfg.method} num_classes={num_classes} ignore_index={ignore_index} "
          f"device={device} seed={cli_cfg.seed}")
    print(f"[main] Evaluating on {n_scans} val scans (dataset has {len(val_loader)}).")

    methods = ["bn_adapt", "entropy_min"] if cli_cfg.method == "both" else [cli_cfg.method]

    for method in methods:
        print(f"\n{'=' * 20} Method: {method} {'=' * 20}")

        # Fresh model straight from the checkpoint for every experiment: the
        # checkpoint file itself is never written to (load_model only reads
        # it), and this guarantees bn_adapt / entropy_min never contaminate
        # each other's state.
        model = load_model(cfg, cli_cfg.checkpoint, device)

        if cli_cfg.skip_baseline_pass:
            miou_before, iou_before, valid_before = None, None, None
            print(f"[{method}] Skipping baseline pass (--skip-baseline-pass).")
        else:
            miou_before, iou_before, valid_before = run_eval_loop(
                model, val_loader, num_classes, ignore_index, cli_cfg.limit, device, f"{method}/before"
            )
            print(f"[{method}] BEFORE (frozen, no adaptation) mIoU={miou_before:.4f}")

        if method == "bn_adapt":
            model = bn_adapt_inference(
                model, val_loader, device, cli_cfg.bn_num_passes, cli_cfg.limit
            )
        elif method == "entropy_min":
            model = entropy_min_inference(
                model, val_loader, device, cli_cfg.entropy_lr, cli_cfg.entropy_num_epochs, cli_cfg.limit
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        miou_after, iou_after, valid_after = run_eval_loop(
            model, val_loader, num_classes, ignore_index, cli_cfg.limit, device, f"{method}/after"
        )

        report(
            method,
            miou_before, iou_before, valid_before,
            miou_after, iou_after, valid_after,
            num_classes, class_names,
        )

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
