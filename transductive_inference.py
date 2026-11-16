"""
Transductive, inference-time refinement of a frozen PTv3 nuScenes-lidarseg model.

The model (RegisTrainSegmentor / PT-v3m1, see
configs/nuscenes/semseg-pt-v3m1-0-gfsregistrain_k5.py) is already trained.
No weights are changed, no backward pass, no optimizer. Per val scan
(processed independently, no temporal / cross-scan edges) we:

  1. Run one frozen forward pass to get softmax probabilities P and the
     penultimate per-point features F (captured via a forward_pre_hook on
     the final classifier heads).
  2. Build a sparse, geometry-topology / feature-weighted graph over the
     scan's points (k-NN on xyz, self-tuning Gaussian weights on F).
  3. Run confidence-aware label propagation (~20 iterations) to get refined
     probabilities Z, only mixing uncertain points with their neighbors.
  4. Compare argmax(P) vs argmax(Z) against GT with per-class IoU.

Repo facts this script relies on (see conversation / code for verification):
  - RegisTrainSegmentor.forward (pointcept/models/default.py) runs the PTv3
    backbone, whose decoder unpools back to the SAME point count as the
    input (point-resolution, not voxel-resolution) -> point.feat [N, 64].
    Two parallel heads consume that feature: seg_head (64->12 base,
    nn.Linear) and novel_seg_head (64->4 novel, nn.Linear, bias=False).
    seg_logits = cat([seg_head(feat), novel_seg_head(feat)], dim=1) [N, 16].
  - Checkpoints saved by CheckpointSaver_MultiRegis store a DDP-wrapped
    state_dict (all keys prefixed "module."), regardless of how many GPUs
    this script itself uses.
  - NuScenesDataset_TEST (data.val in the gfsregistrain config) already
    produces point-aligned coord/feat/segment: GridSample(mode="train")
    picks one representative point per voxel and applies that SAME index
    selection to coord, feat and segment, so there is no voxel->raw-point
    "inverse" remap in this val pipeline (unlike GFSSemSegTester's
    fragment-based precise-eval path). We still handle "inverse" if a
    future config provides it, mirroring the exact pattern used in
    pointcept/engines/test.py::GFSSemSegTester.one_regis_test.
  - segment labels are in the unified 16-class space
    (NUSCENES_BASE_NAMES + NUSCENES_NOVEL_NAMES, i.e. cfg.data.names),
    ignore_index=-1 for noise/unannotated points. There is no separate
    17th "noise" logit; the model never predicts noise.
  - intersection_and_union_gpu (pointcept/utils/misc.py) is reused as-is
    for IoU accounting, matching GFS_SemSegEvaluator / GFSSemSegFastTester.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F_
from torch_cluster import knn_graph

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pointcept.utils.config import Config
from pointcept.utils.env import set_seed
from pointcept.utils.misc import intersection_and_union_gpu
from pointcept.models import build_model
from pointcept.datasets import build_dataset, collate_fn


@dataclass
class TransductiveConfig:
    """CLI-exposed knobs for graph-propagation refinement."""

    config_file: str
    checkpoint: str
    k: int = 8
    alpha_base: float = 0.99999
    num_iters: int = 20
    ignore_index: int | None = None  # None -> resolved from cfg.data.ignore_index
    num_classes: int | None = None  # None -> resolved from cfg.data.num_base_novels
    device: str = "cuda"
    seed: int = 42
    limit: int | None = None


def parse_args() -> TransductiveConfig:
    """Parse CLI args into a TransductiveConfig.

    num_classes / ignore_index default to whatever the loaded model config
    specifies (cfg.data.num_base_novels / cfg.data.ignore_index) once the
    config file is read; the CLI values below are only used if the user
    explicitly overrides them.
    """
    parser = argparse.ArgumentParser(
        description="Transductive graph-propagation refinement for a frozen "
        "PTv3 nuScenes-lidarseg model (no weight updates)."
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
    parser.add_argument("--k", type=int, default=8, help="k for the spatial k-NN graph.")
    parser.add_argument(
        "--alpha-base",
        type=float,
        default=0.99999,
        help="Max mixing coefficient (applied when confidence is 0).",
    )
    parser.add_argument(
        "--num-iters", type=int, default=20, help="Propagation iterations."
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
        help="Override number of classes (default: taken from config, normally 16). "
        "Must match the checkpoint's head widths -- only override if you know "
        "the checkpoint was trained with a different class count.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N val scans (fast iteration).",
    )
    args = parser.parse_args()

    return TransductiveConfig(
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        k=args.k,
        alpha_base=args.alpha_base,
        num_iters=args.num_iters,
        ignore_index=args.ignore_index,  # None unless user overrides
        num_classes=args.num_classes,  # None unless user overrides
        device=args.device,
        seed=args.seed,
        limit=args.limit,
    )


# ---------------------------------------------------------------------------
# Model / checkpoint loading
# ---------------------------------------------------------------------------


def load_model(cfg: Config, checkpoint_path: str, device: str) -> torch.nn.Module:
    """Build RegisTrainSegmentor from cfg.model and load a checkpoint.

    Mirrors GFS_VL_Trainer.build_model / TesterBase.build_model
    (pointcept/engines/train.py, pointcept/engines/test.py): checkpoints
    saved under DDP have all keys prefixed "module.", which we strip since
    this script always runs the model un-wrapped (single process, no DDP).
    """
    model = build_model(cfg.model).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = OrderedDict()
    for key, value in ckpt["state_dict"].items():
        if key.startswith("module."):
            key = key[len("module.") :]
        state_dict[key] = value
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, (
        f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
    )
    print(f"[load_model] Loaded checkpoint '{checkpoint_path}' (epoch {ckpt.get('epoch')})")

    for p in model.parameters():
        p.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Frozen forward pass + penultimate-feature capture
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_scan_outputs(
    model: torch.nn.Module, batch: dict, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one frozen forward pass on a single scan.

    Args:
        model: eval-mode RegisTrainSegmentor.
        batch: a single-scan batch from the val collate_fn (coord, grid_coord,
            feat, segment, offset all present; offset == [N] since batch size
            is 1 scan).
        device: torch device string.

    Returns:
        xyz: [N, 3] raw point coordinates (float32, on device).
        feat: [N, d] penultimate per-point features, input to the final
            classifier heads (captured via forward_pre_hook on seg_head).
        prob: [N, C] softmax probabilities over the C classes.
        segment: [N] GT labels (ignore_index where unannotated).
    """
    captured: dict[str, torch.Tensor] = {}

    def _capture_pre_hook(module, inputs):
        # seg_head and novel_seg_head both consume the identical penultimate
        # feature tensor (pointcept/models/default.py:117-118); hooking
        # either one captures it. We hook seg_head.
        captured["feat"] = inputs[0]

    handle = model.seg_head.register_forward_pre_hook(_capture_pre_hook)
    try:
        input_dict = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        output_dict = model(input_dict)
    finally:
        handle.remove()

    assert "feat" in captured, (
        "forward_pre_hook on model.seg_head did not fire -- "
        "the model's forward() path changed; re-inspect pointcept/models/default.py."
    )

    seg_logits = output_dict["seg_logits"]  # [N, C]
    prob = F_.softmax(seg_logits.float(), dim=-1)
    xyz = input_dict["coord"].float()
    feat = captured["feat"].float()
    segment = input_dict["segment"]

    assert xyz.shape[0] == feat.shape[0] == prob.shape[0] == segment.shape[0], (
        f"Point count mismatch: xyz={xyz.shape[0]}, feat={feat.shape[0]}, "
        f"prob={prob.shape[0]}, segment={segment.shape[0]}"
    )
    return xyz, feat, prob, segment


# ---------------------------------------------------------------------------
# Sparse geometry-topology / feature-weighted graph
# ---------------------------------------------------------------------------


def build_graph(xyz: torch.Tensor, feat: torch.Tensor, k: int) -> torch.Tensor:
    """Build a symmetric-normalized sparse propagation matrix S.

    Topology comes from k-NN on xyz; edge weights come from a self-tuning
    (local-scale) Gaussian kernel on L2-normalized feat, following Zelnik-
    Manor & Perona's self-tuning spectral clustering. Since the graph
    topology here is built with a single k (there is no separate
    feature-space k-NN pass specified), sigma_i is computed as the max
    feature-space distance among that SAME k-neighbor set (i.e. the
    farthest of the k spatial neighbors already selected), not a second
    independent k-NN search.

    Args:
        xyz: [N, 3] point coordinates, on GPU.
        feat: [N, d] penultimate features, on GPU.
        k: number of spatial neighbors.

    Returns:
        S: [N, N] sparse_coo_tensor, symmetric-normalized (D^-1/2 W D^-1/2).
    """
    n = xyz.shape[0]
    device = xyz.device

    # torch_cluster.knn_graph returns edge_index = [neighbor, center]: for
    # each center point (edge_index[1]), it appears exactly k times, paired
    # with its k nearest neighbors (edge_index[0]). We rely on that to get
    # exactly k distances per node when computing sigma below. Direction is
    # otherwise irrelevant since W is symmetrized right after.
    edge_index = knn_graph(xyz, k=k, loop=False)  # [2, E]
    nbr, center = edge_index[0], edge_index[1]

    feat_n = F_.normalize(feat, p=2, dim=-1, eps=1e-8)
    diff = feat_n[nbr] - feat_n[center]
    sq_dist = (diff * diff).sum(dim=-1)  # [E] = ||f_i - f_j||^2

    # sigma_c = feature-space distance from c to its k-th (i.e. farthest of
    # its k) spatial neighbor. Scatter over `center`, which appears exactly
    # k times per node, so every node gets a well-defined sigma.
    dist = sq_dist.clamp_min(0).sqrt()
    sigma = torch.zeros(n, device=device, dtype=feat.dtype)
    sigma.scatter_reduce_(0, center, dist, reduce="amax", include_self=False)
    sigma = sigma.clamp_min(1e-8)  # guard against 0 (identical features)

    denom = sigma[nbr] * sigma[center]
    weight = torch.exp(-sq_dist / denom)

    w = torch.sparse_coo_tensor(edge_index, weight, (n, n))
    w_t = torch.sparse_coo_tensor(edge_index.flip(0), weight, (n, n))
    w_sym = (w + w_t).coalesce()
    w_sym = torch.sparse_coo_tensor(
        w_sym.indices(), w_sym.values() * 0.5, (n, n)
    ).coalesce()

    degree = torch.sparse.sum(w_sym, dim=1).to_dense().clamp_min(1e-8)
    d_inv_sqrt = degree.pow(-0.5)
    idx = w_sym.indices()
    norm_values = w_sym.values() * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
    s = torch.sparse_coo_tensor(idx, norm_values, (n, n)).coalesce()
    return s


# ---------------------------------------------------------------------------
# Confidence-aware propagation
# ---------------------------------------------------------------------------


def propagate(
    prob: torch.Tensor, s: torch.Tensor, alpha_base: float, num_iters: int
) -> torch.Tensor:
    """Confidence-aware sparse label propagation. Refinement touches only Z;
    nothing here flows back into the model.

    Args:
        prob: [N, C] initial (frozen-model) softmax probabilities P.
        s: [N, N] sparse, symmetric-normalized propagation matrix.
        alpha_base: max per-point mixing coefficient (at zero confidence).
        num_iters: number of propagation iterations.

    Returns:
        z: [N, C] refined probabilities.
    """
    confidence = prob.max(dim=-1).values  # [N]
    alpha = (alpha_base * (1.0 - confidence)).unsqueeze(-1)  # [N, 1]

    z = prob
    for _ in range(num_iters):
        z = (1.0 - alpha) * prob + alpha * torch.sparse.mm(s, z)
    return z


# ---------------------------------------------------------------------------
# Per-scan refinement
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    labels_before: torch.Tensor
    labels_after: torch.Tensor
    segment: torch.Tensor
    max_prob: torch.Tensor


def refine_scan(
    model: torch.nn.Module, batch: dict, cfg: TransductiveConfig, device: str
) -> ScanResult:
    """Run the full per-scan pipeline: forward -> graph -> propagate -> decode."""
    xyz, feat, prob, segment = extract_scan_outputs(model, batch, device)

    s = build_graph(xyz, feat, cfg.k)
    z = propagate(prob, s, cfg.alpha_base, cfg.num_iters)

    labels_before = prob.argmax(dim=-1)
    labels_after = z.argmax(dim=-1)

    # If a config ever supplies a voxel->raw-point "inverse" mapping (as
    # GFSSemSegTester's fragment path does; data.val here does not), apply
    # the SAME mapping to both before/after and to segment so the
    # before/after comparison stays apples-to-apples.
    if "inverse" in batch:
        inverse = batch["inverse"].to(device)
        labels_before = labels_before[inverse]
        labels_after = labels_after[inverse]
        segment = batch["origin_segment"].to(device)

    return ScanResult(
        labels_before=labels_before,
        labels_after=labels_after,
        segment=segment,
        max_prob=prob.max(dim=-1).values,
    )


# ---------------------------------------------------------------------------
# Main: val loop, accumulation, reporting
# ---------------------------------------------------------------------------


def main() -> None:
    cli_cfg = parse_args()
    set_seed(cli_cfg.seed)

    device = cli_cfg.device
    assert torch.cuda.is_available() or device == "cpu", (
        "CUDA requested but not available; pass --device cpu explicitly if intended "
        "(the graph/propagation ops are written for GPU sparse tensors)."
    )

    cfg = Config.fromfile(cli_cfg.config_file)

    # num_classes / ignore_index: prefer the loaded config's values (they
    # define the checkpoint's architecture / label space) unless the user
    # explicitly passed a CLI override.
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

    print(
        f"[main] k={cli_cfg.k} alpha_base={cli_cfg.alpha_base} "
        f"num_iters={cli_cfg.num_iters} num_classes={num_classes} "
        f"ignore_index={ignore_index} device={device} seed={cli_cfg.seed}"
    )

    model = load_model(cfg, cli_cfg.checkpoint, device)

    val_dataset = build_dataset(cfg.data.val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # process each scan independently, no cross-scan edges
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    n_scans = len(val_loader) if cli_cfg.limit is None else min(cli_cfg.limit, len(val_loader))
    print(f"[main] Evaluating on {n_scans} val scans (dataset has {len(val_loader)}).")

    inter_before = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union_before = torch.zeros(num_classes, dtype=torch.float64, device=device)
    inter_after = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union_after = torch.zeros(num_classes, dtype=torch.float64, device=device)

    conf_hist = torch.zeros(10, dtype=torch.float64)  # 10 bins over [0, 1]
    n_points_total = 0
    n_changed_total = 0

    for i, batch in enumerate(val_loader):
        if cli_cfg.limit is not None and i >= cli_cfg.limit:
            break

        result = refine_scan(model, batch, cli_cfg, device)

        ib, ub, _ = intersection_and_union_gpu(
            result.labels_before.clone(), result.segment.clone(), num_classes, ignore_index
        )
        ia, ua, _ = intersection_and_union_gpu(
            result.labels_after.clone(), result.segment.clone(), num_classes, ignore_index
        )
        inter_before += ib
        union_before += ub
        inter_after += ia
        union_after += ua

        conf_hist += torch.histc(
            result.max_prob.double().cpu(), bins=10, min=0.0, max=1.0
        )
        n_points_total += result.labels_before.numel()
        n_changed_total += (result.labels_after != result.labels_before).sum().item()

        if (i + 1) % 50 == 0 or (i + 1) == n_scans:
            print(f"[main] Eval: [{i + 1}/{n_scans}]")

    # ---- mIoU before/after ----
    iou_before = (inter_before / union_before.clamp_min(1e-10)).cpu().numpy()
    iou_after = (inter_after / union_after.clamp_min(1e-10)).cpu().numpy()
    valid = (union_before.cpu().numpy() > 0) | (union_after.cpu().numpy() > 0)

    miou_before = float(np.mean(iou_before[valid]))
    miou_after = float(np.mean(iou_after[valid]))
    print(
        f"\n===== Overall mIoU: BEFORE={miou_before:.4f}  AFTER={miou_after:.4f}  "
        f"DELTA={miou_after - miou_before:+.4f} ====="
    )

    # ---- per-class table sorted by delta ----
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

    # ---- sanity logging ----
    conf_hist_frac = (conf_hist / conf_hist.sum().clamp_min(1)).numpy()
    print("\n===== Sanity: max_c P_ic histogram (10 bins over [0,1]) =====")
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        print(f"  [{lo:.1f}, {hi:.1f}): {conf_hist_frac[b]:.4f}")

    change_frac = n_changed_total / max(n_points_total, 1)
    print(
        f"\n===== Sanity: label changed on {n_changed_total}/{n_points_total} "
        f"points ({change_frac:.4%}) ====="
    )


if __name__ == "__main__":
    main()
