"""
TransCLIP-style transductive refinement of a frozen PTv3 nuScenes-lidarseg
model.

This is a sibling of transductive_inference.py: same frozen model, same
per-scan forward pass + penultimate-feature capture, same eval harness.
Only the refinement step changes. Where transductive_inference.py runs a
hand-built confidence-gated Gaussian-kernel diffusion, this script ports the
core solver of TransCLIP (Zanella et al., NeurIPS 2024 --
gfs_vl/transduction-for-vlms/TransCLIP_solver/TransCLIP_utils.py) from its
original "many CLIP image embeddings vs. text-prototype classes" setting to
"many per-point PTv3 features vs. segmentation classes", one scan at a time.

Concept mapping from TransCLIP (VLM/CLIP) to here (PTv3/lidar):
  - query_features (CLIP image embeddings, L2-normalized)
      -> penultimate per-point features F [N, 64], L2-normalized here too,
         captured via the same forward_pre_hook on seg_head used by
         transductive_inference.extract_scan_outputs.
  - clip_prototypes / cosine-similarity-to-text-embedding logits
      -> not used. Our seg_head/novel_seg_head are already a trained linear
         classifier, not a zero-shot cosine-similarity proxy, so we plug the
         model's own softmax(seg_logits) in directly as TransCLIP's
         "initial_predictions" (the "On-top" mode in TransCLIP's main.py),
         skipping the clip_prototypes machinery entirely.
  - few-shot "shots" / support set + gamma sweep
      -> not used. GFS-VL's novel classes were already few-shot-registered
         at train time (baked into the checkpoint); every scan is run
         through TransCLIP's support_features=None ("zero-shot") branch.
  - dataset-level query set (thousands of images, dense O(N^2) CPU affinity
    is fine)
      -> one scan's points (tens of thousands). Dense O(N^2) affinity is
         infeasible, so build_affinity_matrix here uses torch_cluster's
         knn_graph in feature space (cosine similarity, since features are
         L2-normalized) instead of a dense topk over a full similarity
         matrix. The resulting W has the same semantics as TransCLIP's W
         (raw cosine-similarity edge weights, asymmetric topology, no
         degree normalization) -- just built efficiently.
  - K classes, dataset-fixed
      -> num_classes from cfg.data.num_base_novels (16), same as
         transductive_inference.py.

TransCLIP's temperature (hardcoded 50 in the original), lambda_value,
n_neighbors, max_iter and std_init are tuned for CLIP's cosine-similarity
feature scale (~512-768d, values in [-1, 1]). PTv3's feature scale/dim (64)
differs, so all of these are exposed as CLI flags here rather than
hardcoded, to be retuned empirically.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_
from torch_cluster import knn_graph

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pointcept.utils.config import Config
from pointcept.utils.env import set_seed
from pointcept.utils.misc import intersection_and_union_gpu
from pointcept.datasets import build_dataset, collate_fn

from transductive_inference import load_model, extract_scan_outputs, ScanResult


@dataclass
class TransCLIPConfig:
    """CLI-exposed knobs for the TransCLIP-style solver."""

    config_file: str
    checkpoint: str
    n_neighbors: int = 3
    temperature: float = 50.0
    lambda_value: float = 1.0
    max_iter: int = 10
    inner_iters: int = 5
    n_most_confident: int = 8
    std_init: float | None = None  # None -> resolved as 1/d once d is known
    chunk_size: int = 4096
    ignore_index: int | None = None  # None -> resolved from cfg.data.ignore_index
    num_classes: int | None = None  # None -> resolved from cfg.data.num_base_novels
    device: str = "cuda"
    seed: int = 42
    limit: int | None = None


def parse_args() -> TransCLIPConfig:
    parser = argparse.ArgumentParser(
        description="TransCLIP-style transductive refinement for a frozen "
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
    parser.add_argument(
        "--n-neighbors", type=int, default=3, help="k for the feature-space kNN affinity graph."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=50.0,
        help="TransCLIP's temperature constant (scales the graph term and the "
        "z-update softmax).",
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=1.0,
        help="Exponent applied to y_hat in the z-update (regularization strength "
        "toward the initial prediction).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=10, help="Outer iterations (z/mu/sigma alternation)."
    )
    parser.add_argument(
        "--inner-iters", type=int, default=5, help="Inner z-update sub-iterations per outer step."
    )
    parser.add_argument(
        "--n-most-confident",
        type=int,
        default=8,
        help="Top-k most-confident points per class used to init mu.",
    )
    parser.add_argument(
        "--std-init",
        type=float,
        default=None,
        help="Initial (shared, diagonal) covariance value (default: 1/d).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size for the GMM likelihood / sigma-update point loops.",
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
    args = parser.parse_args()

    return TransCLIPConfig(
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        n_neighbors=args.n_neighbors,
        temperature=args.temperature,
        lambda_value=args.lambda_value,
        max_iter=args.max_iter,
        inner_iters=args.inner_iters,
        n_most_confident=args.n_most_confident,
        std_init=args.std_init,
        chunk_size=args.chunk_size,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
        device=args.device,
        seed=args.seed,
        limit=args.limit,
    )


# ---------------------------------------------------------------------------
# Shared-covariance diagonal Gaussian over classes (TransCLIP's `Gaussian`)
# ---------------------------------------------------------------------------


class DiagGaussian(nn.Module):
    """K diagonal Gaussians sharing one covariance, over L2-normalized features.

    mu: [K, d] class centroids (unit-norm). std: [d] shared diagonal variance.
    """

    def __init__(self, mu: torch.Tensor, std: torch.Tensor, chunk_size: int):
        super().__init__()
        self.mu = mu
        self.std = std
        self.chunk_size = chunk_size

    def log_likelihood(self, feat: torch.Tensor) -> torch.Tensor:
        """[N, d] -> [N, K] unnormalized log-likelihood (-0.5 * (x-mu)^2/std)."""
        n = feat.shape[0]
        k = self.mu.shape[0]
        out = torch.empty((n, k), dtype=feat.dtype, device=feat.device)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            diff = feat[start:end, None, :] - self.mu[None, :, :]  # [n_chunk, K, d]
            out[start:end] = -0.5 * torch.einsum("ijk,ijk->ij", diff * diff, 1.0 / self.std[None, None, :])
        return out


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def init_mu(feat: torch.Tensor, z: torch.Tensor, num_classes: int, n_most_confident: int) -> torch.Tensor:
    """Per-class centroid from the n_most_confident points per class (unit-norm)."""
    d = feat.shape[1]
    mu = torch.zeros(num_classes, d, device=feat.device, dtype=feat.dtype)
    topk_indices = torch.topk(z, k=min(n_most_confident, feat.shape[0]), dim=0).indices  # [k, K]
    for c in range(num_classes):
        idx = topk_indices[:, c]
        weights = z[idx, c].unsqueeze(-1)
        mu[c] = (feat[idx] * weights).mean(dim=0)
    mu = F_.normalize(mu, p=2, dim=-1, eps=1e-8)
    return mu


def init_sigma(d: int, std_init: float, device, dtype) -> torch.Tensor:
    return torch.full((d,), std_init, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Feature-space affinity graph (TransCLIP's build_affinity_matrix, efficient)
# ---------------------------------------------------------------------------


def build_affinity_matrix(feat: torch.Tensor, n_neighbors: int) -> torch.Tensor:
    """Sparse [N, N] cosine-similarity kNN graph in (L2-normalized) feature space.

    Matches TransCLIP's W: raw cosine-similarity edge weights, asymmetric
    topology (row i's neighbors need not have i as a neighbor), no degree
    normalization. Built via torch_cluster.knn_graph instead of a dense
    NxN topk, since N here is a scan's point count (tens of thousands),
    not an image count.
    """
    n = feat.shape[0]
    edge_index = knn_graph(feat, k=n_neighbors, loop=False)  # [2, E] = [neighbor, center]
    nbr, center = edge_index[0], edge_index[1]
    weight = (feat[center] * feat[nbr]).sum(dim=-1)  # cosine sim (feat already unit-norm)
    w = torch.sparse_coo_tensor(torch.stack([center, nbr]), weight, (n, n)).coalesce()
    return w


# ---------------------------------------------------------------------------
# Alternating z / mu / sigma updates (TransCLIP's update_z/update_mu/update_sigma)
# ---------------------------------------------------------------------------


def update_z(
    gmm_log_likelihood: torch.Tensor,
    y_hat: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    lambda_value: float,
    n_neighbors: int,
    temperature: float,
    inner_iters: int,
) -> torch.Tensor:
    z = z.clone()
    graph_scale = temperature / (n_neighbors * 2)
    for _ in range(inner_iters):
        intermediate = gmm_log_likelihood.clone()
        intermediate += graph_scale * (torch.sparse.mm(w.t(), z) + torch.sparse.mm(w, z))
        intermediate -= intermediate.max(dim=1, keepdim=True).values
        intermediate = (y_hat**lambda_value) * torch.exp(intermediate / temperature)
        z = intermediate / intermediate.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return z


def update_mu(feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    n = z.shape[0]
    weights = z / n
    new_mu = torch.einsum("ij,ik->jk", weights, feat)  # [K, d]
    new_mu = new_mu / (z.sum(dim=0) / n).clamp_min(1e-12).unsqueeze(-1)
    new_mu = F_.normalize(new_mu, p=2, dim=-1, eps=1e-8)
    return new_mu


def update_sigma(feat: torch.Tensor, z: torch.Tensor, mu: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n = z.shape[0]
    d = feat.shape[1]
    std = torch.zeros(d, device=feat.device, dtype=feat.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        diff_sq = (feat[start:end, None, :] - mu[None, :, :]) ** 2  # [n_chunk, K, d]
        std += torch.einsum("ij,ijk->k", z[start:end], diff_sq) / n
    std = std / (z.sum() / n).clamp_min(1e-12)
    return std.clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Per-scan refinement
# ---------------------------------------------------------------------------


def refine_scan(model: torch.nn.Module, batch: dict, cfg: TransCLIPConfig, device: str, num_classes: int) -> ScanResult:
    """Run the full per-scan pipeline: forward -> TransCLIP solver -> decode."""
    _xyz, feat, prob, segment = extract_scan_outputs(model, batch, device)

    feat_n = F_.normalize(feat, p=2, dim=-1, eps=1e-8)
    d = feat_n.shape[1]
    std_init = cfg.std_init if cfg.std_init is not None else 1.0 / d

    # y_hat / z init: TransCLIP's "On-top" mode -- prob is already the frozen
    # model's own softmax, so no extra softmax and no clip-prototype logits.
    y_hat = prob
    z = prob.clone()

    mu = init_mu(feat_n, z, num_classes, cfg.n_most_confident)
    std = init_sigma(d, std_init, device=feat_n.device, dtype=feat_n.dtype)
    gaussian = DiagGaussian(mu, std, cfg.chunk_size)

    w = build_affinity_matrix(feat_n, cfg.n_neighbors)

    for it in range(cfg.max_iter + 1):
        gmm_log_likelihood = gaussian.log_likelihood(feat_n)
        z = update_z(
            gmm_log_likelihood, y_hat, z, w, cfg.lambda_value, cfg.n_neighbors, cfg.temperature, cfg.inner_iters
        )
        if it == cfg.max_iter:
            break
        gaussian.mu = update_mu(feat_n, z)
        gaussian.std = update_sigma(feat_n, z, gaussian.mu, cfg.chunk_size)

    labels_before = prob.argmax(dim=-1)
    labels_after = z.argmax(dim=-1)

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

    print(
        f"[main] n_neighbors={cli_cfg.n_neighbors} temperature={cli_cfg.temperature} "
        f"lambda_value={cli_cfg.lambda_value} max_iter={cli_cfg.max_iter} "
        f"inner_iters={cli_cfg.inner_iters} num_classes={num_classes} "
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

        result = refine_scan(model, batch, cli_cfg, device, num_classes)

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
