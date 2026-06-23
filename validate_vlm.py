"""
Standalone VLM Validation Script
Evaluates OpenScene MinkUNet18A on the nuScenes val split across all 16 GFS classes.
Replicates the OpenScene 3D-distill inference pipeline exactly.

Run from gfs_vl/:
    python validate_vlm.py --data_root data/nuscenes --vlm_weight data/nuscenes_openseg.pth.tar
"""

import os
import sys
import argparse
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.utils.data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointcept.datasets import build_dataset, collate_fn
from pointcept.datasets.preprocessing.nuscenes.nuscenes_constants import (
    NUSCENES_BASE_NAMES,
    NUSCENES_NOVEL_NAMES,
)
from pointcept.utils.misc import intersection_and_union_gpu


NUM_CLASSES = 16
NUM_BASES   = len(NUSCENES_BASE_NAMES)   # 12

# GFS class ordering (base 0-11, novel 12-15) with OpenScene's exact CLIP-friendly
# names. These must match the GT label indices produced by NuScenesDataset_GFS.
#
# Key corrections vs naive GFS names:
#   "construction-vehicle" → "construction vehicle"
#   "traffic-cone"         → "traffic cone"
#   "driveable-surface"    → "drivable surface"
#   "other-flat"           → "other flat"
#   "pedestrian"           → "person"   ← OpenScene uses this in NUSCENES_LABELS_16
CLIP_CLASS_NAMES = [
    "barrier",              # GFS 0  (base)
    "bus",                  # GFS 1  (base)
    "car",                  # GFS 2  (base)
    "construction vehicle", # GFS 3  (base)
    "traffic cone",         # GFS 4  (base)
    "trailer",              # GFS 5  (base)
    "drivable surface",     # GFS 6  (base)
    "other flat",           # GFS 7  (base)
    "sidewalk",             # GFS 8  (base)
    "terrain",              # GFS 9  (base)
    "manmade",              # GFS 10 (base)
    "vegetation",           # GFS 11 (base)
    "truck",                # GFS 12 (novel)
    "motorcycle",           # GFS 13 (novel)
    "bicycle",              # GFS 14 (novel)
    "person",               # GFS 15 (novel) — OpenScene uses "person" not "pedestrian"
]
DISPLAY_NAMES = (
    list(NUSCENES_BASE_NAMES) + list(NUSCENES_NOVEL_NAMES)
)  # human-readable names for the results table


# ─────────────────────────────────────────────────────────────────────────────
# VLM construction
# ─────────────────────────────────────────────────────────────────────────────

def build_vlm(vlm_weight: str, device: torch.device):
    from openscene.models.mink_unet import mink_unet

    vlm = mink_unet(in_channels=3, out_channels=768, D=3, arch="MinkUNet18A")

    if os.path.isfile(vlm_weight):
        print(f"[VLM] Loading local checkpoint: {vlm_weight}")
        ckpt = torch.load(vlm_weight, map_location="cpu")
    else:
        from torch.utils import model_zoo
        url = "https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar"
        print(f"[VLM] Downloading checkpoint from {url}")
        ckpt = model_zoo.load_url(url, progress=True)

    state = ckpt["state_dict"]
    new_state = OrderedDict(
        (k.replace("module.", "").replace("net3d.", ""), v)
        for k, v in state.items()
    )
    vlm.load_state_dict(new_state, strict=True)
    vlm.to(device).eval()
    for p in vlm.parameters():
        p.requires_grad = False

    print(f"[VLM] Loaded. Params: {sum(p.numel() for p in vlm.parameters()):,}")
    return vlm


def build_text_embeddings(device: torch.device, clip_model_name: str = "ViT-L/14@336px"):
    """
    Build CLIP text embeddings using OpenScene's exact setup:
      - model: ViT-L/14@336px
      - template: "a {label} in a scene"  (single template, prompt_eng=True)
      - normalize output embeddings
    """
    import clip

    print(f"[CLIP] Loading {clip_model_name} ...")
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)

    # Exact OpenScene prompt engineering: "a X in a scene"
    prompts = [f"a {name} in a outdoor scene" for name in CLIP_CLASS_NAMES]
    print(f"[CLIP] Prompts:\n" + "\n".join(f"  {p}" for p in prompts))

    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        text_embeddings = clip_model.encode_text(tokens).float()
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    del clip_model
    torch.cuda.empty_cache()
    print(f"[CLIP] text_embeddings: {text_embeddings.shape}")
    return text_embeddings   # (16, 768)


# ─────────────────────────────────────────────────────────────────────────────
# VLM forward pass (coord → voxelize → SparseTensor → unproject → logits)
# ─────────────────────────────────────────────────────────────────────────────

def vlm_forward(vlm, text_embeddings, input_dict, device, voxel_size=0.05):
    """
    Replicates OpenScene's distill evaluation exactly:
      predictions = model(sinput)               # (M, 768) — NOT normalized
      pred = predictions.half() @ text_features.t()   # raw dot-product
    Input features are all-zeros (OpenScene uses zeros for LiDAR, not ones).
    Returns raw logits (N, 16) for accumulation across test repeats.
    """
    import MinkowskiEngine as ME
    from openscene.dataset.voxelization_utils import sparse_quantize

    xyz    = input_dict["coord"]   # (N, 3) on device
    offset = input_dict["offset"]  # cumulative end indices

    coords_list    = []
    feats_list     = []
    sparse_offsets = []

    sparse_ptr = 0
    start      = 0
    for batch_idx, end in enumerate(offset.tolist()):
        xyz_np    = xyz[start:end].cpu().numpy()
        coords_sc = np.floor(xyz_np / voxel_size)

        inds, inds_recon = sparse_quantize(coords_sc, return_index=True)

        coords_vox = torch.from_numpy(coords_sc[inds]).int()
        batch_col  = torch.full((len(inds), 1), batch_idx, dtype=torch.int32)
        coords_list.append(torch.cat([batch_col, coords_vox], dim=1))

        # point_loader.py:169 — when input_color=False, features are replaced
        # with all-ones regardless of the raw data. Use the same convention.
        feats_list.append(torch.ones(len(inds), 3))

        sparse_offsets.append((
            sparse_ptr,
            torch.from_numpy(inds_recon).long().to(device),
            start,
        ))
        sparse_ptr += len(inds)
        start = end

    coords = torch.cat(coords_list, dim=0).to(device)
    feats  = torch.cat(feats_list,  dim=0).to(device)
    sinput = ME.SparseTensor(feats.float(), coords)

    # VLM forward — returns sparse feature tensor .F, shape (M_total, 768)
    sparse_feats = vlm(sinput)

    # Unproject voxel-level M → point-level N
    N         = xyz.shape[0]
    feats_out = torch.zeros(N, 768, device=device)
    for (sp_start, inds_recon, orig_start), end in zip(sparse_offsets, offset.tolist()):
        feats_out[orig_start:end] = sparse_feats[sp_start:][inds_recon]

    # Raw dot-product — OpenScene distill mode does NOT normalise 3D features
    logits = feats_out.half() @ text_embeddings.half().t()   # (N, 16)
    return logits


# ─────────────────────────────────────────────────────────────────────────────
# Val dataloader
# ─────────────────────────────────────────────────────────────────────────────

def build_val_loader(data_root: str, num_workers: int):
    val_cfg = dict(
        type="NuScenesDataset_GFS",
        split="val",
        data_root=data_root,
        sweeps=10,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "strength", "segment"),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    )
    val_data = build_dataset(val_cfg)
    loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"[Data] Val set: {len(val_data)} samples")
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(vlm, text_embeddings, val_loader, device, test_repeats=5, print_freq=100):
    """
    Runs test_repeats passes over the val set and accumulates raw logits before
    taking argmax — matches OpenScene's test_repeats=5 strategy that handles
    MinkowskiNet voxelization randomness.
    """
    intersection_total = np.zeros(NUM_CLASSES)
    union_total        = np.zeros(NUM_CLASSES)
    target_total       = np.zeros(NUM_CLASSES)

    all_logits  = []   # list of (N_total, 16) per-sample tensors
    all_segments = []  # list of (N_total,) GT tensors

    # ── Pass 1: collect logits and GT ────────────────────────────────────────
    print(f"\n[Eval] Pass 1/{test_repeats} ...")
    for i, input_dict in enumerate(tqdm(val_loader)):
        for key, val in input_dict.items():
            if isinstance(val, torch.Tensor):
                input_dict[key] = val.to(device, non_blocking=True)

        with torch.no_grad():
            logits = vlm_forward(vlm, text_embeddings, input_dict, device)  # (N, 16)

        all_logits.append(logits.cpu())
        all_segments.append(input_dict["segment"].cpu())

        # if (i + 1) % print_freq == 0 or (i + 1) == len(val_loader):
        #     print(f"  [{i+1:5d}/{len(val_loader)}]", flush=True)

    # ── Passes 2..test_repeats: accumulate logits with different seeds ────────
    for rep in range(1, test_repeats):
        seed = np.random.randint(10000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"\n[Eval] Pass {rep+1}/{test_repeats} (seed={seed}) ...")

        for i, input_dict in enumerate(val_loader):
            for key, val in input_dict.items():
                if isinstance(val, torch.Tensor):
                    input_dict[key] = val.to(device, non_blocking=True)

            with torch.no_grad():
                logits = vlm_forward(vlm, text_embeddings, input_dict, device)

            all_logits[i] = all_logits[i] + logits.cpu()

            if (i + 1) % print_freq == 0 or (i + 1) == len(val_loader):
                print(f"  [{i+1:5d}/{len(val_loader)}]", flush=True)

    # ── Compute metrics from accumulated logits ───────────────────────────────
    print("\n[Eval] Computing metrics ...")
    for logits, segment in zip(all_logits, all_segments):
        preds = logits.max(dim=1)[1].to(device)
        segment = segment.to(device)
        inter, union, target = intersection_and_union_gpu(
            preds, segment, NUM_CLASSES, ignore_index=-1
        )
        intersection_total += inter.cpu().numpy()
        union_total        += union.cpu().numpy()
        target_total       += target.cpu().numpy()

    return intersection_total, union_total, target_total


def print_results(intersection, union, target):
    iou_class = intersection / (union  + 1e-10)
    acc_class = intersection / (target + 1e-10)

    m_iou     = float(np.mean(iou_class))
    m_acc     = float(np.mean(acc_class))
    all_acc   = float(intersection.sum() / (target.sum() + 1e-10))
    base_iou  = float(np.mean(iou_class[:NUM_BASES]))
    novel_iou = float(np.mean(iou_class[NUM_BASES:]))
    hm_iou    = 2 * base_iou * novel_iou / (base_iou + novel_iou + 1e-10)

    print("\n" + "=" * 62)
    print("  OpenScene VLM (3D distill) — nuScenes val")
    print("=" * 62)
    print(f"  mIoU (all 16):  {m_iou:.2f}")
    print(f"  Base mIoU (12): {base_iou:.2f}")
    print(f"  Novel mIoU (4): {novel_iou:.2f}")
    print(f"  HM IoU:         {hm_iou:.2f}")
    print(f"  mAcc:           {m_acc:.2f}")
    print(f"  Overall Acc:    {all_acc:.2f}")
    print("-" * 62)
    print(f"  {'Class':30s}  {'IoU':>7}  {'Acc':>7}  {'Type':>6}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*6}")
    for i, name in enumerate(DISPLAY_NAMES):
        tag = "NOVEL" if i >= NUM_BASES else "base"
        print(f"  {name:30s}  {iou_class[i]:6.2f}  {acc_class[i]:6.2f}  {tag:>6}")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Validate OpenScene VLM on nuScenes val")
    p.add_argument("--data_root",     default="data/nuscenes")
    p.add_argument("--vlm_weight",    default="data/nuscenes_openseg.pth.tar")
    p.add_argument("--clip_model",    default="ViT-L/14@336px")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--test_repeats",  type=int, default=1,
                   help="Number of inference passes to accumulate. OpenScene uses 5 "
                        "because their Voxelizer applies random rotation on every call. "
                        "Our GridSample is deterministic so >1 gives identical results.")
    p.add_argument("--print_freq",    type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    vlm             = build_vlm(args.vlm_weight, device)
    text_embeddings = build_text_embeddings(device, args.clip_model)
    val_loader      = build_val_loader(args.data_root, args.num_workers)

    intersection, union, target = evaluate(
        vlm, text_embeddings, val_loader, device,
        test_repeats=args.test_repeats,
        print_freq=args.print_freq,
    )
    print_results(intersection, union, target)
