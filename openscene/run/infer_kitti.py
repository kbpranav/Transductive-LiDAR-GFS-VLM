"""
Zero-shot inference on a single SemanticKITTI scan using the pretrained
OpenScene nuScenes distilled model (MinkUNet18A → 768-dim CLIP features).

Usage:
    cd /home/kolarp/workspace/transductive-lidar/openscene
    export PYTHONPATH=.
    python run/infer_kitti.py \
        --scan_path ../datasets/dataset/sequences/08/velodyne/000000.bin \
        --output_dir out/kitti_zeroshot

Optional:
    --label_path  ../datasets/dataset/sequences/08/labels/000000.label
    --voxel_size  0.05  (default, same as nuScenes training)
    --save_features      save per-point 768-d features as .npy
"""

import os
import sys
import argparse
import numpy as np

# Add openscene root to path so imports work without PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils import model_zoo
from collections import OrderedDict

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

import clip

from models.mink_unet import mink_unet
from dataset.voxelization_utils import sparse_quantize

# ── SemanticKITTI label map (19 evaluation classes) ────────────────────────
KITTI_LABELS_19 = (
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
    'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
    'trunk', 'terrain', 'pole', 'traffic-sign',
)

KITTI_LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}

MODEL_URL = 'https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scan_path', required=True, help='Path to a .bin velodyne scan')
    p.add_argument('--label_path', default=None, help='Path to .label file (optional, for mIoU)')
    p.add_argument('--output_dir', default='out/kitti_zeroshot')
    p.add_argument('--voxel_size', type=float, default=0.05)
    p.add_argument('--save_features', action='store_true', help='Save 768-d features as .npy')
    p.add_argument('--model_path', default=None, help='Local .pth.tar path (skip download)')
    return p.parse_args()


def load_kitti_scan(path):
    """Load a SemanticKITTI velodyne .bin → (N,3) xyz, (N,) intensity."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]
    intensity = pts[:, 3]
    return xyz, intensity


def load_kitti_labels(path):
    """Load a .label file → (N,) semantic ids (mapped to 0-19, 0=unlabeled)."""
    raw = np.fromfile(path, dtype=np.uint32)
    sem = (raw & 0xFFFF).astype(np.int32)
    mapped = np.zeros_like(sem)
    for k, v in KITTI_LEARNING_MAP.items():
        mapped[sem == k] = v
    return mapped


def build_model():
    """Build MinkUNet18A with 768-dim output (OpenSeg CLIP feature dim)."""
    net = mink_unet(in_channels=3, out_channels=768, D=3, arch='MinkUNet18A')
    return net


def load_pretrained(model, model_path=None):
    """Load the pretrained nuScenes OpenScene weights."""
    if model_path and os.path.isfile(model_path):
        print(f"Loading local checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location='cpu')
    else:
        print(f"Downloading pretrained model from:\n  {MODEL_URL}")
        ckpt = model_zoo.load_url(MODEL_URL, progress=True)

    state = ckpt['state_dict']

    # strip 'module.' or 'net3d.' prefixes from DDP / DisNet wrapper
    new_state = OrderedDict()
    for k, v in state.items():
        k = k.replace('module.', '').replace('net3d.', '')
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    print("Model loaded successfully.")
    return model


def voxelize(xyz, voxel_size):
    """
    Voxelize point cloud.
    Returns:
        coords_vox  (M, 3)  voxel coordinates
        inds        (M,)    indices into original point cloud
        inds_recon  (N,)    mapping: original → voxelized
    """
    coords_scaled = np.floor(xyz / voxel_size)
    inds, inds_recon = sparse_quantize(coords_scaled, return_index=True)
    coords_vox = coords_scaled[inds]
    return coords_vox, inds, inds_recon


def extract_clip_text_features(labels, model_name="ViT-L/14@336px", prompt_eng=True):
    """Get CLIP text embeddings for label set."""
    print(f"Loading CLIP {model_name} ...")
    clip_model, _ = clip.load(model_name, device='cuda', jit=False)

    if prompt_eng:
        texts = [f"a {label} in a scene" for label in labels]
    else:
        texts = list(labels)

    tokens = clip.tokenize(texts).cuda()
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # (C, 768)


@torch.no_grad()
def run_inference(model, xyz, voxel_size):
    """
    Run the 3D model on a point cloud.
    Returns per-point (N, 768) features.
    """
    coords_vox, inds, inds_recon = voxelize(xyz, voxel_size)

    # Build SparseTensor: features = ones (no color, same as nuScenes training)
    coords_torch = torch.from_numpy(coords_vox).int()
    # prepend batch index 0
    batch_idx = torch.zeros(coords_torch.shape[0], 1, dtype=torch.int)
    coords_torch = torch.cat([batch_idx, coords_torch], dim=1)
    feats_torch = torch.ones(coords_torch.shape[0], 3).float()

    sinput = SparseTensor(feats_torch.cuda(), coords_torch.cuda())
    output = model(sinput)  # (M, 768)

    # map back to original N points
    output_full = output[inds_recon]
    return output_full


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load scan
    print(f"Loading scan: {args.scan_path}")
    xyz, intensity = load_kitti_scan(args.scan_path)
    print(f"  Points: {xyz.shape[0]}")

    # 2. Build & load model
    model = build_model()
    model = load_pretrained(model, args.model_path)
    model = model.cuda().eval()

    # 3. Run inference → per-point 768-d CLIP-aligned features
    print("Running inference ...")
    point_features = run_inference(model, xyz, args.voxel_size)  # (N, 768)
    print(f"  Output features: {point_features.shape}")

    # 4. Save raw features if requested
    scan_name = os.path.splitext(os.path.basename(args.scan_path))[0]
    if args.save_features:
        feat_path = os.path.join(args.output_dir, f'{scan_name}_openscene_feat.npy')
        np.save(feat_path, point_features.cpu().numpy())
        print(f"  Saved features → {feat_path}")

    # 5. Zero-shot classification with SemanticKITTI 19 classes
    text_features = extract_clip_text_features(KITTI_LABELS_19)  # (19, 768)

    # cosine similarity: normalize point features before dot product.
    # OpenScene's original 'distill' mode skips this, but normalization
    # transfers better cross-dataset (nuScenes→KITTI) by removing
    # domain-specific magnitude bias.
    pf = point_features.half()
    pf = pf / (pf.norm(dim=-1, keepdim=True) + 1e-5)
    logits = pf @ text_features.t()  # (N, 19)
    pred = logits.max(dim=1)[1].cpu().numpy()  # 0-indexed into KITTI_LABELS_19

    # save predictions
    pred_path = os.path.join(args.output_dir, f'{scan_name}_pred.npy')
    np.save(pred_path, pred)
    print(f"  Saved predictions → {pred_path}")

    # print class distribution
    print("\n  Predicted class distribution:")
    for cid in range(len(KITTI_LABELS_19)):
        count = (pred == cid).sum()
        if count > 0:
            print(f"    {KITTI_LABELS_19[cid]:>20s}: {count:>8d} ({100*count/len(pred):.1f}%)")

    # 6. Evaluate against GT if label file provided
    if args.label_path and os.path.isfile(args.label_path):
        print(f"\nLoading GT labels: {args.label_path}")
        gt = load_kitti_labels(args.label_path)
        # gt uses 0=unlabeled, 1-19=classes → shift pred to 1-indexed for comparison
        pred_eval = pred + 1  # our pred is 0-indexed (0=car,...,18=traffic-sign) → shift to 1-19

        # only evaluate points with valid GT (1-19)
        valid = (gt >= 1) & (gt <= 19)
        gt_valid = gt[valid]
        pred_valid = pred_eval[valid]

        num_classes = 19
        # per-class IoU, precision, recall
        print("\n  {:>20s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
            "class", "IoU", "Prec", "Recall", "GT_cnt", "Pred_cnt"))
        print("  " + "-" * 62)
        ious = []
        for cid in range(1, num_classes + 1):
            gt_mask = (gt_valid == cid)
            pred_mask = (pred_valid == cid)
            tp = (gt_mask & pred_mask).sum()
            fp = (~gt_mask & pred_mask).sum()
            fn = (gt_mask & ~pred_mask).sum()
            intersection = tp
            union = tp + fp + fn
            iou = intersection / (union + 1e-10)
            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            ious.append(iou)
            print(f"    {KITTI_LABELS_19[cid-1]:>20s} {iou:8.4f} {prec:8.4f} {rec:8.4f} {gt_mask.sum():>8d} {pred_mask.sum():>8d}")
        miou = np.mean(ious)
        print(f"\n  mIoU: {miou:.4f}")

        # Confusion matrix (rows=GT, cols=Pred)
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        for i in range(len(gt_valid)):
            g = gt_valid[i] - 1  # 0-indexed
            p = pred_valid[i] - 1
            if 0 <= g < num_classes and 0 <= p < num_classes:
                conf[g, p] += 1

        # Save confusion matrix
        conf_path = os.path.join(args.output_dir, f'{scan_name}_confusion.npy')
        np.save(conf_path, conf)
        print(f"\n  Saved confusion matrix → {conf_path}")

        # Print top confusions per class (where GT points went)
        print("\n  Top confusions (GT → predicted as):")
        for cid in range(num_classes):
            gt_count = conf[cid].sum()
            if gt_count == 0:
                continue
            row = conf[cid]
            top3 = np.argsort(row)[::-1][:3]
            parts = []
            for t in top3:
                if row[t] > 0:
                    parts.append(f"{KITTI_LABELS_19[t]}({100*row[t]/gt_count:.0f}%)")
            print(f"    {KITTI_LABELS_19[cid]:>20s} [{gt_count:>6d} pts] → {', '.join(parts)}")

    print("\nDone!")


if __name__ == '__main__':
    main()
