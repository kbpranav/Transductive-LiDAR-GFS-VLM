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
"""
Zero-shot inference on a single nuScenes scan using the pretrained
OpenScene nuScenes distilled model (MinkUNet18A → 768-dim CLIP features).

Usage:
    cd /home/kolarp/workspace/transductive-lidar/gfs_vl/openscene
    python run/infer_nuscenes.py \
        --scan_path /home/kolarp/workspace/transductive-lidar/gfs_vl/data/nuscenes/raw/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin \
        --output_dir out/nuscenes_zeroshot

Optional:
    --label_path  path to .bin label file (gt_segment_path from info pkl)
    --voxel_size  0.05  (default)
    --save_features      save per-point 768-d features as .npy
    --model_path  local .pth.tar path (skip download)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils import model_zoo
from collections import OrderedDict

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

import clip

from models.mink_unet import mink_unet
from dataset.voxelization_utils import sparse_quantize


# ── nuScenes 16 evaluation classes ─────────────────────────────────────────
NUSCENES_LABELS = [
    "barrier",              # 0
    "bicycle",              # 1
    "bus",                  # 2
    "car",                  # 3
    "construction-vehicle", # 4
    "motorcycle",           # 5
    "traffic-cone",         # 6
    "trailer",              # 7
    "truck",                # 8
    "driveable-surface",    # 9
    "other-flat",           # 10
    "sidewalk",             # 11
    "terrain",              # 12
    "manmade",              # 13
    "vegetation",           # 14
    "pedestrian",           # 15
]

# nuScenes raw label ID → learning ID (same as NuScenesDataset.get_learning_map)
NUSCENES_LEARNING_MAP = {
    0: -1, 1: -1, 2: 6,  3: 6,  4: 6,  5: -1, 6: 6,  7: -1, 8: -1,
    9: 0,  10: -1, 11: -1, 12: 7, 13: -1, 14: 1,  15: 2,  16: 2,
    17: 3,  18: 4,  19: -1, 20: -1, 21: 5,  22: 8,  23: 9,  24: 10,
    25: 11, 26: 12, 27: 13, 28: 14, 29: -1, 30: 15, 31: -1,
}

MODEL_URL = 'https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scan_path', required=True, help='Path to nuScenes .pcd.bin lidar file')
    p.add_argument('--label_path', default=None, help='Path to gt_segment .bin label file (optional)')
    p.add_argument('--output_dir', default='out/nuscenes_zeroshot')
    p.add_argument('--voxel_size', type=float, default=0.05)
    p.add_argument('--save_features', action='store_true')
    p.add_argument('--model_path', default=None, help='Local .pth.tar path (skip download)')
    return p.parse_args()


def load_nuscenes_scan(path):
    """
    Load a nuScenes lidar .pcd.bin → (N,3) xyz, (N,) intensity.
    nuScenes stores 5 floats per point: x, y, z, intensity, ring_index.
    """
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    xyz = pts[:, :3]
    intensity = pts[:, 3] / 255.0  # normalize to [0, 1]
    return xyz, intensity


def load_nuscenes_labels(path):
    """
    Load a nuScenes gt_segment .bin label file → (N,) learning IDs (0-15, -1=ignore).
    Raw labels are uint8 (0-31), mapped via NUSCENES_LEARNING_MAP.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    mapped = np.full(raw.shape, -1, dtype=np.int32)
    for k, v in NUSCENES_LEARNING_MAP.items():
        mapped[raw == k] = v
    return mapped


def build_model():
    """Build MinkUNet18A with 768-dim output (OpenSeg CLIP feature dim)."""
    return mink_unet(in_channels=3, out_channels=768, D=3, arch='MinkUNet18A')


def load_pretrained(model, model_path=None):
    """Load pretrained nuScenes OpenScene weights."""
    if model_path and os.path.isfile(model_path):
        print(f"Loading local checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location='cpu')
    else:
        print(f"Downloading pretrained model from:\n  {MODEL_URL}")
        ckpt = model_zoo.load_url(MODEL_URL, progress=True)

    state = ckpt['state_dict']
    new_state = OrderedDict()
    for k, v in state.items():
        k = k.replace('module.', '').replace('net3d.', '')
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    print("Model loaded successfully.")
    return model


def voxelize(xyz, voxel_size):
    """Voxelize point cloud, return voxel coords, indices, reconstruction map."""
    coords_scaled = np.floor(xyz / voxel_size)
    inds, inds_recon = sparse_quantize(coords_scaled, return_index=True)
    coords_vox = coords_scaled[inds]
    return coords_vox, inds, inds_recon


def extract_clip_text_features(labels, model_name="ViT-L/14@336px", prompt_eng=True):
    """Get CLIP text embeddings for nuScenes label set."""
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
    return feats  # (16, 768)


@torch.no_grad()
def run_inference(model, xyz, voxel_size):
    """Run the 3D model, return per-point (N, 768) features."""
    coords_vox, inds, inds_recon = voxelize(xyz, voxel_size)

    coords_torch = torch.from_numpy(coords_vox).int()
    batch_idx = torch.zeros(coords_torch.shape[0], 1, dtype=torch.int)
    coords_torch = torch.cat([batch_idx, coords_torch], dim=1)
    feats_torch = torch.ones(coords_torch.shape[0], 3).float()

    sinput = SparseTensor(feats_torch.cuda(), coords_torch.cuda())
    output = model(sinput)  # (M, 768)

    output_full = output[inds_recon]
    return output_full


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load scan
    print(f"Loading scan: {args.scan_path}")
    xyz, intensity = load_nuscenes_scan(args.scan_path)
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
    scan_name = os.path.basename(args.scan_path).replace('.pcd.bin', '').replace('.bin', '')
    if args.save_features:
        feat_path = os.path.join(args.output_dir, f'{scan_name}_feat.npy')
        np.save(feat_path, point_features.cpu().numpy())
        print(f"  Saved features → {feat_path}")

    # 5. Zero-shot classification with nuScenes 16 classes
    text_features = extract_clip_text_features(NUSCENES_LABELS)  # (16, 768)

    pf = point_features.half()
    pf = pf / (pf.norm(dim=-1, keepdim=True) + 1e-5)
    logits = pf @ text_features.t()  # (N, 16)
    pred = logits.max(dim=1)[1].cpu().numpy()  # 0-indexed into NUSCENES_LABELS

    # save predictions
    pred_path = os.path.join(args.output_dir, f'{scan_name}_pred.npy')
    np.save(pred_path, pred)
    print(f"  Saved predictions → {pred_path}")

    # print class distribution
    print("\n  Predicted class distribution:")
    for cid in range(len(NUSCENES_LABELS)):
        count = (pred == cid).sum()
        if count > 0:
            print(f"    {NUSCENES_LABELS[cid]:>25s}: {count:>8d} ({100*count/len(pred):.1f}%)")

    # 6. Evaluate against GT if label file provided
    if args.label_path and os.path.isfile(args.label_path):
        print(f"\nLoading GT labels: {args.label_path}")
        gt = load_nuscenes_labels(args.label_path)

        # only evaluate valid points (learning ID 0-15)
        valid = (gt >= 0) & (gt <= 15)
        gt_valid = gt[valid]
        pred_valid = pred[valid]

        num_classes = 16
        print("\n  {:>25s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
            "class", "IoU", "Prec", "Recall", "GT_cnt", "Pred_cnt"))
        print("  " + "-" * 72)

        ious = []
        for cid in range(num_classes):
            gt_mask   = (gt_valid == cid)
            pred_mask = (pred_valid == cid)
            tp = (gt_mask & pred_mask).sum()
            fp = (~gt_mask & pred_mask).sum()
            fn = (gt_mask & ~pred_mask).sum()
            iou  = tp / (tp + fp + fn + 1e-10)
            prec = tp / (tp + fp + 1e-10)
            rec  = tp / (tp + fn + 1e-10)
            ious.append(iou)
            print(f"    {NUSCENES_LABELS[cid]:>25s} {iou:8.4f} {prec:8.4f} {rec:8.4f} "
                  f"{gt_mask.sum():>8d} {pred_mask.sum():>8d}")

        miou = np.mean(ious)
        print(f"\n  mIoU: {miou:.4f}")

        # Confusion matrix
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        for g, p in zip(gt_valid, pred_valid):
            if 0 <= g < num_classes and 0 <= p < num_classes:
                conf[g, p] += 1

        conf_path = os.path.join(args.output_dir, f'{scan_name}_confusion.npy')
        np.save(conf_path, conf)
        print(f"\n  Saved confusion matrix → {conf_path}")

        print("\n  Top confusions (GT → predicted as):")
        for cid in range(num_classes):
            gt_count = conf[cid].sum()
            if gt_count == 0:
                continue
            top3 = np.argsort(conf[cid])[::-1][:3]
            parts = [
                f"{NUSCENES_LABELS[t]}({100*conf[cid][t]/gt_count:.0f}%)"
                for t in top3 if conf[cid][t] > 0
            ]
            print(f"    {NUSCENES_LABELS[cid]:>25s} [{gt_count:>6d} pts] → {', '.join(parts)}")

    print("\nDone!")


if __name__ == '__main__':
    main()