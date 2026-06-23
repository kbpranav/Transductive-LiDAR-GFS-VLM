"""
WandbLogger Hook

Logs training metrics and LiDAR point cloud visualizations to Weights & Biases.

Per step  : training loss (every log_interval steps)
Per epoch : epoch-avg training loss + point cloud pairs per val scene
            - Pre-training : GT | PTv3-pred  (2 clouds × num_vis scenes)
            - GFS training : GT | PTv3-pred | VLM-pred  (3 clouds × num_vis scenes)
Per regis : val mIoU / base / novel / HM  (from GFS_SemSegEvaluator)
"""

import numpy as np
import torch
from pathlib import Path

import pointcept.utils.comm as comm
from .default import HookBase
from .builder import HOOKS

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── NuScenes 16-class color palette (index = learning ID) ───────────────────
# Base classes  : 0-11
# Novel classes : 12-15  (truck, motorcycle, bicycle, pedestrian)
_CLASS_COLORS = np.array([
    [ 70, 130, 180],  #  0  barrier              steel blue
    [ 60, 179, 113],  #  1  bus                  medium sea green
    [255, 165,   0],  #  2  car                  orange
    [148,   0, 211],  #  3  construction-vehicle dark violet
    [  0, 206, 209],  #  4  traffic-cone          dark turquoise
    [  0, 128,   0],  #  5  trailer               green
    [  0,   0, 205],  #  6  driveable-surface     medium blue
    [165,  42,  42],  #  7  other-flat            brown
    [ 64, 224, 208],  #  8  sidewalk              turquoise
    [ 50, 205,  50],  #  9  terrain               lime green
    [199,  21, 133],  # 10  manmade               medium violet red
    [ 32, 178, 170],  # 11  vegetation            light sea green
    [220,  20,  60],  # 12  truck        (NOVEL)  crimson
    [255,  20, 147],  # 13  motorcycle   (NOVEL)  deep pink
    [255, 215,   0],  # 14  bicycle      (NOVEL)  gold
    [100, 149, 237],  # 15  pedestrian   (NOVEL)  cornflower blue
    # extra slots in case palette is reused for other datasets
    [255, 140,   0],  # 16
    [127, 255,   0],  # 17
    [219, 112, 147],  # 18
    [ 46, 139,  87],  # 19
    [  0, 191, 255],  # 20
    [210, 105,  30],  # 21
    [144, 238, 144],  # 22
    [255, 105, 180],  # 23
    [128, 128,   0],  # 24
], dtype=np.uint8)

_IGNORE_COLOR = np.array([80, 80, 80], dtype=np.uint8)  # dark gray for label == -1


def _labels_to_rgb(labels):
    """Map integer class IDs to RGB.  label < 0  →  dark gray (ignore)."""
    rgb = np.empty((len(labels), 3), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        rgb[i] = _IGNORE_COLOR if lbl < 0 else _CLASS_COLORS[int(lbl) % len(_CLASS_COLORS)]
    return rgb


@HOOKS.register_module()
class WandbLogger(HookBase):
    """
    Add as the LAST hook so it can read current_metric_value that evaluators set.

    Pre-training config:
        dict(type="WandbLogger", project="gfs-lidar", name="pretrain-base")

    Registration training config:
        dict(type="WandbLogger", project="gfs-lidar", name="registrain-k5")

    Args:
        project      : W&B project name.
        name         : W&B run name (defaults to save_path directory stem).
        log_interval : Log per-step training loss every N optimizer steps.
        num_vis      : Number of val scenes to visualize per epoch.
        max_points   : Downsample clouds to this many points before uploading.
    """

    def __init__(
        self,
        project="gfs-lidar",
        name=None,
        log_interval=20,
        num_vis=2,
        max_points=25000,
    ):
        if not _WANDB_AVAILABLE:
            raise ImportError("wandb not installed — run: pip install wandb")
        self.project = project
        self.name = name
        self.log_interval = log_interval
        self.num_vis = num_vis
        self.max_points = max_points

        self._initialized = False   # wandb.init called only once
        self._vis_samples = None    # list of CPU-cached val batches
        self._step = 0              # global optimizer-step counter
        self._global_epoch = 0      # global epoch counter across registrations
        self._epoch_losses = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _ensure_init(self):
        """Initialise wandb on first call from any hook method."""
        if self._initialized or not comm.is_main_process():
            return
        run_name = self.name or Path(self.trainer.cfg.save_path).name
        wandb.init(
            project=self.project,
            name=run_name,
            config={"save_path": self.trainer.cfg.save_path},
            resume="allow",
        )
        self._initialized = True
        self._cache_vis_samples()

    def before_train(self):
        self._ensure_init()

    def after_step(self):
        if not comm.is_main_process():
            return
        self._step += 1
        output = self.trainer.comm_info.get("model_output_dict", {})
        if "loss" not in output:
            return
        loss_val = output["loss"]
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()
        self._epoch_losses.append(loss_val)
        if self._step % self.log_interval == 0:
            wandb.log({"train/loss_step": loss_val, "optimizer_step": self._step})

    def after_epoch(self):
        if not comm.is_main_process():
            return
        self._global_epoch += 1
        prefix = self._regis_prefix()
        log_dict = {"epoch": self._global_epoch}

        if self._epoch_losses:
            log_dict[f"{prefix}train/loss_epoch"] = float(np.mean(self._epoch_losses))
            self._epoch_losses.clear()

        # metric from an evaluator that fires after_epoch (e.g. SemSegEvaluator)
        self._collect_metric(log_dict, prefix)

        if self._vis_samples:
            log_dict.update(self._make_vis_clouds(prefix))

        wandb.log(log_dict)

    def after_proto_extraction(self):
        """
        Visualize every k-shot support scan used to build the novel prototypes.
        Fires once per registration, before training starts.

        Each scan is logged as a point cloud colored by novel class label so
        you can immediately check whether the right scans / class regions were
        selected. Background points (label == -1) are dark gray.

        Example wandb keys:
            regis1/proto_support/truck_s0      ← seed 10, truck support sample 0
            regis1/proto_support/motorcycle_s1
            ...
        """
        self._ensure_init()
        if not comm.is_main_process():
            return
        loader = getattr(self.trainer, "regis_val_loader", None)
        if loader is None:
            return

        class_names = self.trainer.cfg.data.names          # all 16 names
        num_bases   = self.trainer.cfg.data.num_bases       # 12
        prefix      = self._regis_prefix()                  # e.g. "regis1/"

        # Track how many support samples we've seen per novel class
        class_sample_count = {}
        log_dict = {}

        for input_dict in loader:
            xyz      = input_dict["coord"]     # CPU tensor (N, 3)
            labels   = input_dict["segment"]   # CPU tensor (N,)
            name     = input_dict.get("name", ["unknown"])[0] if isinstance(
                input_dict.get("name"), (list, tuple)) else str(
                input_dict.get("name", "unknown"))

            # Identify which novel classes appear in this scan
            novel_ids = [
                lbl.item() for lbl in labels.unique()
                if lbl.item() >= num_bases
            ]
            if not novel_ids:
                continue  # skip scans with no novel labels (shouldn't happen)

            xyz_np    = xyz.numpy()
            label_np  = labels.numpy()

            # Downsample if needed
            N = xyz_np.shape[0]
            if N > self.max_points:
                pick = np.random.choice(N, self.max_points, replace=False)
                xyz_np   = xyz_np[pick]
                label_np = label_np[pick]

            cloud = np.hstack([xyz_np, _labels_to_rgb(label_np)]).astype(np.float32)

            # Use the dominant novel class name as the key (multiple novel classes
            # in one scan is rare but possible — we log once per dominant class)
            dominant = max(novel_ids, key=lambda c: (label_np == c).sum())
            cls_name = class_names[dominant]
            count    = class_sample_count.get(cls_name, 0)
            class_sample_count[cls_name] = count + 1

            key = f"{prefix}proto_support/{cls_name}_s{count}"
            log_dict[key] = wandb.Object3D(cloud)

        if log_dict:
            wandb.log(log_dict)
            self.trainer.logger.info(
                f"WandbLogger: logged {len(log_dict)} proto support scans "
                f"for {prefix.rstrip('/')}"
            )

    def after_train(self):
        """Picks up GFS_SemSegEvaluator metrics (fires after_train, just before this)."""
        if not comm.is_main_process():
            return
        prefix = self._regis_prefix()
        log_dict = {}
        self._collect_metric(log_dict, prefix, suffix="_final")
        if log_dict:
            wandb.log(log_dict)

        if self._is_last_registration():
            wandb.finish()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _regis_prefix(self):
        if hasattr(self.trainer, "regis_str"):
            return f"{self.trainer.regis_str}/"
        return ""

    def _collect_metric(self, log_dict, prefix, suffix=""):
        val = self.trainer.comm_info.get("current_metric_value")
        name = self.trainer.comm_info.get("current_metric_name")
        if val is not None and name is not None:
            log_dict[f"{prefix}val/{name}{suffix}"] = float(val)

    def _is_last_registration(self):
        if hasattr(self.trainer, "regis_str") and hasattr(self.trainer.cfg, "regis_train_list"):
            return self.trainer.regis_str == self.trainer.cfg.regis_train_list[-1]
        return True

    def _cache_vis_samples(self):
        """
        Cache the first num_vis batches from val_loader on CPU.
        Using the SAME scenes every epoch lets you track visual improvement over time.
        """
        val_loader = getattr(self.trainer, "val_loader", None)
        if val_loader is None:
            return
        self._vis_samples = []
        for i, batch in enumerate(val_loader):
            if i >= self.num_vis:
                break
            self._vis_samples.append({
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            })

    def _make_vis_clouds(self, prefix=""):
        """
        Run inference on cached val scenes and return wandb.Object3D entries.

        Pre-training  → 2 clouds per scene : GT | PTv3-pred
        GFS training  → 3 clouds per scene : GT | PTv3-pred | VLM-pred

        The VLM stream shows raw OpenScene predictions (before pseudo-label
        filtering), so you can compare what the teacher "sees" vs. what PTv3
        has learned.
        """
        result = {}
        has_vlm = (
            hasattr(self.trainer, "_run_vlm")
            and hasattr(self.trainer, "vlm_novel_proto")
        )
        self.trainer.model.eval()

        for scene_idx, raw_batch in enumerate(self._vis_samples):
            batch = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in raw_batch.items()
            }

            # ── PTv3 prediction ──────────────────────────────────────────────
            with torch.no_grad():
                out = self.trainer.model(batch)
            ptv3_labels = out["seg_logits"].max(1)[1].cpu().numpy()

            # ── VLM raw prediction ───────────────────────────────────────────
            if has_vlm:
                vlm_labels, _ = self.trainer._run_vlm(batch)
                vlm_labels = vlm_labels.cpu().numpy()
            else:
                vlm_labels = None

            # ── extract first scene from the batch ───────────────────────────
            end = int(batch["offset"].cpu()[0])
            xyz = batch["coord"].cpu().numpy()[:end]           # (N, 3)
            gt_labels = batch["segment"].cpu().numpy()[:end]   # (N,)
            ptv3_labels = ptv3_labels[:end]
            if vlm_labels is not None:
                vlm_labels = vlm_labels[:end]

            # ── optional downsampling ────────────────────────────────────────
            N = xyz.shape[0]
            if N > self.max_points:
                pick = np.random.choice(N, self.max_points, replace=False)
                xyz = xyz[pick]
                gt_labels = gt_labels[pick]
                ptv3_labels = ptv3_labels[pick]
                if vlm_labels is not None:
                    vlm_labels = vlm_labels[pick]

            # ── build wandb point clouds (N, 6): XYZ + RGB ──────────────────
            def _cloud(labels):
                return np.hstack([xyz, _labels_to_rgb(labels)]).astype(np.float32)

            ep = f"ep{self._global_epoch:03d}"
            result[f"{prefix}vis/scene{scene_idx}_gt_{ep}"]    = wandb.Object3D(_cloud(gt_labels))
            result[f"{prefix}vis/scene{scene_idx}_ptv3_{ep}"]  = wandb.Object3D(_cloud(ptv3_labels))
            if vlm_labels is not None:
                result[f"{prefix}vis/scene{scene_idx}_vlm_{ep}"] = wandb.Object3D(_cloud(vlm_labels))

        self.trainer.model.train()
        return result
