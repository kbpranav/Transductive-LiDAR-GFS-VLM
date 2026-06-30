"""
Trainer

Author: Zhaochong An (anzhaochong@outlook.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
import numpy as np
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

# Make openscene importable (run from repo root)
# sys.path.insert(0, './openscene')
sys.path.insert(0, '/home/kolarp/workspace/transductive-lidar/gfs_vl')

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

from pathlib import Path
import torch.distributed as dist

TRAINERS = Registry("trainers")


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    self.before_step()
                    self.run_step()
                    self.after_step()
                self.after_epoch()
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            self.before_train()
            self.logger.info(
                ">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>"
            )
            for self.epoch in range(self.start_epoch, self.max_epoch):
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    self.before_step()
                    self.run_step()
                    self.after_step()
                self.after_epoch()
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = (
            SummaryWriter(self.cfg.save_path)
            if comm.is_main_process()
            else None
        )
        self.logger.info(
            f"Tensorboard writer logging dir: {self.cfg.save_path}"
        )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data
            )
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=(
                False if self.cfg.num_worker_per_gpu == 0 else True
            ),
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_data
                )
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(
            self.cfg.optimizer, self.model, self.cfg.param_dicts
        )

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader) * self.cfg.eval_epoch
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("GFS_VL_Trainer")
class GFS_VL_Trainer(Trainer):
    """
    Trainer for generalized few-shot 3D point cloud segmentation.
    Uses OpenScene MinkUNet18A as the frozen 3D VLM for pseudo-label generation.
    """

    def __init__(self, cfg):
        """Initialize the trainer and build the 3D VLM."""
        super().__init__(cfg)
        self.build_zero_shot_model()

    # ─────────────────────────────────────────────────────────────────────
    # 3D VLM (OpenScene) construction
    # ─────────────────────────────────────────────────────────────────────

    def build_zero_shot_model(self):
        """Entry point — dispatches to the OpenScene builder."""
        self.build_zero_shot_model_openscene()

    def build_zero_shot_model_openscene(self):
        """Build MinkUNet18A and load pretrained OpenScene nuScenes weights."""
        from openscene.models.mink_unet import mink_unet

        # Build MinkUNet18A with 768-dim CLIP-aligned output
        self.VLM_3D = mink_unet(
            in_channels=3,
            out_channels=768,
            D=3,
            arch='MinkUNet18A'
        )

        # Load pretrained OpenScene nuScenes weights
        from torch.utils import model_zoo

        if os.path.isfile(self.cfg.vlm_3d_weight):
            self.logger.info(f"Loading local OpenScene checkpoint: {self.cfg.vlm_3d_weight}")
            ckpt = torch.load(self.cfg.vlm_3d_weight, map_location='cpu')
        else:
            MODEL_URL = 'https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar'
            self.logger.info(f"Downloading OpenScene checkpoint from {MODEL_URL}")
            ckpt = model_zoo.load_url(MODEL_URL, progress=True)

        state = ckpt['state_dict']
        new_state = OrderedDict()
        for k, v in state.items():
            k = k.replace('module.', '').replace('net3d.', '')
            new_state[k] = v
        self.VLM_3D.load_state_dict(new_state, strict=True)

        self.VLM_3D.cuda().eval()
        for param in self.VLM_3D.parameters():
            param.requires_grad = False

        # Build CLIP text embeddings for all classes (base + novel)
        self.vlm_feat_dim = 768
        self._build_clip_text_embeddings_openscene()

        self.logger.info(
            f"OpenScene VLM loaded. text_embeddings: {self.text_embeddings.shape}, "
            f"num_bases={self.cfg.data.num_bases}, "
            f"num_base_novels={self.cfg.data.num_base_novels}"
        )

    def _build_clip_text_embeddings_openscene(self):
        """
        Build CLIP text embeddings for all base+novel classes.
        Single template "a {name} in a outdoor scene" — matches OpenScene's exact setup.
        """
        import clip
        clip_model, _ = clip.load("ViT-L/14@336px", device='cuda', jit=False)

        class_names = (
            self.train_loader.dataset.base_class_names
            + self.train_loader.dataset.novel_class_names
        )
        self.logger.info(f"Building CLIP text embeddings for {len(class_names)} classes: {class_names}")

        # ── Multi-template prompt ensembling (commented out — not working) ────
        # templates = [
        #     "a {} in a street scene",
        #     "a {} on the road",
        #     "a {} in an outdoor scene",
        #     "a photo of a {}",
        #     "a lidar scan of a {}",
        #     "a point cloud of a {}",
        # ]
        # with torch.no_grad():
        #     all_embeddings = []
        #     for tmpl in templates:
        #         texts = [tmpl.format(name) for name in class_names]
        #         tokens = clip.tokenize(texts).cuda()
        #         emb = clip_model.encode_text(tokens).float()
        #         emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-5)
        #         all_embeddings.append(emb)
        #     # Average across templates then re-normalise
        #     self.text_embeddings = torch.stack(all_embeddings).mean(dim=0)
        #     self.text_embeddings = self.text_embeddings / (
        #         self.text_embeddings.norm(dim=-1, keepdim=True) + 1e-5
        #     )
        # self.logger.info(f"Text embeddings built with {len(templates)} templates: {self.text_embeddings.shape}")
        # ─────────────────────────────────────────────────────────────────────

        # Single template — OpenScene original: "a {name} in a outdoor scene"
        prompts = [f"a {name} in a outdoor scene" for name in class_names]
        self.logger.info("Prompts:\n" + "\n".join(f"  {p}" for p in prompts))

        with torch.no_grad():
            tokens = clip.tokenize(prompts).cuda()
            self.text_embeddings = clip_model.encode_text(tokens).float()
            self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)

        self.logger.info(f"Text embeddings: {self.text_embeddings.shape}")
        # text_embeddings: (num_classes, 768)
        del clip_model
        torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────
    # 3D VLM forward pass (voxelize → forward → unproject)
    # ─────────────────────────────────────────────────────────────────────

    def construct_3D_vlm_input_openscene(self, input_dict):
        """
        Prepare MinkowskiEngine SparseTensor input for OpenScene.
        Stores `self._sparse_offsets` for later M→N unprojection in `_run_vlm`.
        """
        import MinkowskiEngine as ME
        from openscene.dataset.voxelization_utils import sparse_quantize

        xyz = input_dict["coord"]        # (N, 3) on CUDA
        offset = input_dict["offset"]    # cumulative end indices, e.g. [512] or [512, 1024]

        voxel_size = 0.1
        coords_list = []
        feats_list = []
        self._sparse_offsets = []        # list of (sparse_start, inds_recon_cuda, orig_start)

        sparse_ptr = 0
        start = 0
        for batch_idx, end in enumerate(offset.tolist()):
            xyz_batch = xyz[start:end].cpu().numpy()                   # (Nb, 3)
            coords_scaled = np.floor(xyz_batch / voxel_size)           # (Nb, 3)

            inds, inds_recon = sparse_quantize(
                coords_scaled, return_index=True
            )  # inds: unique voxel indices (Mb,), inds_recon: (Nb,)

            coords_vox = torch.from_numpy(coords_scaled[inds]).int()   # (Mb, 3)
            batch_col = torch.full(
                (coords_vox.shape[0], 1), batch_idx, dtype=torch.int32
            )
            coords_list.append(torch.cat([batch_col, coords_vox], dim=1))  # (Mb, 4)
            feats_list.append(torch.ones(coords_vox.shape[0], 3))          # (Mb, 3)

            self._sparse_offsets.append((
                sparse_ptr,
                torch.from_numpy(inds_recon).long().cuda(),
                start,
            ))
            sparse_ptr += len(inds)
            start = end

        coords = torch.cat(coords_list, dim=0).cuda()   # (M_total, 4) int, on GPU
        feats = torch.cat(feats_list, dim=0).cuda()    # (M_total, 3) float, on GPU
        return ME.SparseTensor(feats.float(), coords)

    @torch.no_grad()
    def _run_vlm(self, input_dict):
        """
        Run OpenScene VLM. Returns per-point (N,) seg_preds and (N, 768) normalized features.
        Unprojects from voxel-level (M) back to point-level (N).
        """
        sinput = self.construct_3D_vlm_input_openscene(input_dict)
        sparse_feats = self.VLM_3D(sinput)   # (M_total, 768) voxel-level — model returns .F directly

        # Unproject M → N
        N = input_dict["coord"].shape[0]
        feats = torch.zeros(N, self.vlm_feat_dim, device='cuda')
        offset = input_dict["offset"]
        for (sp_start, inds_recon, orig_start), end in zip(
            self._sparse_offsets, offset.tolist()
        ):
            feats[orig_start:end] = sparse_feats[sp_start:][inds_recon]

        assert feats.shape[0] == input_dict["coord"].shape[0], (
            f"VLM unprojection broken: {feats.shape[0]} vs {input_dict['coord'].shape[0]}"
        )

        # ── Normalised 3D features + dot product (commented out) ─────────────
        # OpenScene distill mode does NOT normalise 3D features before the dot
        # product — normalising here changes the effective similarity scale and
        # was causing degraded pseudo-label quality.
        # feats_norm = feats / (feats.norm(dim=-1, keepdim=True) + 1e-5)
        # logits = feats_norm.half() @ self.text_embeddings.half().t()
        # seg_preds = logits.max(dim=1)[1]
        # return seg_preds, feats_norm
        # ─────────────────────────────────────────────────────────────────────

        # Raw dot-product — OpenScene original (no 3D feature normalisation)
        logits = feats.half() @ self.text_embeddings.half().t()   # (N, num_classes)
        seg_preds = logits.max(dim=1)[1]                          # (N,)

        return seg_preds, feats

    # ─────────────────────────────────────────────────────────────────────
    # Prototype extraction (novel classes, from registration set)
    # ─────────────────────────────────────────────────────────────────────

    def extract_3d_vlm_proto_openscene(self, regis_val_loader):
        """Compute novel class prototypes using OpenScene features. Returns (num_novel, 768)."""
        num_novel = self.cfg.data.num_base_novels - self.cfg.data.num_bases
        avg_features = torch.zeros(num_novel, self.vlm_feat_dim, device='cuda')
        class_counts = torch.zeros(num_novel, device='cuda')

        if comm.is_main_process():
            for input_dict in regis_val_loader:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.cuda(non_blocking=True)

                # _run_vlm handles voxelization + unprojection → (N, 768)
                _, vlm_feats = self._run_vlm(input_dict)

                for class_id in input_dict["segment"].unique():
                    class_id = class_id.item()
                    if class_id < 0:
                        continue
                    assert class_id >= self.cfg.data.num_bases, (
                        f"REGIS dataset emitted base-class id {class_id}; "
                        f"expected only novel ids ≥ {self.cfg.data.num_bases}"
                    )
                    mask = input_dict["segment"] == class_id
                    idx = class_id - self.cfg.data.num_bases
                    avg_features[idx] += vlm_feats[mask].sum(dim=0)
                    class_counts[idx] += mask.sum()

            assert torch.all(class_counts > 0), (
                "Some novel classes have no support points — check registration dataset"
            )
            avg_features /= class_counts.unsqueeze(1)
            self.logger.info(f"vlm_novel_proto extracted: {avg_features.shape}")

        if comm.get_world_size() > 1:
            dist.broadcast(avg_features, src=0)

        return avg_features

    # ─────────────────────────────────────────────────────────────────────
    # Model build (loads base-trained backbone weights)
    # ─────────────────────────────────────────────────────────────────────

    def build_model(self):
        """Construct the segmentor, convert for DDP, load pre-trained weights."""
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.logger.info(f"Num params: {n_parameters}")

        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )

        # Load pre-trained weights if available
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key
                weight[key] = value
            strict = (
                self.cfg.load_strict if "load_strict" in self.cfg else True
            )
            missing_keys, unexpected_keys = model.load_state_dict(
                weight, strict=strict
            )
            self.logger.info(
                f"=> Loaded weight '{self.cfg.weight}' (epoch {checkpoint['epoch']})"
            )
            self.logger.info(
                f"=> missing_keys {missing_keys}, unexpected_keys {unexpected_keys}"
            )

        return model

    # ─────────────────────────────────────────────────────────────────────
    # Main training loop (multi-registration)
    # ─────────────────────────────────────────────────────────────────────

    def train(self):
        """
        Iterate over registration datasets. For each:
          1. Extract novel-class prototypes from the registration set using OpenScene.
          2. Build the k-shot training loader.
          3. Run the standard training loop with VLM-calibrated targets.
        """
        for self.regis_str in self.cfg.regis_train_list:
            if hasattr(self.cfg.data, self.regis_str):
                val_data_cfg = getattr(self.cfg.data, self.regis_str)
            else:
                raise AttributeError(
                    f"{self.regis_str} not found in self.cfg.data"
                )

            # ── Resume: skip training if this registration already completed ──
            last_ckpt = os.path.join(
                self.cfg.save_path, "model",
                f"{self.regis_str}_model_last.pth",
            )
            if os.path.isfile(last_ckpt):
                saved = torch.load(last_ckpt, map_location="cpu")
                if saved.get("epoch", 0) >= self.cfg.eval_epoch:
                    self.logger.info(
                        f"=> {self.regis_str} already completed "
                        f"(epoch {saved['epoch']}). "
                        f"Loading checkpoint and running eval only."
                    )
                    weight = OrderedDict(
                        (k, v) for k, v in saved["state_dict"].items()
                    )
                    self.model.load_state_dict(weight, strict=True)
                    with EventStorage() as self.storage:
                        for h in self.hooks:
                            h.before_train()
                        self.after_train()
                    continue
            # ──────────────────────────────────────────────────────────────────

            val_data = build_dataset(val_data_cfg)
            regis_val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=None,
                collate_fn=collate_fn,
            )

            # 1. Extract novel-class prototypes from the current registration set
            self.regis_val_loader = regis_val_loader  # exposed for hooks
            self.vlm_novel_proto = self.extract_3d_vlm_proto_openscene(regis_val_loader)
            for h in self.hooks:
                h.after_proto_extraction()

            # 2. Rebuild k-shot training loader with this registration's seed
            self.cfg.data.train.k_shot = self.cfg.k_shot
            self.cfg.data.train.seed = val_data_cfg["seed"]
            self.cfg.data.train.nb_mix_blks = self.cfg.nb_mix_blks
            self.train_loader = self.build_train_loader()

            # 3. Training loop for this registration
            with EventStorage() as self.storage, ExceptionWriter():
                self.before_train()
                self.logger.info(
                    ">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>"
                )
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    if comm.get_world_size() > 1:
                        self.train_loader.sampler.set_epoch(self.epoch)
                    self.model.train()
                    self.data_iterator = enumerate(self.train_loader)
                    self.before_epoch()
                    for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                    ) in self.data_iterator:
                        self.before_step()
                        self.run_step()
                        self.after_step()
                    self.after_epoch()
                self.after_train()

        # Aggregate final metrics across multiple registration sets
        if comm.is_main_process() and len(self.cfg.regis_train_list) > 1:
            eval_hook = next(
                (h for h in self.hooks if hasattr(h, "mean_mIoUs") and hasattr(h, "mean_iou_list")),
                None,
            )
            if eval_hook is not None and len(eval_hook.mean_iou_list) > 0:
                class_names = (
                    self.train_loader.dataset.base_class_names
                    + self.train_loader.dataset.novel_class_names
                )
                mean_mIoUs = eval_hook.mean_mIoUs / len(eval_hook.mean_iou_list)
                mIoU_val, base_mIoU, novel_mIoU, hm_mIoU = mean_mIoUs
                self.logger.info(
                    f"Final Eval result from {self.cfg.regis_train_list}: mIoU: {mIoU_val:.4f}, "
                    f"BASE: {base_mIoU:.4f}, NOVEL: {novel_mIoU:.4f}, hm_mIoU: {hm_mIoU:.4f}"
                )
                stack_iou = np.mean(
                    np.stack(eval_hook.mean_iou_list, axis=0), axis=0
                )
                for i, iou in enumerate(stack_iou):
                    self.logger.info(
                        f"Class_{i} - {class_names[i]} Final Result: iou {iou:.4f}"
                    )

    # ─────────────────────────────────────────────────────────────────────
    # Per-step: VLM target calibration → forward → loss → backward
    # ─────────────────────────────────────────────────────────────────────

    def run_step(self):
        """Single training step with VLM-calibrated targets."""
        input_dict = self.comm_info["input_dict"]
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.cuda(non_blocking=True)

        new_target = self.target_calibrate(input_dict)

        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            seg_logits = output_dict.pop("seg_logits")
            criteria = (
                self.model.module.criteria
                if comm.get_world_size() > 1
                else self.model.criteria
            )
            loss = criteria(seg_logits, new_target)
            output_dict["loss"] = loss

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    # ─────────────────────────────────────────────────────────────────────
    # Pseudo-label generation
    # ─────────────────────────────────────────────────────────────────────

    def pseudo_label_selection(
        self, all_pred_cloud, all_feat_cloud, offset, ps_thresh=0.6
    ):
        """
        Filter raw 3D VLM predictions based on prototype similarity.

        Args:
            all_pred_cloud (Tensor): Raw 3D VLM predictions for all points.
            all_feat_cloud (Tensor): Feature vectors for all points.
            offset (list): Batch offsets indicating point indices.
            ps_thresh (float): Cosine similarity threshold for filtering.

        Returns:
            Tensor: Filtered 3D VLM predictions.
        """
        filtered_preds = []
        for start_idx, end_idx in zip([0] + offset[:-1], offset):
            class_prototypes = {}
            pred_cloud = all_pred_cloud[start_idx:end_idx].clone()
            feat_cloud = all_feat_cloud[start_idx:end_idx].clone()

            # Mean feature per predicted class in the batch
            for class_id in pred_cloud.unique():
                class_mask = pred_cloud == class_id
                class_prototypes[class_id.item()] = feat_cloud[class_mask].mean(dim=0)

            # Filter predictions based on similarity with novel prototypes
            for class_id, proto in class_prototypes.items():
                if class_id < self.cfg.data.num_bases:
                    # Ignore base classes (handled by ground-truth labels)
                    pred_cloud[pred_cloud == class_id] = -1
                else:
                    novel_proto = self.vlm_novel_proto[
                        class_id - self.cfg.data.num_bases
                    ]
                    similarity = torch.cosine_similarity(
                        proto, novel_proto, dim=0
                    )
                    if similarity < ps_thresh:
                        self.logger.info(
                            f"=> Filtering class {class_id}, similarity {similarity}"
                        )
                        pred_cloud[pred_cloud == class_id] = -1

            filtered_preds.append(pred_cloud)

        return torch.cat(filtered_preds, dim=0)

    @torch.no_grad()
    def target_calibrate(self, input_dict):
        """
        Calibrate segmentation target labels using 3D VLM predictions + few-shot data.

        Flow:
          1. Run OpenScene VLM → per-point CLIP features + raw class predictions.
          2. Pseudo-Label Selection: filter low-confidence novel predictions.
          3. Adaptive Infilling: assign novel labels to remaining unlabeled points.
          4. Replace background (-1) regions of the GT label with refined predictions.
        """
        label = input_dict["segment"].clone()
        bg_label_mask = label == -1  # default background label

        # 1. Run VLM
        prediction, vlm_feats = self._run_vlm(input_dict)

        # 2. Pseudo-label selection
        prediction = self.pseudo_label_selection(
            prediction,
            vlm_feats,
            input_dict["offset"].tolist(),
            ps_thresh=self.cfg.ps_thresh,
        )

        # 3. Adaptive Infilling
        offset = input_dict["offset"].tolist()
        new_predictions = []
        for start_idx, end_idx in zip([0] + offset[:-1], offset):
            current_prototypes = {}
            pred_cloud = prediction[start_idx:end_idx].clone()
            feat_cloud = vlm_feats[start_idx:end_idx].clone()
            cur_label = label[start_idx:end_idx].clone()
            cur_bg_mask = bg_label_mask[start_idx:end_idx]
            bg_pred_cloud = pred_cloud[cur_bg_mask]
            bg_feat_cloud = feat_cloud[cur_bg_mask]

            if "mask" in input_dict:
                noda_mask = input_dict["mask"][start_idx:end_idx].clone()

            # Build prototypes from novel classes present in the background region
            for class_id in bg_pred_cloud.unique():
                if class_id < 0:
                    continue
                assert class_id >= self.cfg.data.num_bases
                class_features = bg_feat_cloud[bg_pred_cloud == class_id]
                current_prototypes[class_id.item()] = class_features.mean(dim=0)

            updated_prototypes = []
            updated_class_ids = []
            for class_id in range(
                self.cfg.data.num_bases, self.cfg.data.num_base_novels
            ):
                if class_id in current_prototypes:
                    updated_prototypes.append(current_prototypes[class_id])
                else:
                    updated_prototypes.append(
                        self.vlm_novel_proto[
                            class_id - self.cfg.data.num_bases
                        ].clone()
                    )
                updated_class_ids.append(class_id)
            updated_prototypes = torch.stack(updated_prototypes)

            # Identify unassigned points (-1) for refinement
            if "mask" in input_dict:
                mask_neg_one = (
                    (pred_cloud == -1) & (cur_label == -1) & (noda_mask == 1)
                )
            else:
                mask_neg_one = (pred_cloud == -1) & (cur_label == -1)
            feat_neg_one = feat_cloud[mask_neg_one]

            # Cosine similarity → assign label if above threshold
            similarities = torch.cosine_similarity(
                feat_neg_one[:, None, :],
                updated_prototypes[None, :, :],
                dim=-1,
            )
            max_similarity, max_class_idx = similarities.max(dim=1)
            refined_labels = pred_cloud.clone()
            refined_labels[mask_neg_one] = torch.where(
                max_similarity >= self.cfg.ai_thresh,
                torch.tensor(updated_class_ids, device=pred_cloud.device)[
                    max_class_idx
                ],
                -1,
            )
            new_predictions.append(refined_labels)

        # 4. Merge: keep GT labels, replace background with VLM predictions
        prediction = torch.cat(new_predictions, dim=0)
        label_one_hot = label.clone()
        label_one_hot[bg_label_mask] = prediction[bg_label_mask]

        return label_one_hot