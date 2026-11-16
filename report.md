# Project Report — Transductive LiDAR GFS-VLM

**Last updated:** 2026-11-16 (written as a handoff/shelving doc before pausing work)
**Repo:** `gfs_vl/` → `git@github.com:kbpranav/Transductive-LiDAR-GFS-VLM.git`
**Active branch:** `new-novel` (all real work lives here; `main` is still just the untouched upstream merge)

> **If you're picking this back up later: read section 9 ("Uncommitted work") and section 11 ("Where things stand / next steps") first.** There is real uncommitted work sitting in the working tree.

---

## 1. What this project is

Goal: take **GFS-VL** — *Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Models* (Zhaochong An et al., CVPR 2025) — which was built and benchmarked for **indoor** scans (ScanNet / ScanNet200 / ScanNet++), and adapt it to **outdoor LiDAR** data (**nuScenes**). GFS-VL's core idea: combine *dense-but-noisy* knowledge distilled from a 3D vision-language model with a *small, precise* few-shot support set to let a segmentation model generalize to novel classes it only saw 1 or 5 labeled examples of.

The adaptation to LiDAR is the entire contribution of this repo on top of upstream GFS-VL. Nothing in the original codebase targeted nuScenes.

---

## 2. Workspace layout

`/home/kolarp/workspace/transductive-lidar/` is **not itself a git repo**. It's a workspace folder containing this project plus vendored reference code and raw data:

| Path | What it is |
|---|---|
| `gfs_vl/` | **This project.** Own repo, `kbpranav/Transductive-LiDAR-GFS-VLM`. |
| `gfs_vl/transduction-for-vlms/` | Vendored clone of `MaxZanella/transduction-for-vlms` — **TransCLIP** (Zanella et al., NeurIPS 2024), a transductive refinement method for CLIP zero/few-shot predictions. Used as **reference only** — its math was hand-ported into `transclip_inference.py` (see §6), it is never imported as a library. |
| `gfs_vl/pointcept/models/PLA/` | Vendored **RegionPLC** code (`CVMI-Lab/PLA`, `regionplc` branch) — the open-vocabulary 3D VLM machinery referenced by the original GFS-VL paper. Present but **not currently the VLM actually used for nuScenes** — see §6. |
| `Pointcept/` | Standalone clone of upstream `Pointcept/Pointcept` (the general 3D perception framework GFS-VL is built on). Separate from the copy vendored inside `gfs_vl/pointcept/`. |
| `old_stuff/GFS-3DSeg_GWs/` | Vendored clone of `Pixie8888/GFS-3DSeg_GWs` — *"Generalized Few-Shot Point Cloud Segmentation via Geometric Words"* (ICCV 2023). Reference/baseline, S3DIS-focused. |
| `old_stuff/GWGPTI/` | Just a notebook (`check_class_distribution.ipynb`) + a generic Pointcept build-command text file. **No README — what "GWGPTI" stands for is unclear; not resolved.** |
| `old_stuff/jong_project_code/` | An earlier/base few-shot SemanticKITTI pipeline (Cylinder3D-style base training + episodic few-shot fine-tuning + **RePRI**-style transductive inference). Looks like a predecessor to the current SemanticKITTI track in `gfs_vl/exp/semantic_kitti/`. |
| `datasets/` (169G) | **SemanticKITTI** raw data (odometry format, sequences 00–21). Used for the (currently minor, one-run) SemanticKITTI pretraining track. |
| `nuScenes/` (471G) | Full official nuScenes download: `v1.0-trainval`, `v1.0-test`, `v1.0-mini`, samples/sweeps/lidarseg/maps, raw archives. |
| `nuScenes_mini/` (9.0G) | A separate standalone mini-split download — redundant with the mini subset already inside `nuScenes/`; probably downloaded early for quick iteration before the full set was pulled. |
| `nuscenes-processed/` (530M) | **What training actually reads.** mmdet3d-style info pickles (`info/nuscenes_infos_10sweeps_{train,val,test}.pkl`) + few-shot registration split files (`nuscenes_regis_5_{10,20,30,40,50}.txt`, etc. — one per registration seed). Its `raw/` subfolder is empty; actual point data is read via paths inside the info pickles. |

Inside `gfs_vl/data/` (symlinks wired into the training pipeline):
```
data/nuscenes          -> ../../nuscenes-processed
data/semantic_kitti     -> ../../datasets
data/nuscenes_openseg.pth.tar   (187 MB, actual file — the VLM weight, see §6)
```
Note: the two huge raw nuScenes dirs (`nuScenes/`, `nuScenes_mini/`, ~480G combined) are **not** directly symlinked in — they were the upstream input to whatever produced `nuscenes-processed/`, which is the only thing the training code touches.

---

## 3. Starting point (upstream, `main` branch, commit `e5e2b1b`, 2026-06-23)

Single "Initial commit" merging three upstream codebases as-is:
- **GFS-VL** (the paper/framework itself)
- **OpenScene** components
- **Pointcept** (training/eval engine, model zoo, dataset loaders)

Configs shipped for ScanNet, ScanNet200, ScanNet++, and nuScenes-base-only configs — but no few-shot/novel-class nuScenes adaptation existed yet. `main` has never been touched since; it's the pristine upstream baseline.

---

## 4. Timeline of committed work (`new-novel` branch)

All dates below are commit dates, author Pranav Kolar Bhaskarpantula.

| Date | Commit | What it did |
|---|---|---|
| 2026-06-25 | `2c9cca2` | Swapped which nuScenes classes count as "novel": **bus ↔ bicycle**. Touched `nuscenes_constants.py`, `nuscenes.py`, the pretrain config, and training scripts. |
| 2026-06-30 | `7907e74` | Fixed pretrain batch size in the nuScenes pretrain config. |
| 2026-06-30 | `3358260` | Added `SemSegValTester` to `pointcept/engines/test.py` — a standalone quick-eval tester that replicates the per-epoch val-eval hook so you can re-run the same eval outside of training. Linked it into `scripts/train.sh`/command list. |
| 2026-06-30 | `c348f9e` | Added a script (`test_dataloader.py`) to sanity-check/generate novel-class prototype metadata before training. |
| 2026-06-30 | `7ae6f29` | Fixed the new-novel class ID mapping in the k5 registration-train config and `nuscenes.py`. |
| 2026-06-30 | `7550397` | Reworked VLM single-prototype extraction and fixed 3D feature normalization in `pointcept/engines/train.py` (83-line change — this is the core "how VLM knowledge is turned into novel-class prototypes" logic). |
| 2026-07-02 | `3c32497` | Fixed `timm.layers` deprecation warnings across PT-v3, SparseUNet, and Stratified Transformer model files. |
| 2026-07-02 | `e4b1545` | Added `run_gpu_cleaner.sh` (kills zombie GPU processes over a memory threshold) + more training command entries. |

This is the committed history as of `origin/new-novel`. **There is substantial additional work past this point that was never committed — see next section.**

---

## 5. nuScenes class split (current, final state)

From `pointcept/datasets/preprocessing/nuscenes/nuscenes_constants.py`:

- **16 total classes**, unified learning-ID space 0–15.
- **12 base classes:** barrier, bicycle, car, construction-vehicle, traffic-cone, trailer, driveable-surface, other-flat, sidewalk, terrain, manmade, vegetation.
- **4 novel classes** (few-shot registered): **truck, motorcycle, bus, pedestrian** — unified IDs 12–15 respectively in the `NUSCENES_BASE_NOVEL_NAMES` ordering used by configs (`test_dataloader.py` confirms: truck=12, motorcycle=13, bus=14, pedestrian=15).
- Note the file carries some historical cruft (an older commented-out base list, a `NUSCENES_LEARNING_ID_TO_NAME` dict whose inline "# NOVEL" comments correctly agree with the final novel set, but which is easy to misread next to `NUSCENES_LID_TO_SEMANTIC` — both agree on the same 4 novel classes, just don't be confused by having two parallel ID→name tables in the same file if revisiting this).

---

## 6. What VLM is actually used

This is a point worth being explicit about, because the README/citations reference a different VLM than what's actually wired up for nuScenes:

- The original GFS-VL paper (and the vendored `pointcept/models/PLA/` code) uses **RegionPLC** as the 3D VLM.
- For the nuScenes track, the configs (`configs/nuscenes/semseg-pt-v3m1-0-gfsregistrain_k{1,5}.py`) set `vlm_3d_weight = "./data/nuscenes_openseg.pth.tar"` — this is **OpenScene's "OpenSeg" 2D→3D distilled feature weight** (downloaded from `https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar`), **not** RegionPLC.
- `validate_vlm.py` (standalone script, see §7) exists specifically to validate this OpenScene/OpenSeg VLM's own segmentation quality on nuScenes' 16 GFS classes, independent of the GFS-VL pipeline — i.e. "is the VLM we're distilling from actually good at this before we blame the few-shot pipeline."
- `RegionPLC`/`PLA` code is present in the tree but, as far as the data wiring shows, **not the weight source currently driving the nuScenes pipeline.**

---

## 7. New/modified scripts (what each one does)

### Committed
- **`pointcept/engines/test.py` → `SemSegValTester`** (new class): quick single-pass val eval replicating the per-epoch eval hook, runnable standalone via `scripts/test.sh ... -o "test.type=SemSegValTester"`.
- **`test_dataloader.py`** (repo root, 48 lines): sanity-check script — builds a `NuScenesDataset_REGIS` (k=5) and a `NuScenesDataset_REGISTrain` (mix-blocks disabled) and asserts shapes/label ranges are correct (20 samples for 5-shot × 4 novel classes, correct label IDs, mask all-ones when `nb_mix_blks=0`). Also has the side effect of generating the `nuscenes_regis_5_10.txt` split file consumed by training.
- **`run_gpu_cleaner.sh`** (repo root): shell script that scans all GPUs via `nvidia-smi` and `kill -9`s any process using more than 30% of a GPU's memory — cleanup for orphaned/zombie training processes left behind after crashes.
- **`validate_vlm.py`** (repo root, 359 lines): standalone evaluation of the OpenScene OpenSeg VLM weight directly against nuScenes val GT across all 16 GFS classes, independent of the GFS-VL training pipeline. Includes a corrected CLIP-friendly class-name list (e.g. `"drivable surface"` not `"driveable-surface"`, `"person"` not `"pedestrian"`) needed to query the VLM's own text-embedding space correctly.

### Uncommitted, untracked (new, not yet in git — see §9)
- **`transductive_inference.py`** (515 lines): test-time refinement of an already-trained, **frozen** PT-v3 model — no weight updates. Per val scan: one forward pass → capture penultimate per-point features + softmax probs → build a k-NN graph (k=8 default) over points in feature space → run ~20 iterations of confidence-aware label propagation → compare refined vs. raw predictions against GT via per-class IoU. This is a hand-built Gaussian-kernel diffusion approach to transductive inference.
- **`transductive_inference_2.py`** (477 lines): two test-time-**adaptation** baselines (these *do* touch model state, unlike the script above) for comparison:
  1. `bn_adapt_inference` — BatchNorm running-stat adaptation only (no gradients): flips just the model's `BatchNorm1d` layers to `.train()`, accumulates a cumulative (not EMA) moving average over all target scans.
  2. `entropy_min_inference` — Tent-style (Wang et al. 2021) entropy minimization: only norm-layer affine params (`BatchNorm1d` + `LayerNorm` weight/bias) get gradients; everything else frozen, model stays in `.eval()` otherwise.
  Both run as two clean phases (adapt over all target scans, then a separate eval pass), not the original online single-batch Tent loop. Every experiment reloads a fresh copy of the checkpoint so the two methods never contaminate each other and the on-disk `.pth` is never touched.
- **`transclip_inference.py`** (489 lines): a sibling of `transductive_inference.py` that swaps the refinement step for a **port of the actual TransCLIP solver** (`transduction-for-vlms/TransCLIP_solver/TransCLIP_utils.py`), remapped from "CLIP image embeddings vs. text-prototype classes" to "per-point PTv3 features vs. segmentation classes." Model's own `softmax(seg_logits)` stands in for TransCLIP's cosine-similarity-to-text initial predictions; dense O(N²) CPU affinity is replaced with `torch_cluster.knn_graph` for tractability at per-scan point counts. TransCLIP's temperature/lambda/n_neighbors/max_iter/std_init hyperparameters (tuned originally for CLIP's feature scale) are all exposed as CLI flags rather than hardcoded, since PTv3's 64-dim feature space needs its own tuning.

**Important:** I found **no saved output/result logs anywhere for these three inference scripts** — they appear to be implemented but not yet (successfully) run end-to-end to produce recorded numbers. `transductive_inference_2.py`'s docstring references a "69% frozen-baseline" figure for context, but no artifact in the repo currently backs that number up with a log.

---

## 8. Key config settings (current state, `new-novel`)

### Pretraining — `configs/nuscenes/semseg-pt-v3m1-0-gfspretrain.py`
- `batch_size = 4` (total across GPUs), `epoch = 30`, `eval_epoch = 30`
- Optimizer: `AdamW`, `lr=0.002`, `weight_decay=0.005`; OneCycle-style `max_lr=[0.002, 0.0002]`; backbone-block param group at `lr=0.0002`
- `GridSample grid_size = 0.1` (**not** reduced to 0.05 like the registration configs — see §9, this is now an inconsistency worth resolving)
- Trains the base-class backbone only (12 base classes), later used as `backbone_weight` for registration training.

### Registration training (few-shot) — `configs/nuscenes/semseg-pt-v3m1-0-gfsregistrain_k5.py`
- Trainer: `GFS_VL_Trainer`; Tester: `GFSSemSegFastTester` (fast approx — switch to `GFSSemSegTester` for paper-faithful numbers)
- `batch_size = 6`, `num_worker = 6`, `epoch = 20`, `eval_epoch = 20`
- Optimizer: `Adam`, `lr=0.001`; backbone param group at `lr=0.0001`; `MultiStepLR` dropping at epochs 21/27 (gamma 0.5)
- `backbone_weight = "./exp/nuscenes/pt_v3_nuscenes_newnovel1_pretrain/model/model_best.pth"` — i.e. registration training starts from the `newnovel1` pretrain run's best checkpoint
- `vlm_3d_weight = "./data/nuscenes_openseg.pth.tar"` (OpenScene OpenSeg — see §6)
- VLM pseudo-label filtering thresholds **lowered** from defaults: `ps_thresh = 0.50` (was 0.54 — lets more VLM predictions survive filtering), `ai_thresh = 0.90` (was 0.95 — lets rare classes receive pseudo-labels at all)
- Loss class weighting: base classes weight 1.0; **novel classes up-weighted** — truck/motorcycle/pedestrian at 2.0, **bicycle at 5.0** (commented as "rarest class, received essentially no pseudo-labels in initial runs")
- `k_shot = 5`; 5 independent registration sets (`regis1..regis5`, seeds 10/20/30/40/50) — final reported performance is the average across all 5
- Model: PT-v3m1 backbone (`in_channels=4` — coord + intensity, no color/normal since this is LiDAR not RGB-D; `enc_depths=(2,2,2,6,2)`, `drop_path=0.3`, `pdnorm_conditions=("nuScenes","SemanticKITTI","Waymo")` — multi-domain point-cloud norm support carried over from upstream)
- `GridSample grid_size = 0.05` on **all** splits (train/regisN/val/test) — this was reduced from `0.1` in the uncommitted diff (see §9)

### K1 config (`semseg-pt-v3m1-0-gfsregistrain_k1.py`)
File timestamp is **older** (Apr 30) than the k5 config's recent edits — it has **not** received the grid_size=0.05 / batch_size / threshold tuning that k5 has. If resuming 1-shot experiments, check this config is still consistent with the k5 settings before trusting it.

---

## 9. Uncommitted work currently sitting in the working tree

`git status` on `gfs_vl` (branch `new-novel`) shows:

**Modified, not committed:**
- `configs/nuscenes/semseg-pt-v3m1-0-gfsregistrain_k5.py` — `batch_size` 8→6, `num_worker` 4→6, and **every** `GridSample grid_size` 0.1→0.05 (finer voxelization, presumably to recover detail on small novel objects like bicycles/pedestrians at the cost of more points/memory — consistent with the batch size being lowered at the same time).
- `pointcept/engines/hooks/misc.py` — `CheckpointSaver_MultiRegis` periodic epoch snapshots are now namespaced per registration set (`{regis_str}_epoch_{N}.pth` instead of a bare `epoch_{N}.pth` that would collide across regis1..regis5).
- `pointcept/engines/train.py` — **adds checkpoint-resume support** to `GFS_VL_Trainer`: a new `_resume_from_checkpoint` method restores model/optimizer/scheduler/scaler/best-metric state, and the main per-registration loop now has three paths: (a) fully completed registration → load weights, eval only (unchanged); (b) partial checkpoint + `cfg.resume=True` → resume training from the next un-run epoch; (c) partial checkpoint + `cfg.resume=False` → restart from scratch with a warning. A `cfg.resume_epoch` override lets you force-resume from an earlier periodic snapshot instead of the latest. (This pairs directly with the `misc.py` namespacing fix above — resuming needs per-regis snapshots to exist in the first place.) Also changed the trainer's internal `voxel_size` from `0.1` to `0.05` to match the config change.
- `scripts/train.sh` — `CUDA_VISIBLE_DEVICES` widened from `4,5,6,7` (4 GPUs) to `2,3,4,5,6,7` (6 GPUs), matching the k5 config's new `-g 6` usage.
- `scripts/gfs_pretrain_commands.txt` — new example commands appended: a `pt_v3_nuscenes_registrain_k5_newnovel3` run (with a follow-up `resume=true` command demonstrating the new resume feature), and a block of example invocations for all three new transductive-inference scripts (§7), including flag documentation (`--method bn_adapt|entropy_min|both`, `--limit`, `--bn-num-passes`, `--entropy-lr`, `--entropy-num-epochs`).

**Untracked (new files, never `git add`ed):**
- `transclip_inference.py`, `transductive_inference.py`, `transductive_inference_2.py` (§7)
- `transduction-for-vlms/` — the vendored TransCLIP reference clone itself (it has its own nested `.git`, so it won't get pulled into this repo's history by a normal `git add .`; needs an explicit decision — submodule, `.gitignore`, or `rm -rf .git` and vendor it flat — before committing).

**None of this is backed up in git right now.** If this machine/disk is at risk, `git add` + commit (or at least `git stash`) before walking away.

---

## 10. Experiment results so far

`gfs_vl/exp/` (20G) holds real training runs with checkpoints, TensorBoard event files, and a 55MB `train copy.log`. W&B tracking (`project="gfs-lidar"`) was used across ~25+ runs (`gfs_vl/wandb/`, project `gfs-lidar`).

**SemanticKITTI:** one run, `pt_v3_nuscenes_pretrain`-equivalent — `exp/semantic_kitti/pt_v2_kitti_pretrain` — minor/exploratory, not iterated on further.

**nuScenes runs (chronological):** `pt_v3_nuscenes_pretrain(_2/_eval)` (May 12) → `pt_v3_nuscenes_registrain_k5` and its `_2105`/`_29051`/`_0601`/`_0601_1`/`_0603` variants (May 21 – Jun 8, pre-"new-novel"-class-swap) → `pt_v3_nuscenes_newnovel1_pretrain` (Jun 30, pretrain redone after the bus↔bicycle novel-class swap) → `pt_v3_nuscenes_registrain_k5_newnovel`, `_newnovel2`, **`_newnovel3`** (Jul 1–8) — the latest, most complete run, using the current novel-class split.

**Latest results — `pt_v3_nuscenes_registrain_k5_newnovel3` (5 registration sets, final val eval per set):**

| Registration set | mIoU | Base mIoU | Novel mIoU | HM | mAcc | allAcc |
|---|---|---|---|---|---|---|
| regis1 | 0.6613 | 0.6675 | 0.6429 | 0.6549 | 0.7204 | 0.9308 |
| regis2 | **0.6742** (best overall) | 0.6790 | 0.6598 | 0.6692 | 0.7350 | 0.9314 |
| regis3 | 0.6570 | 0.6714 | 0.6138 | 0.6413 | 0.7182 | 0.9306 |
| regis4 | 0.6593 | 0.6705 | 0.6257 | 0.6474 | 0.7198 | 0.9309 |
| regis5 | 0.6561 | 0.6670 | 0.6234 | 0.6445 | 0.7150 | 0.9299 |
| **Average** | **≈0.6616** | ≈0.6711 | ≈0.6331 | ≈0.6515 | ≈0.7217 | ≈0.9307 |

(`regis2_model_best.pth` — the single best checkpoint, 0.6742 mIoU — is what the three new uncommitted transductive-inference scripts (§7, §9) are set up to run against by default in `scripts/gfs_pretrain_commands.txt`, but as noted in §7 no results from actually running them were found in the repo.)

---

## 11. Where things stand / next steps when resuming

1. **Commit or stash the uncommitted work first** (§9) — especially decide how to handle the nested `transduction-for-vlms/.git` before it gets `git add`ed as a mess, and commit the resume-training feature, the grid_size/batch_size tuning, and the three new inference scripts.
2. **Actually run the three transductive-inference scripts** (`transductive_inference.py`, `transductive_inference_2.py` with `--method bn_adapt|entropy_min|both`, `transclip_inference.py`) against `regis2_model_best.pth` and record the results — this is clearly the planned next experiment (commands are already drafted in `scripts/gfs_pretrain_commands.txt`) but hasn't been executed/logged yet as far as this repo shows.
3. **Reconcile the pretrain config's stale `grid_size=0.1`** with the registration configs' `grid_size=0.05` (§8) — currently pretrain and registration-train use different voxel resolutions, which may or may not be intentional.
4. **Bring `semseg-pt-v3m1-0-gfsregistrain_k1.py` up to date** with the tuning k5 has received (batch size, grid size, thresholds) if 1-shot results are still wanted — it looks untouched since before the new-novel-class work.
5. Resolve what `old_stuff/GWGPTI/` is for, if it still matters (currently undocumented, just a notebook + build-command text file).
6. Clean up: `gfs_vl/train copy.log` (55MB) and the large `wandb/` dir are local-only artifacts worth deciding whether to keep, move, or prune before any future `git add -A`.
