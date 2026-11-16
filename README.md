# Transductive LiDAR GFS-VLM

Adapting **GFS-VL** — *Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Models* (An et al., CVPR 2025) — from indoor RGB-D scans (ScanNet) to outdoor **LiDAR** segmentation on **nuScenes**.

GFS-VL's core idea is to combine *dense-but-noisy* pseudo-labels distilled from a 3D vision-language model with a handful of *precise* few-shot examples, so a model can generalize to novel classes it was given only 1 or 5 labeled points of. This repo ports that idea to a domain it was never built or tested for: sparse, color-less, large-scale outdoor point clouds, with a vision-language distillation source (OpenScene's OpenSeg) swapped in for the original indoor RegionPLC, and pseudo-label filtering / loss weighting / voxelization re-tuned for outdoor class imbalance.

For the full story and current results, see:
- **[`paper.md`](paper.md)** — workshop-paper-style writeup (what/why/how + results analysis).
- **[`report.md`](report.md)** — engineering handoff doc (workspace layout, exact settings, experiment history, open TODOs).

---

## What's here vs. upstream GFS-VL

This started as a straight merge of upstream GFS-VL + OpenScene + Pointcept (see `git log main`). Everything nuScenes-specific lives on the `new-novel` branch:

- `configs/nuscenes/semseg-pt-v3m1-0-gfspretrain.py` / `semseg-pt-v3m1-0-gfsregistrain_k{1,5}.py` — base pretraining and k-shot registration-training configs for nuScenes, with outdoor-tuned pseudo-label thresholds, loss weighting, and voxel grid size.
- `pointcept/datasets/nuscenes.py`, `pointcept/datasets/preprocessing/nuscenes/nuscenes_constants.py` — nuScenes dataset loaders and the 12-base/4-novel class split (novel: **truck, motorcycle, bus, pedestrian**).
- `pointcept/engines/train.py` (`GFS_VL_Trainer`) — VLM prototype extraction/normalization for LiDAR features, plus checkpoint-resume support across the 5 registration sets.
- `pointcept/engines/test.py` (`SemSegValTester`) — a standalone quick-eval tester.
- `validate_vlm.py` — validates the OpenScene VLM's own segmentation quality on nuScenes independently of the GFS-VL pipeline.
- `test_dataloader.py` — sanity-checks the few-shot dataset wiring before a training run.
- `transductive_inference.py`, `transductive_inference_2.py`, `transclip_inference.py` — inference-time refinement/adaptation of an already-trained, frozen checkpoint (label propagation, BatchNorm adaptation, entropy minimization, and a LiDAR-ported TransCLIP solver). No additional labels, no further few-shot supervision — see `paper.md` §4 for why these are the proposed next step.
- `run_gpu_cleaner.sh` — kills zombie GPU processes left behind by crashed runs.

## Vision-language model used

Registration training distills from **OpenScene's OpenSeg weights** (`data/nuscenes_openseg.pth.tar`, downloaded from `cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar`) rather than the original paper's RegionPLC — OpenSeg's 3D features are distilled from outdoor multi-camera driving data, which matches this domain. RegionPLC/PLA code is still vendored (`pointcept/models/PLA/`) for reference but is not the weight actually wired into the nuScenes configs.

## Dataset

**nuScenes-lidarseg**, 16 classes total:
- **Base (12):** barrier, bicycle, car, construction-vehicle, traffic-cone, trailer, driveable-surface, other-flat, sidewalk, terrain, manmade, vegetation.
- **Novel (4, few-shot):** truck, motorcycle, bus, pedestrian.

Training reads from `data/nuscenes` (symlinked to `../nuscenes-processed`, outside this repo) — mmdet3d-style info pickles plus per-seed few-shot registration split files. See `report.md` §2 for the full workspace/data layout.

---

## Installation

Same base dependency stack as upstream Pointcept/GFS-VL:

```bash
python -m venv gfs_vl
source gfs_vl/bin/activate

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install h5py pyyaml ninja sharedarray tensorboard tensorboardx addict einops scipy plyfile termcolor timm urllib3 fsspec==2024.2.0 easydict==1.13 yapf==0.40.1
pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
pip install spconv-cu118 # see https://github.com/traveller59/spconv for details

# Pointops CUDA dependencies
cd libs/pointops
TORCH_CUDA_ARCH_LIST="8.0" python setup.py install   # 8.0 for A100, 8.6 for RTX 3090
cd ../../libs/pointops2
TORCH_CUDA_ARCH_LIST="8.0" python setup.py install
```

CUDA 11.8+, PyTorch 1.13+. Tested on RTX 3090 / A100.

RegionPLC/`pointcept/models/PLA` build steps (needed only if you intend to use RegionPLC rather than OpenScene as the VLM) are unchanged from upstream — see that folder's own `README.md`. `transduction-for-vlms/` is a git submodule (reference code for the TransCLIP solver ported into `transclip_inference.py`); run `git submodule update --init` if you need its contents.

## Data preparation

nuScenes: follow the [Pointcept nuScenes data guidelines](https://github.com/Pointcept/Pointcept#data-preparation) to produce the `data_root` directory of info pickles, then point `data/nuscenes` at it (see `report.md` §2 for how this repo's own data currently is wired via symlinks). `test_dataloader.py` will also generate the few-shot registration split files (`nuscenes_regis_{k}_{seed}.txt`) the first time it's run against a new k-shot/seed combination.

## Training

**1. Base-class pretraining:**
```bash
sh scripts/train.sh -p python -d nuscenes -c semseg-pt-v3m1-0-gfspretrain -n ${EXP_NAME} -g 4
```

**2. Few-shot registration training** (5 independent registration sets, k=5 by default):
```bash
sh scripts/train.sh -p python -d nuscenes -c semseg-pt-v3m1-0-gfsregistrain_k5 -n ${EXP_NAME} -g 6 \
  -o "backbone_weight=./exp/nuscenes/${PRETRAIN_EXP_NAME}/model/model_best.pth"
```
Resume a partially completed registration run (see `report.md` §9 for exactly how this is resolved per registration set):
```bash
sh scripts/train.sh -p python -d nuscenes -c semseg-pt-v3m1-0-gfsregistrain_k5 -n ${EXP_NAME} -g 6 -o "resume=true"
```
`regis_train_list` in the config controls which registration sets run; final performance is reported as the average across all 5.

More worked examples (including the exact commands used for every run in `report.md`) are in `scripts/gfs_pretrain_commands.txt`.

## Evaluation

```bash
sh scripts/test.sh -d nuscenes -c semseg-pt-v3m1-0-gfsregistrain_k5 -n ${EXP_NAME} -w regis1_model_last -g 1
```
For a quick re-run of the same eval that fires after each training epoch (no fragment-based precise eval), use `SemSegValTester`:
```bash
sh scripts/test.sh -d nuscenes -c semseg-pt-v3m1-0-gfspretrain -n ${EXP_NAME} -w model_best -g 1 -o "test.type=SemSegValTester"
```

## Transductive / test-time refinement

Run against an already-trained, frozen registration checkpoint — no further labels needed:
```bash
# Confidence-aware label propagation
python transductive_inference.py --config-file configs/nuscenes/semseg-pt-v3m1-0-gfsregistrain_k5.py \
  --checkpoint ${CHECKPOINT_PATH} --k 8

# BN-stat adaptation / entropy minimization / both
python transductive_inference_2.py --checkpoint ${CHECKPOINT_PATH} --method bn_adapt|entropy_min|both

# TransCLIP-ported refinement
python transclip_inference.py --checkpoint ${CHECKPOINT_PATH}
```
All three accept `--limit N` for a quick smoke test on the first N scans. See `scripts/gfs_pretrain_commands.txt` for the full flag reference.

---

## Results (nuScenes, k=5, averaged over 5 registration sets)

| | mIoU | Base | Novel | HM |
|---|---|---|---|---|
| Base pretrain only | 0.7320 | — | — | — |
| Registration training | 0.6616 | 0.6711 | 0.6331 | 0.6515 |

See `paper.md` for the full per-class breakdown and analysis (including a clear forgetting effect on trailer/construction-vehicle, and bicycle remaining the hardest class despite loss upweighting).

---

## Citation

This repo builds directly on the GFS-VL framework and codebase. If you use this work, please cite the original paper:

```bibtex
@inproceedings{an2025generalized,
  title={Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model},
  author={An, Zhaochong and Sun, Guolei and Liu, Yun and Li, Runjia and Han, Junlin and Konukoglu, Ender and Belongie, Serge},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgement

Built on [GFS-VL](https://github.com/ZhaochongAn/GFS_VL) by Zhaochong An, which itself builds on [COSeg](https://github.com/ZhaochongAn/COSeg), [Pointcept](https://github.com/Pointcept/Pointcept), [RegionPLC](https://github.com/CVMI-Lab/PLA/tree/regionplc), and [OpenScene](https://github.com/pengsongyou/openscene). The TransCLIP solver ported into `transclip_inference.py` is from [transduction-for-vlms](https://github.com/MaxZanella/transduction-for-vlms) (Zanella et al., NeurIPS 2024).

Released under the same [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license as upstream GFS-VL (non-commercial use only).
