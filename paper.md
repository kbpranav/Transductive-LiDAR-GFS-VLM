# Generalized Few-Shot LiDAR Segmentation with Vision-Language Distillation: Extending GFS-VL from Indoor Scans to Outdoor Point Clouds

## Abstract

Generalized few-shot 3D point cloud segmentation (GFS-PCS) methods that distill knowledge from vision-language models (VLMs) have shown strong novel-class generalization on indoor benchmarks such as ScanNet, where scenes are dense, bounded, and captured with RGB-D sensors well-matched to the 2D VLMs supplying the distilled supervision. Autonomous driving, however, needs the same capability on sparse, large-scale, intensity-only outdoor LiDAR sweeps, where no prior GFS-VL-style result exists. We adapt **GFS-VL** — a framework that fuses dense-but-noisy VLM pseudo-labels with a handful of precise few-shot examples — to the **nuScenes** LiDAR segmentation benchmark, replacing its indoor RGB-D vision-language backbone with **OpenScene's OpenSeg** 2D-to-3D distilled features, re-tuning pseudo-label confidence filtering and loss weighting for outdoor class imbalance, and evaluating a Point Transformer V3 (PT-v3) backbone under a 12-base / 4-novel class split (novel: truck, motorcycle, bus, pedestrian). Averaged over 5 independent 5-shot registration runs, our adaptation reaches **66.2% mIoU** (67.1% base, 63.3% novel, 65.1% harmonic mean), with base-class pretraining alone reaching 73.2% mIoU. We analyze per-class behavior — including a clear forgetting effect on classes like trailer and construction-vehicle after few-shot registration — and outline transductive test-time refinement as the concrete next step toward closing the remaining base/novel gap.

## 1. Introduction

Semantic segmentation models deployed in the real world inevitably encounter object categories that were rare or entirely absent from their training distribution. The *generalized few-shot point cloud segmentation* (GFS-PCS) setting formalizes this: a model is pretrained on a set of well-annotated **base** classes, then must learn to segment a disjoint set of **novel** classes from only a handful (*k*-shot) of labeled examples each, while *retaining* base-class performance rather than catastrophically forgetting it.

GFS-VL (An et al., CVPR 2025) is the current state of the art for this setting on indoor benchmarks (ScanNet, ScanNet200, ScanNet++). Its key idea is to treat a 3D-distilled vision-language model as a source of *dense but noisy* supervision for novel classes — propagating pseudo-labels across an entire scene from VLM predictions — and to let the sparse, *precise* few-shot annotations correct and calibrate those pseudo-labels rather than being used in isolation. This combination is what lets the model generalize to novel classes beyond the literal k-shot examples it was given.

Every component of that pipeline, however, was designed around indoor RGB-D scans: dense per-point color, bounded room-scale geometry, and a 3D VLM (RegionPLC) distilled from indoor-domain 2D vision-language models. None of this is given for outdoor LiDAR: nuScenes point clouds carry only geometry and return intensity (no color), scenes are orders of magnitude larger and sparser, class frequencies are far more imbalanced (a handful of pedestrian/bicycle points against vast stretches of road and vegetation), and no RegionPLC-style indoor-tuned VLM exists for this domain. Whether GFS-VL's core idea — *dense noisy VLM knowledge + sparse precise few-shot examples* — transfers to this regime at all was, prior to this work, untested.

This paper reports on adapting GFS-VL end-to-end to nuScenes LiDAR segmentation, with the following specific contributions:

1. **VLM substitution.** We replace GFS-VL's indoor RegionPLC distillation source with **OpenScene's OpenSeg** weights, which distill 2D vision-language features into 3D specifically from outdoor, multi-camera driving data (nuScenes), and validate this VLM's own segmentation quality independently before trusting it as a pseudo-label source.
2. **Outdoor-appropriate re-tuning.** We lower VLM pseudo-label confidence thresholds, increase per-class loss weighting for the rarest novel class, and reduce voxelization grid size — all changes motivated by outdoor LiDAR's sparsity and class imbalance, as detailed in Section 3.
3. **Empirical characterization on nuScenes.** We report base-class pretraining performance, 5-shot registration results averaged over 5 independent registration sets, and a full per-class breakdown — surfacing a clear forgetting effect on several base classes after few-shot registration that is not visible from the summary mIoU alone.

## 2. Method

### 2.1 Task setup

We adopt nuScenes-lidarseg's 16 semantic classes, split into **12 base classes** (barrier, bicycle, car, construction-vehicle, traffic-cone, trailer, driveable-surface, other-flat, sidewalk, terrain, manmade, vegetation) and **4 novel classes** (truck, motorcycle, bus, pedestrian). The backbone is pretrained on the base classes only; it then undergoes *registration training*, during which it is given access to a **k=5-shot** support set per novel class and must learn to segment all 16 classes without forgetting the base 12. To control for the variance of which 5 examples per class are sampled, we run registration training independently for **5 different registration sets** (distinct random seeds) and report both per-set and averaged results.

### 2.2 Backbone and pretraining

The segmentation backbone is **Point Transformer V3 (PT-v3m1)**, operating on 4-channel input (xyz + intensity — no color, unlike the indoor setting). Pretraining uses `AdamW` (lr 0.002, one-cycle schedule down to 0.0002 for a dedicated backbone-block parameter group) for 30 epochs at a total batch size of 4, with a voxel grid size of 0.1m.

### 2.3 Vision-language distillation source

GFS-VL's novel-class pseudo-labels come from a 3D VLM that assigns an open-vocabulary class to every point in a scene without any task-specific labels. The original framework uses **RegionPLC**, trained/distilled for indoor scans. We instead distill from **OpenScene's OpenSeg weights** (`nuscenes_openseg.pth.tar`), which are specifically produced for outdoor driving scenes by projecting multi-view 2D CLIP-style features onto the LiDAR point cloud. We validate this VLM's raw segmentation quality directly against nuScenes ground truth across all 16 GFS classes as an independent sanity check *before* trusting its pseudo-labels inside the GFS-VL pipeline — since any weakness in the VLM's own outdoor class recognition would otherwise be invisible, masked by the few-shot correction signal, until much later in the pipeline.

### 2.4 Registration training

For each registration set, the few-shot support examples are used to build **per-class prototypes** in the backbone's feature space; the frozen VLM's point-wise predictions are then filtered against these prototypes to produce the dense pseudo-label supervision used for novel classes. Two filtering thresholds control how much of this noisy supervision is trusted:

- `ps_thresh` (pseudo-label survival threshold): lowered from the indoor-default 0.54 to **0.50** — outdoor novel-class points are far rarer per scene than indoor ones, so a stricter threshold left too few pseudo-labels for the few-shot prototypes to learn from.
- `ai_thresh` (ambiguity-interpolation threshold): lowered from 0.95 to **0.90** — at the default value, rare classes (notably bicycle) received essentially zero pseudo-labels in initial runs, starving their training signal entirely.

Training-loss class weighting is also adjusted for outdoor imbalance: novel classes truck, motorcycle, and pedestrian are weighted 2.0× relative to base classes, and **bicycle is weighted 5.0×** — a direct response to it being the single worst-performing class in early runs (see Section 3). Finally, the point-cloud voxelization grid size for registration training, support sets, and validation is reduced from 0.1m to **0.05m**, trading higher point density (and GPU memory — batch size is correspondingly reduced from 8 to 6) for finer geometric resolution on small novel objects such as pedestrians and bicycles.

### 2.5 Evaluation

We report per-class IoU and accuracy, overall mIoU, the mIoU restricted to base classes, the mIoU restricted to novel classes, and their harmonic mean (HM) — HM is the metric that best captures whether a model has genuinely learned novel classes *without* sacrificing base-class performance, since a model that ignores novel classes entirely would still score well on plain mIoU.

## 3. Results

### 3.1 Base-class pretraining

Base-only pretraining reaches **mIoU 0.7320 / mAcc 0.8004 / allAcc 0.9363** across the 12 base classes:

| Class | IoU | Accuracy |
|---|---|---|
| barrier | 0.6722 | 0.7476 |
| bicycle | 0.4183 | 0.5040 |
| car | 0.9451 | 0.9618 |
| construction-vehicle | 0.5896 | 0.6368 |
| traffic-cone | 0.5554 | 0.6537 |
| trailer | 0.7174 | 0.7529 |
| driveable-surface | 0.9571 | 0.9797 |
| other-flat | 0.6883 | 0.8037 |
| sidewalk | 0.7120 | 0.8443 |
| terrain | 0.7237 | 0.8139 |
| manmade | 0.9090 | 0.9652 |
| vegetation | 0.8964 | 0.9417 |

Large, geometrically regular classes (car, driveable-surface, manmade, vegetation) are already near or above 0.90 IoU at this stage. **Bicycle is the weakest base class by a wide margin (0.42 IoU)** even before any few-shot complications are introduced — its low point count and thin, irregular geometry make it the hardest class in the dataset overall, which directly motivated the 5× loss upweighting described in Section 2.4.

### 3.2 Few-shot registration training

Averaged across the 5 registration sets, final evaluation reaches:

| Metric | Regis1 | Regis2 | Regis3 | Regis4 | Regis5 | **Average** |
|---|---|---|---|---|---|---|
| mIoU | 0.6613 | **0.6742** | 0.6570 | 0.6593 | 0.6561 | **0.6616** |
| Base mIoU | 0.6675 | 0.6790 | 0.6714 | 0.6705 | 0.6670 | 0.6711 |
| Novel mIoU | 0.6429 | 0.6598 | 0.6138 | 0.6257 | 0.6234 | 0.6331 |
| HM (base/novel) | 0.6549 | 0.6692 | 0.6413 | 0.6474 | 0.6445 | 0.6515 |
| mAcc | 0.7204 | 0.7350 | 0.7182 | 0.7198 | 0.7150 | — |
| allAcc | 0.9308 | 0.9314 | 0.9306 | 0.9309 | 0.9299 | — |

Two observations stand out. First, **novel-class mIoU (0.6331 average) is within 4 points of base-class mIoU (0.6711 average)** despite novel classes having seen only 5 labeled examples each — this is the central result supporting the claim that GFS-VL's dense-VLM-plus-sparse-shots strategy transfers to outdoor LiDAR, not just indoor scans. Second, **variance across registration sets is non-trivial** (mIoU ranges from 0.6561 to 0.6742, a 1.8-point spread driven almost entirely by novel-class variance: 0.6138–0.6598), confirming that which 5 examples happen to be sampled per novel class still matters and that single-registration-set results would be unreliable without the 5-run average.

Per-class IoU reveals a pattern the aggregate numbers hide:

| Class | Base-only | Regis1 | Regis2 | Regis3 | Regis4 | Regis5 | Final Avg |
|---|---|---|---|---|---|---|---|
| barrier | 0.6722 | 0.6501 | 0.6277 | 0.5918 | 0.6561 | 0.6648 | 0.6381 |
| bicycle | 0.4183 | 0.2493 | 0.2987 | 0.2936 | 0.2604 | 0.2153 | 0.2634 |
| car | 0.9451 | 0.8849 | 0.8943 | 0.8924 | 0.8828 | 0.8882 | 0.8885 |
| construction-vehicle | 0.5896 | 0.3249 | 0.3921 | 0.4226 | 0.3985 | 0.4253 | 0.3927 |
| traffic-cone | 0.5554 | 0.5546 | 0.5496 | 0.5355 | 0.5444 | 0.5743 | 0.5517 |
| trailer | 0.7174 | 0.4931 | 0.5252 | 0.4698 | 0.4481 | 0.4270 | 0.4727 |
| driveable-surface | 0.9571 | 0.9603 | 0.9605 | 0.9603 | 0.9605 | 0.9598 | 0.9603 |
| other-flat | 0.6883 | 0.6513 | 0.6524 | 0.6473 | 0.6498 | 0.6167 | 0.6435 |
| sidewalk | 0.7120 | 0.7184 | 0.7236 | 0.7227 | 0.7197 | 0.7159 | 0.7201 |
| terrain | 0.7237 | 0.7443 | 0.7487 | 0.7443 | 0.7473 | 0.7420 | 0.7453 |
| manmade | 0.9090 | 0.8927 | 0.8895 | 0.8910 | 0.8928 | 0.8913 | 0.8914 |
| vegetation | 0.8964 | 0.8855 | 0.8854 | 0.8860 | 0.8859 | 0.8829 | 0.8851 |
| truck (novel) | — | 0.7093 | 0.7331 | 0.7112 | 0.6923 | 0.7009 | 0.7094 |
| motorcycle (novel) | — | 0.5255 | 0.5705 | 0.4555 | 0.5111 | 0.4852 | 0.5095 |
| bus (novel) | — | 0.7135 | 0.7110 | 0.6734 | 0.6824 | 0.6869 | 0.6934 |
| pedestrian (novel) | — | 0.6232 | 0.6243 | 0.6151 | 0.6172 | 0.6207 | 0.6201 |

**Every base class loses IoU relative to its base-only pretrained performance** once registration training is introduced, but the *size* of that drop is highly uneven. Geometrically regular, high-frequency classes are nearly unaffected (car: −0.057, driveable-surface: +0.003, manmade: −0.018, vegetation: −0.011). In sharp contrast, **trailer drops from 0.7174 to 0.4727 (−0.245)** and **construction-vehicle drops from 0.5896 to 0.3927 (−0.197)** — the two largest regressions in the whole table, both far exceeding what the 5-way base-vs-novel class reweighting alone would predict. We read this as evidence of a specific **forgetting interaction**: trailer and construction-vehicle are themselves rare, large-vehicle-like classes that are visually and geometrically confusable with the *novel* truck and bus classes the model is simultaneously being pushed (via upweighted loss and VLM pseudo-labels) to recognize — registration training appears to partially overwrite these base classes' decision boundaries in favor of their novel-class look-alikes, rather than forgetting uniformly across all base classes.

Among novel classes, **motorcycle is both the weakest (0.5095 average IoU) and the least stable across registration sets (0.4555–0.5705, a 0.115 spread)** — more than double the spread of any other novel class — indicating its 5-shot support sets are especially sensitive to which specific examples are drawn, plausibly because motorcycle shares a long, narrow silhouette with the base bicycle class at LiDAR resolution. **Bicycle remains the single hardest class in the entire taxonomy post-registration (0.2634 average IoU, actually lower than its already-weak 0.4183 base-only score)** despite the 5× loss upweighting introduced specifically to address it — indicating that loss reweighting alone was an insufficient fix and the class needs either more shots, better VLM pseudo-label coverage, or a different geometric representation to resolve from its sparse points.

## 4. Conclusion

We adapted the GFS-VL generalized few-shot point cloud segmentation framework from its original indoor, RGB-D setting to outdoor LiDAR segmentation on nuScenes, substituting OpenScene's OpenSeg weights for the original indoor RegionPLC VLM and re-tuning pseudo-label filtering, loss weighting, and voxelization resolution for outdoor class imbalance and sparsity. The resulting pipeline reaches 66.2% mIoU averaged over 5 independent 5-shot registration runs (67.1% base / 63.3% novel / 65.1% HM), demonstrating that GFS-VL's core strategy — dense-but-noisy VLM pseudo-labels corrected by a handful of precise examples — does transfer outside the indoor domain it was designed for, closing a gap left open by the original work. At the same time, our per-class analysis shows this transfer is uneven: it comes with a measurable, class-specific forgetting effect concentrated on base classes that are visually confusable with the novel classes being registered (trailer, construction-vehicle), and it does not resolve the dataset's hardest class (bicycle) despite targeted loss reweighting.

The natural next step — already scaffolded in this codebase but not yet evaluated — is **transductive test-time refinement** of the trained, frozen model: label propagation over a k-NN graph in feature space, BatchNorm statistic adaptation, Tent-style entropy minimization over norm-layer affine parameters, and a LiDAR-adapted port of the TransCLIP solver. Because these methods require no additional labels and operate purely at inference time on the unlabeled target scans, they are a direct lever for recovering some of the base/novel gap and the motorcycle/bicycle instability documented above without further changing what few-shot supervision the model receives. Running and reporting on these methods is the immediate continuation of this work.
