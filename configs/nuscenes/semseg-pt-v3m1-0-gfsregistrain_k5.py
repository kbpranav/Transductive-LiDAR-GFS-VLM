from pointcept.datasets.preprocessing.nuscenes.nuscenes_constants import (
    NUSCENES_BASE_NAMES,
    NUSCENES_NOVEL_NAMES,
    NUSCENES_BASE_NOVEL_NAMES,
)

_base_ = ["../_base_/default_runtime.py"]

# Trainer
train = dict(type="GFS_VL_Trainer")
# Tester (use GFSSemSegTester for precise paper numbers)
test = dict(type="GFSSemSegFastTester")

# misc custom setting
batch_size = 8  # bs: total bs in all gpus
num_worker = 4
mix_prob = 0
empty_cache = True
enable_amp = True

hooks = [
    dict(type="BackboneLoader", strict=False),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver_MultiRegis", save_freq=None),
    dict(type="GFS_SemSegEvaluator"),
    dict(type="WandbLogger", project="gfs-lidar", name="registrain-k5"),
]

evaluate = True
vis = None
ps_thresh = 0.50   # lowered from 0.54 — more VLM predictions survive filtering
ai_thresh = 0.90   # lowered from 0.95 — rare classes can now receive pseudo-labelsx
nb_mix_blks = 0

weight = ""
backbone_weight = "./exp/nuscenes/pt_v3_nuscenes_newnovel1_pretrain/model/model_best.pth"
vlm_3d_weight = "./data/nuscenes_openseg.pth.tar"

# model settings
model = dict(
    type="RegisTrainSegmentor",
    num_base_classes=len(NUSCENES_BASE_NAMES),
    num_novel_classes=len(NUSCENES_BASE_NOVEL_NAMES) - len(NUSCENES_BASE_NAMES),
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,                          # coord+strength, no color/normal
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(512, 512, 512, 512, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(512, 512, 512, 512),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
            # Upweight novel classes; bicycle (index 14) gets 5× because it
            # is the rarest class and received essentially no pseudo-labels
            # in initial runs.  Adjust if other classes degrade.
            weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # base 0-5
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # base 6-11
                2.0, 2.0, 5.0, 2.0,              # truck, motorcycle, bicycle, pedestrian
            ],
        ),
    ],
)

# scheduler settings
epoch = 20
eval_epoch = 20
optimizer = dict(type="Adam", lr=0.001)
scheduler = dict(
    type="MultiStepLR",
    milestones=[21, 27],  # drop lr at 70% and 90% of training
    gamma=0.5,
)
param_dicts = [dict(keyword="backbone", lr=0.0001)]

# dataset settings
data_root = "data/nuscenes"
ignore_index = -1
k_shot = 5
regis_train_list = ["regis1", "regis2", "regis3", "regis4", "regis5"]

# # shared transforms
# def nuscenes_train_transform():
#     return [
#         dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
#         dict(type="RandomScale", scale=[0.9, 1.1]),
#         dict(type="RandomFlip", p=0.5),
#         dict(type="RandomJitter", sigma=0.005, clip=0.02),
#         dict(
#             type="GridSample",
#             grid_size=0.05,
#             hash_type="fnv",
#             mode="train",
#             return_grid_coord=True,
#             keys=("coord", "strength", "segment"),
#         ),
#         dict(type="ToTensor"),
#         dict(
#             type="Collect",
#             keys=("coord", "grid_coord", "segment", "mask"),
#             feat_keys=("coord", "strength"),
#         ),
#     ]

# def nuscenes_regis_transform():
#     return [
#         dict(
#             type="GridSample",
#             grid_size=0.05,
#             hash_type="fnv",
#             mode="train",
#             return_grid_coord=True,
#             keys=("coord", "strength", "segment"),
#         ),
#         dict(type="ToTensor"),
#         dict(
#             type="Collect",
#             keys=("coord", "grid_coord", "segment", "name"),
#             feat_keys=("coord", "strength"),
#         ),
#     ]

data = dict(
    num_bases=len(NUSCENES_BASE_NAMES),
    num_base_novels=len(NUSCENES_BASE_NOVEL_NAMES),
    ignore_index=ignore_index,
    names=NUSCENES_BASE_NOVEL_NAMES,
    train=dict(
        type="NuScenesDataset_REGISTrain",
        split="train",
        data_root=data_root,
        sweeps=10,             # <-- ADD THIS
        k_shot=k_shot,         # <-- ADD THIS (uses the k_shot=5 defined earlier in your config)
        seed=10,               # <-- ADD THIS
        nb_mix_blks=0,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="SphereCrop", point_max=50000, mode="random"),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "strength", "segment"),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "mask"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    regis1=dict(
        type="NuScenesDataset_REGIS",
        split="train",
        seed=10,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "strength", "segment"),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    regis2=dict(
        type="NuScenesDataset_REGIS",
        split="train",
        seed=20,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "strength", "segment"),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    regis3=dict(
        type="NuScenesDataset_REGIS",
        split="train",
        seed=30,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "strength", "segment"),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    regis4=dict(
        type="NuScenesDataset_REGIS",
        split="train",
        seed=40,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "strength", "segment"),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    regis5=dict(
        type="NuScenesDataset_REGIS",
        split="train",
        seed=50,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "strength", "segment"),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type="NuScenesDataset_TEST",
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.1,
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
        ignore_index=ignore_index,
    ),
    test=dict(
        type="NuScenesDataset_TEST",
        split="val",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1])],
            ],
        ),
        ignore_index=ignore_index,
    ),
)