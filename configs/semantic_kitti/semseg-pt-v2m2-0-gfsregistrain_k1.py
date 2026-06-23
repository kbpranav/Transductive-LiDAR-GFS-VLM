_base_ = ["../_base_/default_runtime.py"]

# Trainer
train = dict(type="GFS_VL_Trainer")
# Tester
test = dict(type="GFSSemSegTester")

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = True
hooks = [
    dict(type="BackboneLoader", strict=False),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver_MultiRegis", save_freq=None),
    dict(type="GFS_MultiregTrain_PreciseEvaluator", test_last=True),
]
evaluate = False  # not evaluate after each epoch training process
vis = None  # visualization save path, set to None to disable visualization
ps_thresh = 0.6  # threshold for Pseudo-label Selection
ai_thresh = 0.95  # threshold for Adaptive Infilling
nb_mix_blks = 3  # number of novel blocks to mix

weight = ""  # trained model weight or checkpoint
vlm_3d_weight = (
    "./pretrain_weights/sparseunet32_636.pth"  # 3D VLM pretrained weight  # TODO: verify for SemanticKITTI
)
backbone_weight = "./exp/semantic_kitti/pt_v2_kitti_pretrain/model/model_best.pth"

num_base_classes = 15
num_novel_classes = 4

# model settings
model = dict(
    type="RegisTrainSegmentor",
    num_base_classes=num_base_classes,
    num_novel_classes=num_novel_classes,
    backbone_out_channels=48,  # = dec_channels[0] from PTv2 config
    backbone=dict(
        type="PT-v2m2",
        in_channels=4,
        num_classes=0,  # feature-only mode; seg head is in RegisTrainSegmentor
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.15, 0.375, 0.9375, 2.34375),  # x3, x2.5, x2.5, x2.5
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="map",  # map / interp
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 20
eval_epoch = 20
optimizer = dict(type="Adam", lr=0.001)
scheduler = dict(
    type="MultiStepLR",
    milestones=[50, 100],
    gamma=0.5,
)
param_dicts = [dict(keyword="backbone", lr=0.0001)]

# dataset settings
data_root = "data/semantic_kitti"
ignore_index = -1
k_shot = 1
regis_train_list = ["regis1", "regis2", "regis3", "regis4", "regis5"]

base_names = [
    "car", "bicycle", "other-vehicle", "person", "motorcyclist",
    "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole",
]
novel_names = ["motorcycle", "truck", "bicyclist", "traffic-sign"]
all_names = base_names + novel_names

data = dict(
    num_bases=num_base_classes,
    num_base_novels=num_base_classes + num_novel_classes,
    ignore_index=ignore_index,
    names=all_names,
    train=dict(
        type="SemanticKITTIDataset_REGISTrain",
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=120000, mode="random"),
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
        type="SemanticKITTIDataset_REGIS",
        split="train",
        seed=10,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
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
        type="SemanticKITTIDataset_REGIS",
        split="train",
        seed=20,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
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
        type="SemanticKITTIDataset_REGIS",
        split="train",
        seed=30,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
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
        type="SemanticKITTIDataset_REGIS",
        split="train",
        seed=40,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
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
        type="SemanticKITTIDataset_REGIS",
        split="train",
        seed=50,
        k_shot=k_shot,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
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
        type="SemanticKITTIDataset_TEST",
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
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
        type="SemanticKITTIDataset_TEST",
        split="val",
        data_root=data_root,
        transform=[
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
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
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
