"""
nuScenes Dataset

Author: Pranav Kolar Bhaskarpantula (kolarp@informatik.uni-freiburg.de)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections.abc import Sequence
from collections import defaultdict
import pickle
import random
import torch
import pointcept.utils.comm as comm

from .builder import DATASETS
from .defaults import DefaultDataset

from .preprocessing.nuscenes.nuscenes_constants import (
    NUSCENES_LEARNING_MAP,
    NUSCENES_BASE_LEARNING_MAP,
    NUSCENES_BASE_ID_REMAP,
    NUSCENES_BASE_NAMES,
    NUSCENES_NOVEL_NAMES,
    NUSCENES_BASE_NOVEL_NAMES,
    NUSCENES_BASE_IDS,
    NUSCENES_NOVEL_IDS,
    CLASS_LABELS_NUSCENES,
    NUSCENES_LID_TO_SEMANTIC,
)


@DATASETS.register_module()
class NuScenesDataset(DefaultDataset):
    def __init__(self, sweeps=10, ignore_index=-1, **kwargs):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map



@DATASETS.register_module()
class NuScenesDataset_BASETrain(NuScenesDataset):
    """
    nuScenes dataset for GFS base-class training.

    Replaces the full 16-class learning map with a base-only map that:
      1. suppresses novel class points → ignore_index
      2. remaps the 12 surviving base learning IDs to contiguous 0..11
    """

    def __init__(self, sweeps=10, ignore_index=-1, **kwargs):
        super().__init__(sweeps=sweeps, ignore_index=ignore_index, **kwargs)

        # Build a raw-ID → contiguous-base-ID map, novel raw IDs → ignore_index
        self.learning_map = {
            raw: (
                NUSCENES_BASE_ID_REMAP[lid]
                if lid in NUSCENES_BASE_ID_REMAP
                else ignore_index
            )
            for raw, lid in NUSCENES_BASE_LEARNING_MAP.items()
        }

        self.base_index_mapping = {
            name: idx for idx, name in enumerate(NUSCENES_BASE_NAMES)
        }
@DATASETS.register_module()
class NuScenesDataset_GFS(NuScenesDataset):
    """
    Prototype class for GFS stages 2 and 3.
    Builds a unified label lookup (base 0..11, novel 12..15)
    and overrides learning_map so parent get_data() works unchanged.
    """

    def __init__(self, ignore_index=-1, **kwargs):
        self.bg_label = ignore_index
        super().__init__(ignore_index=ignore_index, **kwargs)

        self.base_class_names = list(NUSCENES_BASE_NAMES)
        self.novel_class_names = list(NUSCENES_NOVEL_NAMES)

        # Unified mapping: base first (0..11), novel after (12..15)
        cls_to_idx = {
            name: idx
            for idx, name in enumerate(self.base_class_names + self.novel_class_names)
        }

        # Override learning_map: raw_id → unified contiguous ID
        self.learning_map = {}
        for raw_id in range(32):
            lid = NUSCENES_LEARNING_MAP.get(raw_id, -1)
            if lid == -1:
                self.learning_map[raw_id] = ignore_index
            else:
                name = NUSCENES_LID_TO_SEMANTIC[lid]
                self.learning_map[raw_id] = cls_to_idx.get(name, ignore_index)



# Stage 3 — Test (base + novel)


@DATASETS.register_module()
class NuScenesDataset_TEST(NuScenesDataset_GFS):
    """
    nuScenes dataset for GFS testing on all 16 classes (base + novel).
    Inherits unified learning_map from GFS — no extra logic needed.
    Parent get_data() works as-is.
    """
    pass



# Stage 2 — Registration Support Set (k-shot)


@DATASETS.register_module()
class NuScenesDataset_REGIS(NuScenesDataset_GFS):
    """
    nuScenes k-shot registration (support) dataset.
    Provides k examples per novel class for the registration stage.
    Auto-generates the registration file if it doesn't exist.
    """

    def __init__(self, seed=None, k_shot=None, ignore_index=-1, **kwargs):
        super().__init__(ignore_index=ignore_index, **kwargs)

        # raw_id ↔ name mappings (first occurrence wins for multi-raw classes)
        self.orgid2name = {}
        self.name2orgid = {}
        for raw_id in range(32):
            lid = NUSCENES_LEARNING_MAP.get(raw_id, -1)
            if lid != -1:
                name = NUSCENES_LID_TO_SEMANTIC[lid]
                self.orgid2name[raw_id] = name
                if name not in self.name2orgid:  # first raw_id wins
                    self.name2orgid[name] = raw_id

        self.train_classes = [
            self.name2orgid[n] for n in self.base_class_names if n in self.name2orgid
        ]
        self.test_classes = [
            self.name2orgid[n] for n in self.novel_class_names if n in self.name2orgid
        ]
        self.all_classes_orgid = self.train_classes + self.test_classes

        self.k_shot = k_shot
        self._create_data_list(seed)

        # Cache token → info entry for fast lookup in get_data
        info_path = os.path.join(
            self.data_root, "info",
            f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
        )
        with open(info_path, "rb") as f:
            all_info = pickle.load(f)
        self.token2entry = {d["lidar_token"]: d for d in all_info}

    def get_data_name(self, idx):
        return self.data_list[idx % len(self.data_list)][1]
    def _create_data_list(self, seed):
        """
        Load registration file (generate if missing).
        File format: <raw_class_id>\t<lidar_token>
        """
        reg_file = os.path.join(
            self.data_root, f"nuscenes_regis_{self.k_shot}_{seed}.txt"
        )

        if not os.path.exists(reg_file) and comm.is_main_process():
            class2scans_file = os.path.join(
                self.data_root, "nuscenes_class2trainscans.pkl"
            )
            if not os.path.exists(class2scans_file):
                self._create_class2scans(class2scans_file)
            with open(class2scans_file, "rb") as f:
                self.class2scans = pickle.load(f)

            np.random.seed(seed)
            random.seed(seed)
            open(reg_file, "w").close()

            used_tokens = []
            for novel_cls in self.test_classes:
                available = [
                    t for t in self.class2scans[novel_cls]
                    if t not in used_tokens
                ]
                selected = np.random.choice(available, self.k_shot, replace=False)
                used_tokens.extend(selected.tolist())
                with open(reg_file, "a") as f:
                    for token in selected:
                        f.write(f"{novel_cls}\t{token}\n")

        comm.synchronize()

        self.data_list = []
        with open(reg_file, "r") as f:
            for line in f.read().splitlines():
                class_id_str, lidar_token = line.split("\t")
                self.data_list.append((class_id_str, lidar_token))

    def _create_class2scans(self, class2scans_file):
        """
        Build mapping: raw_class_id → list of lidar_tokens containing it.
        Filters frames with fewer than min_points of that class.
        """
        min_points = 50
        class2scans = {cls: [] for cls in self.all_classes_orgid}

        info_path = os.path.join(
            self.data_root, "info",
            f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
        )
        with open(info_path, "rb") as f:
            all_info = pickle.load(f)

        for entry in all_info:
            if "gt_segment_path" not in entry:
                continue
            gt_path = os.path.join(self.data_root, "raw", entry["gt_segment_path"])
            raw_seg = np.fromfile(str(gt_path), dtype=np.uint8, count=-1)
            token = entry["lidar_token"]
            for cls_id in np.unique(raw_seg):
                if cls_id not in self.all_classes_orgid:
                    continue
                if np.sum(raw_seg == cls_id) > min_points:
                    class2scans[cls_id].append(token)

        print("==== nuScenes class-to-scans mapping completed ====")
        for cls_id in self.all_classes_orgid:
            name = self.orgid2name.get(cls_id, str(cls_id))
            assert len(class2scans[cls_id]) > 0, (
                f"Class {cls_id} ({name}) has no frames. Lower min_points threshold."
            )
            print(f"\t Class {cls_id} | {name} | frames: {len(class2scans[cls_id])}")

        with open(class2scans_file, "wb") as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)

    def get_data(self, idx):
        """
        Load a k-shot support sample.
        Returns binary segment: novel class → unified_id, everything else → bg_label.
        """
        class_id_str, lidar_token = self.data_list[idx % len(self.data_list)]
        class_id = int(class_id_str)
        entry = self.token2entry[lidar_token]

        lidar_path = os.path.join(self.data_root, "raw", entry["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255

        gt_path = os.path.join(self.data_root, "raw", entry["gt_segment_path"])
        raw_seg = np.fromfile(str(gt_path), dtype=np.uint8, count=-1).reshape([-1])

        unified_id = self.learning_map[class_id]
        seg = np.where(raw_seg == class_id, unified_id, self.bg_label).astype(np.int64)

        return dict(coord=coord, strength=strength, segment=seg, name=lidar_token)



# Stage 2 — Registration Training


@DATASETS.register_module()
class NuScenesDataset_REGISTrain(NuScenesDataset_BASETrain):
    """
    nuScenes dataset for GFS registration training.
    Full training scenes (base labels) augmented with novel instance
    patches copy-pasted from the k-shot registration set.
    """

    def __init__(self, nb_mix_blks=3, k_shot=1, seed=10, **kwargs):
        super().__init__(**kwargs)
        self.nb_mix_blks = nb_mix_blks
        self.bg_label = self.ignore_index

        self.base_class_names = list(NUSCENES_BASE_NAMES)
        self.novel_class_names = list(NUSCENES_NOVEL_NAMES)

        # Load registration file
        reg_file = os.path.join(
            self.data_root, f"nuscenes_regis_{k_shot}_{seed}.txt"
        )
        self.regis_cls2tokens = defaultdict(list)
        with open(reg_file, "r") as f:
            for line in f.read().splitlines():
                class_id_str, lidar_token = line.split("\t")
                self.regis_cls2tokens[class_id_str].append(lidar_token)
        self.novel_classes = list(self.regis_cls2tokens.keys())

        # Unified label lookup: raw_id → unified ID (base 0..11, novel 12..15)
        cls_mapping = {
            name: idx
            for idx, name in enumerate(self.base_class_names + self.novel_class_names)
        }
        self.regis_lookup = {}
        for raw_id in range(32):
            lid = NUSCENES_LEARNING_MAP.get(raw_id, -1)
            if lid == -1:
                self.regis_lookup[raw_id] = self.bg_label
            else:
                name = NUSCENES_LID_TO_SEMANTIC[lid]
                self.regis_lookup[raw_id] = cls_mapping.get(name, self.bg_label)

        # # Cache token → info entry for fast lookup
        # self.token2entry = {d["lidar_token"]: d for d in self.data_list}
        
        # Cache token → info entry for fast lookup.
        # Always load from train info pkl directly — registration tokens are train-split,
        # so this works regardless of what `split` was passed to this dataset.
        info_path = os.path.join(
            self.data_root, "info",
            f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
        )
        with open(info_path, "rb") as f:
            all_train_info = pickle.load(f)
        self.token2entry = {d["lidar_token"]: d for d in all_train_info}


        # Sanity check: novel raw IDs must map to GFS unified IDs 12..15
        # truck=raw23→12, motorcycle=raw21→13, bicycle=raw14→14, pedestrian=raw2→15
        expected = {23: 12, 21: 13, 14: 14, 2: 15}
        for raw_id, expected_uid in expected.items():
            assert self.regis_lookup[raw_id] == expected_uid, (
                f"regis_lookup[{raw_id}] = {self.regis_lookup[raw_id]}, "
                f"expected {expected_uid}. Check NUSCENES_BASE_NAMES ordering."
            )


    def _get_instances_from_class(self, class_id_str):
        """
        Load novel instance frames for a given raw class ID string.
        Returns list of dicts: coord, strength, segment (binary), class_id.
        """
        instances = []
        class_int = int(class_id_str)
        unified_id = self.regis_lookup[class_int]

        for lidar_token in self.regis_cls2tokens[class_id_str]:
            entry = self.token2entry.get(lidar_token)
            if entry is None or "gt_segment_path" not in entry:
                continue

            lidar_path = os.path.join(self.data_root, "raw", entry["lidar_path"])
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
            coord = points[:, :3]
            strength = points[:, 3].reshape([-1, 1]) / 255

            gt_path = os.path.join(self.data_root, "raw", entry["gt_segment_path"])
            raw_seg = np.fromfile(str(gt_path), dtype=np.uint8, count=-1).reshape([-1])

            # Binary mask: 1 where this novel class exists
            binary_seg = (raw_seg == class_int).astype(np.int32)

            instances.append(dict(
                coord=coord,
                strength=strength,
                segment=binary_seg,
                class_id=np.array([unified_id], dtype=np.int32),
                name=lidar_token,
            ))
        return instances

    def _augment_with_novel_samples(self, coords, strength, labels, novel_samples):
        """
        Insert novel instance patches at scene edges.
        Returns augmented arrays + mask (1=original, 0=inserted novel).
        """
        scene_z_min = np.min(coords[:, 2])
        aug_coords = coords
        aug_strength = strength
        aug_labels = labels
        orig_mask = np.ones(coords.shape[0], dtype=np.int32)
        available_edges = ["bottom", "right", "top", "left"]

        for sample in novel_samples:
            inst_coords = sample["coord"]
            inst_strength = sample["strength"]

            # Compute bounding box of the full instance frame
            x_size = np.max(inst_coords[:, 0]) - np.min(inst_coords[:, 0])
            y_size = np.max(inst_coords[:, 1]) - np.min(inst_coords[:, 1])
            half_x = x_size / 2
            half_y = y_size / 2

            # Find center of the novel object (where segment == 1)
            target = inst_coords[sample["segment"] == 1]
            if len(target) == 0:
                continue
            x_center = (np.max(target[:, 0]) + np.min(target[:, 0])) / 2
            y_center = (np.max(target[:, 1]) + np.min(target[:, 1])) / 2

            # Square crop around the object center
            mask_x = np.abs(inst_coords[:, 0] - x_center) <= half_x / 2
            mask_y = np.abs(inst_coords[:, 1] - y_center) <= half_y / 2
            square_mask = mask_x & mask_y

            # Fallback: if crop misses most of object, pick random object point as center
            if np.sum(square_mask & (sample["segment"] == 1)) < 0.1 * np.sum(sample["segment"] == 1):
                rand_idx = np.random.choice(np.where(sample["segment"] == 1)[0])
                x_center = inst_coords[rand_idx, 0]
                y_center = inst_coords[rand_idx, 1]
                mask_x = np.abs(inst_coords[:, 0] - x_center) <= half_x / 2
                mask_y = np.abs(inst_coords[:, 1] - y_center) <= half_y / 2
                square_mask = mask_x & mask_y

            patch_coords = inst_coords[square_mask]
            patch_strength = inst_strength[square_mask]
            patch_labels = sample["segment"][square_mask]
            # Map binary mask → unified novel class ID; background → ignore
            patch_labels = np.where(
                patch_labels == 1, sample["class_id"], self.bg_label
            )

            # Pick a scene edge and compute translation to place patch there
            if len(available_edges) == 0:
                available_edges = ["bottom", "right", "top", "left"]
            sel_edge = random.choice(available_edges)
            available_edges.remove(sel_edge)

            if sel_edge == "bottom":
                base_pt = coords[np.argmin(coords[:, 1])]
                inst_pt = patch_coords[np.argmax(patch_coords[:, 1])]
            elif sel_edge == "right":
                base_pt = coords[np.argmax(coords[:, 0])]
                inst_pt = patch_coords[np.argmin(patch_coords[:, 0])]
            elif sel_edge == "top":
                base_pt = coords[np.argmax(coords[:, 1])]
                inst_pt = patch_coords[np.argmin(patch_coords[:, 1])]
            else:  # left
                base_pt = coords[np.argmin(coords[:, 0])]
                inst_pt = patch_coords[np.argmax(patch_coords[:, 0])]

            translation = np.array([base_pt[0] - inst_pt[0], base_pt[1] - inst_pt[1], 0.0])
            translated_patch = patch_coords + translation
            # Align patch floor to scene floor
            translated_patch[:, 2] += scene_z_min - np.min(translated_patch[:, 2])

            aug_coords = np.vstack((aug_coords, translated_patch))
            aug_strength = np.vstack((aug_strength, patch_strength))
            aug_labels = np.concatenate((aug_labels, patch_labels))
            orig_mask = np.concatenate(
                (orig_mask, np.zeros(len(translated_patch), dtype=np.int32))
            )

        return aug_coords, aug_strength, aug_labels, orig_mask

    @staticmethod
    def _fromfile_with_retry(path, dtype, max_retries=5):
        for attempt in range(max_retries):
            try:
                return np.fromfile(path, dtype=dtype, count=-1)
            except (PermissionError, OSError):
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = self._fromfile_with_retry(str(lidar_path), dtype=np.float32).reshape([-1, 5])
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255

        if "gt_segment_path" in data:
            gt_path = os.path.join(self.data_root, "raw", data["gt_segment_path"])
            raw_seg = self._fromfile_with_retry(str(gt_path), dtype=np.uint8).reshape([-1])
            # Base-only map: novel → ignore_index, base → contiguous 0..11
            segment = np.vectorize(self.learning_map.__getitem__)(raw_seg).astype(np.int64)
        else:
            segment = np.ones(coord.shape[0], dtype=np.int64) * self.bg_label

        # Sample nb_mix_blks novel classes and get one instance each
        selected = random.sample(
            self.novel_classes, min(self.nb_mix_blks, len(self.novel_classes))
        )
        novel_samples = []
        for cls in selected:
            instances = self._get_instances_from_class(cls)
            if len(instances) > 0:
                novel_samples.extend(random.sample(instances, 1))

        # Copy-paste novel patches into the training scene
        coord, strength, segment, mask = self._augment_with_novel_samples(
            coord, strength, segment, novel_samples
        )

        return dict(
            coord=coord,
            strength=strength,
            segment=segment,
            mask=mask,
            name=self.get_data_name(idx),
        )

# @DATASETS.register_module()
# class NuScenesDataset_TEST(NuScenesDataset):
#     """
#     nuScenes dataset for GFS testing on all 16 classes (base + novel).
#     Uses the full learning map and remaps to unified contiguous IDs.
#     """

#     def __init__(self, ignore_index=-1, **kwargs):
#         self.bg_label = ignore_index
#         super().__init__(ignore_index=ignore_index, **kwargs)

#         self.base_class_names = list(NUSCENES_BASE_NAMES)
#         self.novel_class_names = list(NUSCENES_NOVEL_NAMES)

#         # Build unified lookup: raw_id → contiguous unified ID (0..15)
#         cls_mapping = {
#             name: idx
#             for idx, name in enumerate(
#                 self.base_class_names + self.novel_class_names
#             )
#         }
#         self.unified_lookup = {}
#         for raw_id in range(32):
#             lid = NUSCENES_LEARNING_MAP.get(raw_id, -1)
#             if lid == -1:
#                 self.unified_lookup[raw_id] = ignore_index
#             else:
#                 name = NUSCENES_LID_TO_SEMANTIC[lid]
#                 self.unified_lookup[raw_id] = cls_mapping.get(name, ignore_index)

#         # Override learning_map with unified lookup
#         self.learning_map = self.unified_lookup

#     def get_data(self, idx):
#         data = self.data_list[idx % len(self.data_list)]
#         lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
#         points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
#             [-1, 5]
#         )
#         coord = points[:, :3]
#         strength = points[:, 3].reshape([-1, 1]) / 255

#         if "gt_segment_path" in data:
#             gt_path = os.path.join(self.data_root, "raw", data["gt_segment_path"])
#             raw_seg = np.fromfile(str(gt_path), dtype=np.uint8, count=-1).reshape([-1])
#             segment = np.vectorize(self.learning_map.__getitem__)(raw_seg).astype(
#                 np.int64
#             )
#         else:
#             segment = np.ones(coord.shape[0], dtype=np.int64) * self.bg_label

#         return dict(
#             coord=coord,
#             strength=strength,
#             segment=segment,
#             name=self.get_data_name(idx),
#         )

