"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import random
import pickle
import glob
import numpy as np
from collections import defaultdict

from .builder import DATASETS
from .defaults import DefaultDataset
import pointcept.utils.comm as comm


# ───────────── GFS base / novel split constants ─────────────
KITTI_BASE_NAMES = [
    "car", "bicycle", "other-vehicle", "person", "motorcyclist",
    "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole",
]

KITTI_NOVEL_NAMES = ["motorcycle", "truck", "bicyclist", "traffic-sign"]

KITTI_BASE_NOVEL_NAMES = KITTI_BASE_NAMES + KITTI_NOVEL_NAMES

# Raw semantic label ID ↔ class name (19 classes)
KITTI_RAW_TO_NAME = {
    10: "car", 11: "bicycle", 15: "motorcycle", 18: "truck",
    20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist",
    40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
    50: "building", 51: "fence", 70: "vegetation", 71: "trunk",
    72: "terrain", 80: "pole", 81: "traffic-sign",
}
KITTI_NAME_TO_RAW = {v: k for k, v in KITTI_RAW_TO_NAME.items()}


@DATASETS.register_module()
class SemanticKITTIDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 7, 9, 10],
            val=[8],
            test=[5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv

@DATASETS.register_module()
class SemanticKITTIDataset_BASETrain(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 7, 9, 10],
            val=[8],
            test=[5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    # @staticmethod
    # def get_learning_map(ignore_index):
    #     learning_map = {
    #         0: ignore_index,  # "unlabeled"
    #         1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
    #         10: 0,  # "car"
    #         11: 1,  # "bicycle"
    #         13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    #         15: ignore_index,  # "motorcycle"
    #         16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    #         18: ignore_index,  # "truck"
    #         20: 4,  # "other-vehicle"
    #         30: 5,  # "person"
    #         31: ignore_index,  # "bicyclist"
    #         32: 7,  # "motorcyclist"
    #         40: 8,  # "road"
    #         44: 9,  # "parking"
    #         48: 10,  # "sidewalk"
    #         49: 11,  # "other-ground"
    #         50: 12,  # "building"
    #         51: 13,  # "fence"
    #         52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
    #         60: 8,  # "lane-marking" to "road" ---------------------------------mapped
    #         70: 14,  # "vegetation"
    #         71: 15,  # "trunk"
    #         72: 16,  # "terrain"
    #         80: 17,  # "pole"
    #         81: ignore_index,  # "traffic-sign"
    #         99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
    #         252: 0,  # "moving-car" to "car" ------------------------------------mapped
    #         253: ignore_index,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    #         254: 5,  # "moving-person" to "person" ------------------------------mapped
    #         255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    #         256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    #         257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    #         258: ignore_index,  # "moving-truck" to "truck" --------------------------------mapped
    #         259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
    #     }
    #     return learning_map

    # @staticmethod
    # def get_learning_map_inv(ignore_index):
    #     learning_map_inv = {
    #         ignore_index: ignore_index,  # "unlabeled"
    #         0: 10,  # "car"
    #         1: 11,  # "bicycle"
    #         2: 15,  # "motorcycle"
    #         3: 18,  # "truck"
    #         4: 20,  # "other-vehicle"
    #         5: 30,  # "person"
    #         6: 31,  # "bicyclist"
    #         7: 32,  # "motorcyclist"
    #         8: 40,  # "road"
    #         9: 44,  # "parking"
    #         10: 48,  # "sidewalk"
    #         11: 49,  # "other-ground"
    #         12: 50,  # "building"
    #         13: 51,  # "fence"
    #         14: 70,  # "vegetation"
    #         15: 71,  # "trunk"
    #         16: 72,  # "terrain"
    #         17: 80,  # "pole"
    #         18: 81,  # "traffic-sign"
    #     }
    #     return learning_map_inv
    
    @staticmethod
    def get_learning_map(ignore_index):
        # 15-Class Squeezed Map for Pre-training
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier"
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 2,  # "bus" -> "other-vehicle"
            15: ignore_index,  # NOVEL: "motorcycle" -> ignored
            16: 2,  # "on-rails" -> "other-vehicle"
            18: ignore_index,  # NOVEL: "truck" -> ignored
            20: 2,  # "other-vehicle"
            30: 3,  # "person"
            31: ignore_index,  # NOVEL: "bicyclist" -> ignored
            32: 4,  # "motorcyclist"
            40: 5,  # "road"
            44: 6,  # "parking"
            48: 7,  # "sidewalk"
            49: 8,  # "other-ground"
            50: 9,  # "building"
            51: 10, # "fence"
            52: ignore_index,  # "other-structure" -> ignored
            60: 5,  # "lane-marking" -> "road"
            70: 11, # "vegetation"
            71: 12, # "trunk"
            72: 13, # "terrain"
            80: 14, # "pole"
            81: ignore_index,  # NOVEL: "traffic-sign" -> ignored
            99: ignore_index,  # "other-object" -> ignored
            252: 0, # "moving-car" -> "car"
            253: ignore_index, # NOVEL: "moving-bicyclist" -> ignored
            254: 3, # "moving-person" -> "person"
            255: 4, # "moving-motorcyclist" -> "motorcyclist"
            256: 2, # "moving-on-rails" -> "other-vehicle"
            257: 2, # "moving-bus" -> "other-vehicle"
            258: ignore_index, # NOVEL: "moving-truck" -> ignored
            259: 2, # "moving-other-vehicle" -> "other-vehicle"
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        # Inverse map for the 15 Base Classes
        learning_map_inv = {
            ignore_index: ignore_index, 
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 20,  # "other-vehicle"
            3: 30,  # "person"
            4: 32,  # "motorcyclist"
            5: 40,  # "road"
            6: 44,  # "parking"
            7: 48,  # "sidewalk"
            8: 49,  # "other-ground"
            9: 50,  # "building"
            10: 51, # "fence"
            11: 70, # "vegetation"
            12: 71, # "trunk"
            13: 72, # "terrain"
            14: 80, # "pole"
        }
        return learning_map_inv


# ───────────── GFS Registration / Test Dataset Classes ─────────────

def _kitti_unified_learning_map(ignore_index):
    """Unified 19-class map: 15 base (0-14) + 4 novel (15-18)."""
    return {
        0: ignore_index, 1: ignore_index,        # unlabeled, outlier
        10: 0,              # car
        11: 1,              # bicycle
        13: 2,              # bus → other-vehicle
        15: 15,             # motorcycle (NOVEL)
        16: 2,              # on-rails → other-vehicle
        18: 16,             # truck (NOVEL)
        20: 2,              # other-vehicle
        30: 3,              # person
        31: 17,             # bicyclist (NOVEL)
        32: 4,              # motorcyclist
        40: 5,              # road
        44: 6,              # parking
        48: 7,              # sidewalk
        49: 8,              # other-ground
        50: 9,              # building
        51: 10,             # fence
        52: ignore_index,   # other-structure
        60: 5,              # lane-marking → road
        70: 11,             # vegetation
        71: 12,             # trunk
        72: 13,             # terrain
        80: 14,             # pole
        81: 18,             # traffic-sign (NOVEL)
        99: ignore_index,   # other-object
        252: 0,             # moving-car → car
        253: 17,            # moving-bicyclist → bicyclist (NOVEL)
        254: 3,             # moving-person → person
        255: 4,             # moving-motorcyclist → motorcyclist
        256: 2,             # moving-on-rails → other-vehicle
        257: 2,             # moving-bus → other-vehicle
        258: 16,            # moving-truck → truck (NOVEL)
        259: 2,             # moving-other-vehicle → other-vehicle
    }


def _kitti_unified_learning_map_inv(ignore_index):
    return {
        ignore_index: ignore_index,
        0: 10, 1: 11, 2: 20, 3: 30, 4: 32,
        5: 40, 6: 44, 7: 48, 8: 49, 9: 50,
        10: 51, 11: 70, 12: 71, 13: 72, 14: 80,
        15: 15, 16: 18, 17: 31, 18: 81,
    }


def _kitti_get_data_list(data_root, split):
    """Shared helper: build list of velodyne .bin paths for given split."""
    split2seq = dict(
        train=[0, 1, 2, 3, 4, 7, 9, 10],
        val=[8],
        test=[5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    )
    if isinstance(split, str):
        seq_list = split2seq[split]
    elif isinstance(split, list):
        seq_list = []
        for s in split:
            seq_list += split2seq[s]
    else:
        raise NotImplementedError
    data_list = []
    for seq in seq_list:
        seq = str(seq).zfill(2)
        seq_folder = os.path.join(data_root, "dataset", "sequences", seq)
        seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
        data_list += [
            os.path.join(seq_folder, "velodyne", f) for f in seq_files
        ]
    return data_list


def _kitti_load_scan(data_path, learning_map):
    """Shared helper: load a SemanticKITTI scan and apply learning_map."""
    with open(data_path, "rb") as b:
        scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
    coord = scan[:, :3]
    strength = scan[:, -1].reshape([-1, 1])
    label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
    if os.path.exists(label_file):
        with open(label_file, "rb") as a:
            segment = np.fromfile(a, dtype=np.int32).reshape(-1)
            segment = np.vectorize(learning_map.__getitem__)(
                segment & 0xFFFF
            ).astype(np.int32)
    else:
        segment = np.zeros(scan.shape[0]).astype(np.int32)
    return coord, strength, segment


def _kitti_data_name(data_path):
    dir_path, file_name = os.path.split(data_path)
    sequence_name = os.path.basename(os.path.dirname(dir_path))
    frame_name = os.path.splitext(file_name)[0]
    return f"{sequence_name}_{frame_name}"


def _kitti_get_data_list(data_root, split):
    """Shared helper: build list of velodyne .bin paths for given split."""
    split2seq = dict(
        train=[5, 6],
        val=[8],
        test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    )
    if isinstance(split, str):
        seq_list = split2seq[split]
    elif isinstance(split, list):
        seq_list = []
        for s in split:
            seq_list += split2seq[s]
    else:
        raise NotImplementedError
    data_list = []
    for seq in seq_list:
        seq = str(seq).zfill(2)
        seq_folder = os.path.join(data_root, "dataset", "sequences", seq)
        seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
        data_list += [
            os.path.join(seq_folder, "velodyne", f) for f in seq_files
        ]
    return data_list


@DATASETS.register_module()
class SemanticKITTIDataset_TEST(DefaultDataset):
    """
    SemanticKITTI dataset for testing on base and novel classes.
    Uses the unified 19-class (15 base + 4 novel) label space.
    """

    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = _kitti_unified_learning_map(ignore_index)
        self.learning_map_inv = _kitti_unified_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.base_class_names = list(KITTI_BASE_NAMES)
        self.novel_class_names = list(KITTI_NOVEL_NAMES)
    
    def get_data_list(self):
        return _kitti_get_data_list(self.data_root, self.split)

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        coord, strength, segment = _kitti_load_scan(data_path, self.learning_map)
        return dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )

    def get_data_name(self, idx):
        return _kitti_data_name(self.data_list[idx % len(self.data_list)])


@DATASETS.register_module()
class SemanticKITTIDataset_REGIS(DefaultDataset):
    """
    SemanticKITTI registration set.
    Selects k-shot frames per novel class from training sequences.
    Each sample returns the full scan with a binary mask for one novel class.
    """

    def __init__(self, seed=None, k_shot=None, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.bg_label = ignore_index
        self.learning_map = _kitti_unified_learning_map(ignore_index)

        self.base_class_names = list(KITTI_BASE_NAMES)
        self.novel_class_names = list(KITTI_NOVEL_NAMES)

        # Novel class raw label IDs
        self.novel_raw_ids = [KITTI_NAME_TO_RAW[n] for n in KITTI_NOVEL_NAMES]

        # raw label → unified index
        cls_to_idx = {name: idx for idx, name in enumerate(KITTI_BASE_NOVEL_NAMES)}
        self.raw_to_unified = {}
        for raw_id, name in KITTI_RAW_TO_NAME.items():
            if name in cls_to_idx:
                self.raw_to_unified[raw_id] = cls_to_idx[name]

        self.k_shot = k_shot
        self.seed = seed
        super().__init__(ignore_index=ignore_index, **kwargs)
        # Override data_list with registration entries after super().__init__
        self._create_data_list(seed)

    def get_data_list(self):
        return _kitti_get_data_list(self.data_root, self.split)

    def _create_data_list(self, seed):
        reg_file = os.path.join(
            self.data_root, f"sk_regis_{self.k_shot}_{seed}.txt"
        )
        if not os.path.exists(reg_file) and comm.is_main_process():
            class2scans_file = os.path.join(
                self.data_root, "sk_class2trainscans.pkl"
            )
            if not os.path.exists(class2scans_file):
                self._create_class2scans(class2scans_file)
            with open(class2scans_file, "rb") as f:
                class2scans = pickle.load(f)

            np.random.seed(seed)
            random.seed(seed)
            open(reg_file, "w").close()

            used_scans = []
            for novel_raw_id in self.novel_raw_ids:
                available = [
                    s for s in class2scans[novel_raw_id] if s not in used_scans
                ]
                selected = np.random.choice(
                    available, self.k_shot, replace=False
                )
                used_scans.extend(selected)
                with open(reg_file, "a") as f:
                    for rel_scan in selected:
                        f.write(f"{novel_raw_id}\t{rel_scan}\n")
        comm.synchronize()

        self.data_list = []
        with open(reg_file, "r") as f:
            for line in f.read().splitlines():
                class_id_str, rel_scan = line.split("\t")
                abs_scan = os.path.join(self.data_root, rel_scan)
                self.data_list.append((class_id_str, abs_scan))

    def _create_class2scans(self, class2scans_file):
        min_points = 100
        class2scans = {raw_id: [] for raw_id in self.novel_raw_ids}
        all_train_paths = _kitti_get_data_list(self.data_root, "train")
        for data_path in all_train_paths:
            label_file = data_path.replace("velodyne", "labels").replace(
                ".bin", ".label"
            )
            if not os.path.exists(label_file):
                continue
            with open(label_file, "rb") as f:
                labels = np.fromfile(f, dtype=np.int32).reshape(-1) & 0xFFFF
            for raw_id in self.novel_raw_ids:
                count = np.count_nonzero(labels == raw_id)
                if count > min_points:
                    rel_path = os.path.relpath(data_path, self.data_root)
                    class2scans[raw_id].append(rel_path)

        print("==== Class-to-scans mapping completed (SemanticKITTI) ====")
        for raw_id in self.novel_raw_ids:
            name = KITTI_RAW_TO_NAME[raw_id]
            assert (
                len(class2scans[raw_id]) > 0
            ), f"Class {raw_id} ({name}) has no data."
            print(
                f"\tClass {raw_id} | min_points: {min_points} "
                f"| Name: {name} | Scans: {len(class2scans[raw_id])}"
            )
        with open(class2scans_file, "wb") as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)

    def get_data(self, idx):
        class_id_str, data_path = self.data_list[idx % len(self.data_list)]
        class_raw_id = int(class_id_str)

        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(
            ".bin", ".label"
        )
        with open(label_file, "rb") as a:
            raw_labels = np.fromfile(a, dtype=np.int32).reshape(-1) & 0xFFFF

        # Binary mask: novel class → unified index, others → bg_label
        segment = np.full(raw_labels.shape, self.bg_label, dtype=np.int32)
        mask = raw_labels == class_raw_id
        if class_raw_id in self.raw_to_unified:
            segment[mask] = self.raw_to_unified[class_raw_id]

        return dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=_kitti_data_name(data_path),
        )

    def get_data_name(self, idx):
        _, data_path = self.data_list[idx % len(self.data_list)]
        return _kitti_data_name(data_path)


@DATASETS.register_module()
class SemanticKITTIDataset_REGISTrain(SemanticKITTIDataset_BASETrain):
    """
    SemanticKITTI dataset for registration training.
    Augments base training data with novel instance patches extracted
    from the registration set.
    """

    def __init__(
        self, nb_mix_blks=3, k_shot=5, seed=10, ignore_index=-1, **kwargs
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.nb_mix_blks = nb_mix_blks

        # Load registration file
        reg_file = os.path.join(
            self.data_root, f"sk_regis_{k_shot}_{seed}.txt"
        )
        self.regis_cls2scans = defaultdict(list)
        with open(reg_file, "r") as f:
            for line in f.read().splitlines():
                cls_id, rel_scan = line.split("\t")
                abs_scan = os.path.join(self.data_root, rel_scan)
                self.regis_cls2scans[cls_id].append(abs_scan)

        self.base_class_names = list(KITTI_BASE_NAMES)
        self.novel_class_names = list(KITTI_NOVEL_NAMES)

        # raw label → unified index
        cls_to_idx = {
            name: idx for idx, name in enumerate(KITTI_BASE_NOVEL_NAMES)
        }
        self.raw_to_unified = {}
        for raw_id, name in KITTI_RAW_TO_NAME.items():
            if name in cls_to_idx:
                self.raw_to_unified[raw_id] = cls_to_idx[name]

        self.novel_classes = list(self.regis_cls2scans.keys())

    def _get_instances_from_class(self, class_id_str):
        """
        Load novel instance samples for a given class from registration scans.
        Returns list of dicts with coord, strength, segment (binary mask).
        """
        instances = []
        class_raw_id = int(class_id_str)
        for scan_path in self.regis_cls2scans[class_id_str]:
            with open(scan_path, "rb") as b:
                scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
            coord = scan[:, :3]
            strength = scan[:, -1].reshape([-1, 1])

            label_file = scan_path.replace("velodyne", "labels").replace(
                ".bin", ".label"
            )
            with open(label_file, "rb") as a:
                raw_labels = (
                    np.fromfile(a, dtype=np.int32).reshape(-1) & 0xFFFF
                )

            segment = (raw_labels == class_raw_id).astype(np.int32)
            unified_idx = self.raw_to_unified.get(class_raw_id, -1)

            instances.append(
                dict(
                    coord=coord.astype(np.float32),
                    strength=strength.astype(np.float32),
                    segment=segment,
                    class_id=np.array([unified_idx], dtype=np.int32),
                )
            )
        return instances

    def _augment_with_novel_instances(
        self, train_coords, train_strengths, train_labels, novel_instances
    ):
        """
        Augment the training point cloud by inserting novel instance patches.
        Returns augmented coords, strengths, labels, and a mask
        (1 = original, 0 = novel).
        """
        scene_min = np.min(train_coords, axis=0)
        scene_z_min = scene_min[2]

        aug_coords = train_coords
        aug_strengths = train_strengths
        aug_labels = train_labels
        orig_mask = np.ones(train_coords.shape[0], dtype=np.int32)

        available_edges = ["bottom", "right", "top", "left"]

        for novel in novel_instances:
            inst_coords = novel["coord"]
            inst_strengths = novel["strength"]

            x_size = np.max(inst_coords[:, 0]) - np.min(inst_coords[:, 0])
            y_size = np.max(inst_coords[:, 1]) - np.min(inst_coords[:, 1])
            half_x, half_y = x_size / 2, y_size / 2

            target = inst_coords[novel["segment"] == 1]
            if len(target) == 0:
                continue
            x_center = (np.max(target[:, 0]) + np.min(target[:, 0])) / 2
            y_center = (np.max(target[:, 1]) + np.min(target[:, 1])) / 2

            mask_x = np.abs(inst_coords[:, 0] - x_center) <= half_x / 2
            mask_y = np.abs(inst_coords[:, 1] - y_center) <= half_y / 2
            square_mask = mask_x & mask_y

            if np.sum(square_mask & (novel["segment"] == 1)) < 0.1 * np.sum(
                novel["segment"] == 1
            ):
                rand_idx = np.random.choice(
                    np.where(novel["segment"] == 1)[0]
                )
                x_center = inst_coords[rand_idx, 0]
                y_center = inst_coords[rand_idx, 1]
                mask_x = np.abs(inst_coords[:, 0] - x_center) <= half_x / 2
                mask_y = np.abs(inst_coords[:, 1] - y_center) <= half_y / 2
                square_mask = mask_x & mask_y

            patch_coords = inst_coords[square_mask]
            patch_strengths = inst_strengths[square_mask]
            patch_labels = novel["segment"][square_mask]
            patch_labels = np.where(
                patch_labels == 1, novel["class_id"], -1
            )

            if not available_edges:
                break
            sel_edge = random.choice(available_edges)
            available_edges.remove(sel_edge)

            if sel_edge == "bottom":
                base_pt = train_coords[np.argmin(train_coords[:, 1])]
                inst_pt = patch_coords[np.argmax(patch_coords[:, 1])]
            elif sel_edge == "right":
                base_pt = train_coords[np.argmax(train_coords[:, 0])]
                inst_pt = patch_coords[np.argmin(patch_coords[:, 0])]
            elif sel_edge == "top":
                base_pt = train_coords[np.argmax(train_coords[:, 1])]
                inst_pt = patch_coords[np.argmin(patch_coords[:, 1])]
            else:  # left
                base_pt = train_coords[np.argmin(train_coords[:, 0])]
                inst_pt = patch_coords[np.argmax(patch_coords[:, 0])]

            translation = [
                base_pt[0] - inst_pt[0],
                base_pt[1] - inst_pt[1],
                0,
            ]
            translated_patch = patch_coords + translation
            translated_patch[:, 2] += scene_z_min - np.min(
                translated_patch[:, 2]
            )

            aug_coords = np.vstack((aug_coords, translated_patch))
            aug_strengths = np.vstack((aug_strengths, patch_strengths))
            aug_labels = np.concatenate((aug_labels, patch_labels))
            orig_mask = np.concatenate(
                (
                    orig_mask,
                    np.zeros(len(translated_patch), dtype=np.int32),
                )
            )

        return aug_coords, aug_strengths, aug_labels, orig_mask

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        coord, strength, segment = _kitti_load_scan(
            data_path, self.learning_map
        )

        # Sample novel classes and pick one instance per class
        n_mix = min(self.nb_mix_blks, len(self.novel_classes))
        selected_novel = random.sample(self.novel_classes, n_mix)
        novel_samples = []
        for cls in selected_novel:
            insts = self._get_instances_from_class(cls)
            assert len(insts) > 0
            novel_samples.extend(random.sample(insts, 1))

        # Augment training cloud with novel instance patches
        coord, strength, segment, mask = self._augment_with_novel_instances(
            coord, strength, segment, novel_samples
        )

        return dict(
            coord=coord,
            strength=strength,
            segment=segment,
            mask=mask,
            name=self.get_data_name(idx),
        )
