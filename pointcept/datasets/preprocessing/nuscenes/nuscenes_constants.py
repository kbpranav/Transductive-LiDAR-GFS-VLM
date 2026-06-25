# ───────────── nuScenes GFS learning map constants ─────────────

# Full 16-class learning map (raw ID → learning ID)
NUSCENES_LEARNING_MAP = {
    0: -1, 1: -1, 2: 6, 3: 6, 4: 6, 5: -1, 6: 6, 7: -1, 8: -1,
    9: 0, 10: -1, 11: -1, 12: 7, 13: -1, 14: 1, 15: 2, 16: 2,
    17: 3, 18: 4, 19: -1, 20: -1, 21: 5, 22: 8, 23: 9, 24: 10,
    25: 11, 26: 12, 27: 13, 28: 14, 29: -1, 30: 15, 31: -1,
}

# Learning ID → class name (all 16 classes)
NUSCENES_LEARNING_ID_TO_NAME = {
    0:  "barrier",
    1:  "bicycle",            
    2:  "bus",                # NOVEL
    3:  "car",
    4:  "construction-vehicle",
    5:  "motorcycle",         # NOVEL
    6:  "traffic-cone",
    7:  "trailer",
    8:  "truck",              # NOVEL
    9:  "driveable-surface",
    10: "other-flat",
    11: "sidewalk",
    12: "terrain",
    13: "manmade",
    14: "vegetation",
    15: "pedestrian",         # NOVEL
}

# Ordered list of all 16 class names (index = learning ID)
CLASS_LABELS_NUSCENES = [
    NUSCENES_LEARNING_ID_TO_NAME[i] for i in range(16)
]

# NUSCENES_BASE_NAMES = [
#     "barrier", "bus", "car", "construction-vehicle",
#     "traffic-cone", "trailer",
#     "driveable-surface", "other-flat", "sidewalk",
#     "terrain", "manmade", "vegetation",
# ]

NUSCENES_BASE_NAMES = [
    "barrier", "bicycle", "car", "construction-vehicle",
    "traffic-cone", "trailer",
    "driveable-surface", "other-flat", "sidewalk",
    "terrain", "manmade", "vegetation",
]

NUSCENES_NOVEL_NAMES = ["truck", "motorcycle", "bus", "pedestrian"]

NUSCENES_BASE_NOVEL_NAMES = NUSCENES_BASE_NAMES + NUSCENES_NOVEL_NAMES

# Learning IDs for base and novel classes
# Correct semantic name for each learning ID, derived from NUSCENES_LEARNING_MAP:
#   raw 9→lid0 barrier, raw14→lid1 bicycle, raw15/16→lid2 bus, raw17→lid3 car,
#   raw18→lid4 construction-vehicle, raw21→lid5 motorcycle,
#   raw2/3/4/6→lid6 pedestrian, raw12→lid7 traffic-cone, raw22→lid8 trailer,
#   raw23→lid9 truck, raw24→lid10 driveable-surface, raw25→lid11 other-flat,
#   raw26→lid12 sidewalk, raw27→lid13 terrain, raw28→lid14 manmade, raw30→lid15 vegetation
NUSCENES_LID_TO_SEMANTIC = [
    "barrier",              # lid 0
    "bicycle",              # lid 1
    "bus",                  # lid 2
    "car",                  # lid 3
    "construction-vehicle", # lid 4
    "motorcycle",           # lid 5
    "pedestrian",           # lid 6  (raw 2,3,4,6)
    "traffic-cone",         # lid 7  (raw 12)
    "trailer",              # lid 8  (raw 22)
    "truck",                # lid 9  (raw 23)
    "driveable-surface",    # lid 10 (raw 24)
    "other-flat",           # lid 11 (raw 25)
    "sidewalk",             # lid 12 (raw 26)
    "terrain",              # lid 13 (raw 27)
    "manmade",              # lid 14 (raw 28)
    "vegetation",           # lid 15 (raw 30)
]

# Base/novel learning IDs based on NUSCENES_LID_TO_SEMANTIC (not NUSCENES_LEARNING_ID_TO_NAME)
NUSCENES_BASE_IDS  = [0, 1, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15]
NUSCENES_NOVEL_IDS = [9, 5, 2, 6]   # truck=9, motorcycle=5, bus=2, pedestrian=6

# Base-only learning map: novel class raw IDs → -1
NUSCENES_BASE_LEARNING_MAP = {
    raw: (-1 if learning_id in NUSCENES_NOVEL_IDS else learning_id)
    for raw, learning_id in NUSCENES_LEARNING_MAP.items()
}

# Contiguous base ID remap: full learning ID → 0..11
NUSCENES_BASE_ID_REMAP = {
    full_id: base_id
    for base_id, full_id in enumerate(sorted(NUSCENES_BASE_IDS))
}