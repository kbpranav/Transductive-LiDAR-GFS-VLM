import numpy as np
from pointcept.datasets import build_dataset

# New novel classes: truck, motorcycle, bus, pedestrian  (unified IDs 12–15)
# New base classes:  barrier, bicycle, car, construction-vehicle, traffic-cone,
#                    trailer, driveable-surface, other-flat, sidewalk,
#                    terrain, manmade, vegetation  (unified IDs 0–11)

# Test _REGIS (k=5 so it also generates nuscenes_regis_5_10.txt for REGISTrain below)
regis = build_dataset(dict(
    type="NuScenesDataset_REGIS",
    data_root="data/nuscenes",
    split="train",
    sweeps=10,
    k_shot=5,
    seed=10,
    ignore_index=-1,
))
print(f"REGIS length: {len(regis.data_list)}")  # expect 20 (5 shots × 4 novel classes)
assert len(regis.data_list) == 20, f"Expected 20, got {len(regis.data_list)}"

sample = regis.get_data(0)
unique = np.unique(sample["segment"])
print(f"REGIS segment unique values: {unique}")
# Must be exactly two values: [-1, one_of(12,13,14,15)]
# truck=12, motorcycle=13, bus=14, pedestrian=15
assert len(unique) == 2
assert unique[0] == -1
assert unique[1] in [12, 13, 14, 15]

# Test _REGISTrain with nb_mix_blks=0
train = build_dataset(dict(
    type="NuScenesDataset_REGISTrain",
    data_root="data/nuscenes",
    split="train",
    sweeps=10,
    k_shot=5,
    seed=10,
    nb_mix_blks=0,
    ignore_index=-1,
))
sample = train.get_data(0)
unique = np.unique(sample["segment"])
print(f"REGISTrain segment unique values: {unique}")
# Must only contain {-1, 0..11} — no novel IDs 12..15 since nb_mix_blks=0
assert all(v == -1 or 0 <= v <= 11 for v in unique)
assert "mask" in sample
assert np.all(sample["mask"] == 1)  # all-ones since nb_mix_blks=0
print("All checks passed. Ready to train.")