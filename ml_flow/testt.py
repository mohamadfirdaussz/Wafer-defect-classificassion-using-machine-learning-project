
import numpy as np
data = np.load("C:/Users/user/OneDrive - ums.edu.my/FYP 1/data_loader_results/LSWMD_1500_preprocessed.npz", allow_pickle=True)
wafer_maps = data['wafer_maps']
print(wafer_maps.shape)
print(wafer_maps[0])
#
# Your output confirms the preprocessing pipeline worked correctly.
# ✅ Interpretation
#
# Shape (1296, 26, 26)
#
# There are 1,296 wafer samples.
# Each wafer map is a 26×26 matrix (die layout).
# Wafer map sample (as shown)
# Values 0, 1, 2 represent different die states:
# 0 → background / no die
# 1 → functional die
# 2 → defective die
#
# What this means
# Your .npz file now stores every wafer as a normalized 26×26 array ready for feature extraction (e.g. Radon, geometric, or region-based features).
# The accompanying .csv has all non-image metadata (lotName, failureType, etc.) synchronized with the same 1,296 samples.


