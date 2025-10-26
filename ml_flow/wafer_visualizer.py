
import numpy as np
data = np.load("C:/Users/user/OneDrive - ums.edu.my/FYP 1/data_loader_results/cleaned_data.npz", allow_pickle=True)
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


#✅ Tafsiran

# Bentuk (1296, 26, 26)
# Terdapat 1,296 sampel wafer.
# Setiap peta wafer ialah matriks 26×26 (susun atur die).
# Contoh peta wafer (seperti ditunjukkan):
# Nilai 0, 1, 2 mewakili keadaan die yang berbeza:
#
# 0 → latar belakang / tiada die
# 1 → die berfungsi
# 2 → die rosak
#
# Maksudnya:
# Fail .npz anda kini menyimpan setiap wafer sebagai array 26×26 yang telah dinormalisasi, sedia untuk proses pengekstrakan ciri (contohnya Radon, geometri, atau berdasarkan kawasan).
# Fail .csv yang sepadan mengandungi semua metadata bukan imej (seperti lotName, failureType, dan lain-lain) yang diselaraskan dengan 1,296 sampel yang sama.