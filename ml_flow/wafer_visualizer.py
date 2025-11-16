# import pandas as pd

# # ubah path sesuai lokasi file
# pickle_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"

# # load dataset
# df = pd.read_pickle(pickle_path)

# # print daftar kolom
# print("Daftar kolom dalam dataset:")
# print(df.columns.tolist())

# # contoh menampilkan 5 baris awal
# print("\nContoh data:")
# print(df.head())



# #see npz file content
# import numpy as np

# npz_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_data.npz"

# # Load file
# data = np.load(npz_path, allow_pickle=True)

# # Print stored arrays
# print("Keys:", data.files)

# # Show shapes and sample values
# for key in data.files:
#     arr = data[key]
#     print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

# # Optional: show one sample
# print("\nExample wafer map shape:", data["X_train"][0].shape)
# print("Label example:", data["y_train"][0])


#✅ Semak kandungan fail npz visual wafer
# import numpy as np
# import matplotlib.pyplot as plt
# import random

# # Mapping used in your preprocessing script
# label_map = {
#     0: "Center",
#     1: "Donut",
#     2: "Edge-Loc",
#     3: "Edge-Ring",
#     4: "Loc",
#     5: "Random",
#     6: "Scratch",
#     7: "none"
# }

# def load_npz(npz_path):
#     data = np.load(npz_path, allow_pickle=True)
#     return data["X_train"], data["y_train"]

# def plot_5_each_failure(X, y):
#     failure_types = sorted(np.unique(y))
#     num_cat = len(failure_types)
#     sample_size = 5

#     fig, axs = plt.subplots(num_cat, sample_size, figsize=(30, 30))

#     for r, cls in enumerate(failure_types):
#         idxs = np.where(y == cls)[0]
#         random.seed(10)
#         chosen = random.sample(list(idxs), min(sample_size, len(idxs)))

#         for c, idx in enumerate(chosen):
#             axs[r, c].axis('off')
#             axs[r, c].imshow(X[idx], cmap="binary")  # matches xichen notebook style
#             axs[r, c].set_title(f"{cls} | {label_map[int(cls)]}", fontsize=12)

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     npz_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_data.npz"
#     X_train, y_train = load_npz(npz_path)
#     plot_5_each_failure(X_train, y_train)






# import numpy as np
# data = np.load(r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_data.npz", allow_pickle=True)
# wafer_maps = data['wafer_maps']
# print(wafer_maps.shape)
# print(wafer_maps[0])
# #
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


# import numpy as np
# import os

# # --- 1. Define the path to your file ---
# # This path must match the output from your data_preprocessor.py script
# file_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\preprocessing_results\model_ready_data.npz"


# # --- 2. Check if the file exists ---
# if not os.path.exists(file_path):
#     print(f"❌ ERROR: File not found at {file_path}")
#     print("Please check the path and re-run data_preprocessor.py if needed.")

# else:
#     print(f"✅ Loading and inspecting file: {file_path}\n")
    
#     # --- 3. Load the .npz file ---
#     # We use 'with' so it automatically closes the file after
#     with np.load(file_path, allow_pickle=True) as data:
        
#         # --- 4. List all the arrays stored in the file ---
#         print("--- 1. ARRAYS IN FILE ---")
#         print(data.files)
        
#         # --- 5. Print the shape of each array ---
#         print("\n--- 2. ARRAY SHAPES ---")
#         for key in data.files:
#             print(f"Shape of '{key}': \t{data[key].shape}")
            
#         # --- 6. Print a sample of the data ---
#         print("\n--- 3. DATA SAMPLES ---")
        
#         # Check X_train (should be scaled numbers, not 0s and 1s)
#         print(f"\nSample from 'X_train' (first 5 features of first row):")
#         print(data['X_train'][0, :5])
        
#         # Check y_train (should show balanced classes)
#         print(f"\nUnique labels and counts in 'y_train':")
#         labels, counts = np.unique(data['y_train'], return_counts=True)
#         print(dict(zip(labels, counts)))
        
#         # Check feature_names
#         if 'feature_names' in data.files:
#             print(f"\nTotal feature names: {len(data['feature_names'])}")
#             print(f"Sample of 'feature_names' (first 5):")
#             print(data['feature_names'][:5])
        
#     print("\n\nInspection complete.")
    
    
    
    
    
    
import numpy as np
import os

# ───────────────────────────────────────────────
# 1. DEFINE YOUR PATHS
# ───────────────────────────────────────────────

# This is the main output directory from your feature selection script
BASE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results"

# Define the names of the 4 track files saved by the script
track_files = {
    "Track 4A (Baseline)": "data_track_4A_Baseline.npz",
    "Track 4B (RFE)": "data_track_4B_RFE.npz",
    "Track 4C (Embedded RF)": "data_track_4C_Embedded_RF.npz",
    "Track 4D (Embedded L1)": "data_track_4D_Embedded_L1.npz"
}

print("--- Inspecting All Feature Selection Tracks ---")

# ───────────────────────────────────────────────
# 2. LOOP AND INSPECT EACH FILE
# ───────────────────────────────────────────────

for track_name, file_name in track_files.items():
    
    file_path = os.path.join(BASE_DIR, file_name)
    
    print(f"\n\n===============================================")
    print(f"🔎 INSPECTING: {track_name}")
    print(f"   File: {file_name}")
    print(f"===============================================")

    try:
        # Load the file
        with np.load(file_path, allow_pickle=True) as data:
            
            # --- 1. List all arrays inside ---
            print("Arrays found:")
            print(f"  {data.files}")
            
            # --- 2. Print shapes (the most important part) ---
            print("\nData Shapes:")
            
            # Get the shapes
            xtrain_shape = data['X_train'].shape
            xtest_shape = data['X_test'].shape
            ytrain_shape = data['y_train'].shape
            ytest_shape = data['y_test'].shape
            features_shape = data['feature_names'].shape
            
            print(f"  X_train: \t{xtrain_shape}")
            print(f"  X_test: \t{xtest_shape}")
            print(f"  y_train: \t{ytrain_shape}")
            print(f"  y_test: \t{ytest_shape}")
            print(f"  feature_names: {features_shape}")

            # --- 3. Confirm number of features ---
            print(f"\n👉 Result: This track has {xtrain_shape[1]} features.")
            
            # --- 4. Show sample of selected features ---
            print("\nSample of selected feature names:")
            print(f"  {data['feature_names'][:5]}")
            if len(data['feature_names']) > 5:
                print("   ...")
            

    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found at {file_path}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not read file. {e}")

print("\n\n--- Inspection Complete ---")