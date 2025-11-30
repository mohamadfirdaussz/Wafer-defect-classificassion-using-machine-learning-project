# import numpy as np

# file_path =  r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results/cleaned_balanced_wm811k.npz " # tukar ikut nama fail kamu

# data = np.load(file_path, allow_pickle=True)

# print("Keys in file:", list(data.keys()))

# # Print shapes and dtypes
# for key in data:
#     print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")






# -*- coding: utf-8 -*-
# import numpy as np

# import matplotlib.pyplot as plt

# data = np.load(r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results/cleaned_balanced_wm811k.npz", allow_pickle=True)

# print("Keys:", data.files)

# for key in data.files:
#     print(key, "→ shape:", data[key].shape, ", dtype:", data[key].dtype)

# wm = data["waferMap"][0]
# label = data["labels"][0]

# print("First wafer map:", type(wm), "shape:", len(wm), "x", len(wm[0]))
# print("Label:", label)


# plt.imshow(data["waferMap"][0], cmap="binary")
# plt.title(f"Label = {data['labels'][5]}")
# plt.show()


# for key in data.files:
#     arr = data[key]
#     print(f"--- {key} ---")
#     print("Type:", type(arr))
#     print("Shape:", arr.shape)
#     print("Dtype:", arr.dtype)
#     print("Sample:", arr[0])
#     print()



# import numpy as np
# import matplotlib.pyplot as plt

# # load data
# data = np.load(
#     r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results/cleaned_balanced_wm811k.npz",
#     allow_pickle=True
# )

# wafer_maps = data["waferMap"]
# labels = data["labels"]

# # select 10 random indices
# idx = np.random.choice(len(wafer_maps), size=10, replace=False)

# # set up figure
# plt.figure(figsize=(12, 10))

# for i, wafer_idx in enumerate(idx):
#     wm = wafer_maps[wafer_idx]

#     plt.subplot(2, 5, i + 1)
#     plt.imshow(wm, cmap="binary")  # change cmap if needed
#     plt.title(f"Index {wafer_idx}\nLabel: {labels[wafer_idx]}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()





import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# UPDATE THIS PATH to your Stage 1 output
FILE_PATH = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_balanced_wm811k.npz"
# ---------------------------------------------------------

def verify_loader_output(file_path):
    print(f"🕵️‍♂️ INSPECTING: {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    # 1. Load Data
    data = np.load(file_path)
    print(f"🔑 Keys found: {list(data.keys())}")
    
    # We expect 'waferMap' and 'labels' based on your previous code
    if 'waferMap' not in data or 'labels' not in data:
        print("❌ CRITICAL: Missing expected keys ('waferMap', 'labels').")
        return

    maps = data['waferMap']
    labels = data['labels']

    # 2. Check Shapes
    print("\n[DIMENSION CHECK]")
    print(f"  Wafer Maps Shape: {maps.shape}")
    print(f"  Labels Shape:     {labels.shape}")
    
    if len(maps.shape) != 3:
        print("  ❌ ERROR: Wafer Maps should be 3D (N, Height, Width).")
    elif maps.shape[1:] != (64, 64):
        print(f"  ⚠️ WARNING: Maps are {maps.shape[1:]}, expected (64, 64). Did you change target_size?")
    else:
        print("  ✅ Dimensions look correct (N, 64, 64).")

    # 3. Check Data Values (Crucial for Resizing)
    print("\n[PIXEL VALUE CHECK]")
    unique_vals = np.unique(maps)
    print(f"  Unique pixel values found: {unique_vals}")
    
    # We expect discrete integers for wafer maps: 0, 1, 2
    if len(unique_vals) <= 3 and all(val in [0, 1, 2] for val in unique_vals):
        print("  ✅ PASS: Image data is discrete (0, 1, 2). Nearest-neighbor resizing worked.")
    else:
        print("  ⚠️ WARNING: Found unexpected values! (e.g. 0.5, 1.2).")
        print("     This means interpolation happened (blurring). Models prefer discrete 0/1/2.")

    # 4. Visual Sanity Check
    print("\n[VISUAL CHECK]")
    print("  Generating a plot of random samples per class...")
    
    plot_samples(maps, labels)

def plot_samples(maps, labels):
    """
    Plots one random example for every defect class found.
    """
    # Map integers back to names for the plot title
    label_map = {
        0: "Center", 1: "Donut", 2: "Edge-Loc", 3: "Edge-Ring",
        4: "Loc", 5: "Random", 6: "Scratch", 7: "none"
    }
    
    unique_labels = np.unique(labels)
    plt.figure(figsize=(15, 6))
    
    for i, lbl in enumerate(unique_labels):
        # Find indices where this label exists
        indices = np.where(labels == lbl)[0]
        if len(indices) == 0: continue
        
        # Pick a random one
        idx = np.random.choice(indices)
        img = maps[idx]
        
        # Plot
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap='inferno') # 'inferno' highlights defects (2) brightly
        title = label_map.get(lbl, f"Label {lbl}")
        plt.title(f"{title}\n(Index: {idx})")
        plt.axis('off')

    plt.suptitle("Sanity Check: One Random Sample per Class", fontsize=16)
    plt.tight_layout()
    plt.show()
    print("  ✅ Plot generated. Check the popup window.")

if __name__ == "__main__":
    verify_loader_output(FILE_PATH)




