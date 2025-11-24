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



import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.load(
    r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results/cleaned_balanced_wm811k.npz",
    allow_pickle=True
)

wafer_maps = data["waferMap"]
labels = data["labels"]

# select 10 random indices
idx = np.random.choice(len(wafer_maps), size=10, replace=False)

# set up figure
plt.figure(figsize=(12, 10))

for i, wafer_idx in enumerate(idx):
    wm = wafer_maps[wafer_idx]

    plt.subplot(2, 5, i + 1)
    plt.imshow(wm, cmap="binary")  # change cmap if needed
    plt.title(f"Index {wafer_idx}\nLabel: {labels[wafer_idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()






