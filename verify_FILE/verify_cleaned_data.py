import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Path to the file created by data_loader.py
FILE_PATH = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_full_wm811k.npz"

def verify_data(file_path):
    print(f"🔍 Inspecting file: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ Error: File not found! Run data_loader.py first.")
        return

    # 1. Load Data
    data = np.load(file_path)
    X = data['waferMap']
    y = data['labels']
    
    print("\n--- 1. SHAPE CHECK ---")
    print(f"✅ Wafer Map Shape: {X.shape}  (Expected: ~172k, 64, 64)")
    print(f"✅ Labels Shape:    {y.shape}  (Expected: ~172k)")
    
    # 2. Check Data Integrity
    print("\n--- 2. VALUE CHECK ---")
    unique_vals = np.unique(X)
    print(f"Pixel values found: {unique_vals}")
    if np.array_equal(unique_vals, [0, 1, 2]):
        print("✅ PASS: Only discrete values (0, 1, 2) found.")
    else:
        print("⚠️ WARNING: Found unexpected pixel values (did resizing create floats?)")

    # 3. Check Class Distribution
    print("\n--- 3. CLASS DISTRIBUTION (Should be IMBALANCED) ---")
    mapping_rev = {
        0: "Center", 1: "Donut", 2: "Edge-Loc", 3: "Edge-Ring",
        4: "Loc", 5: "Random", 6: "Scratch", 7: "none"
    }
    
    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    print(f"{'Class ID':<10} {'Name':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    
    for label, count in zip(unique, counts):
        name = mapping_rev.get(label, "Unknown")
        pct = (count / total_samples) * 100
        print(f"{label:<10} {name:<15} {count:<10} {pct:.2f}%")





         # 4. VISUALIZATION (Using Binary or Greens colormap)
    print("\n--- 4. VISUALIZATION CHECK ---")

    rng = np.random.default_rng(42)

    available_classes = [c for c in sorted(mapping_rev.keys()) if np.any(y == c)]
    num_classes = len(available_classes)

    print(f"Displaying {num_classes} available classes...")

    cols = 4
    rows = int(np.ceil(num_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    # Consistent scaling for discrete values
    vmin, vmax = 0, 2

    # --- CHANGE COLORMAP HERE ---
    # cmap = "binary"     # black–white
    cmap = "Greens"       # green intensity scale
    # ----------------------------

    for ax in axes:
        ax.axis("off")

    for i, cls in enumerate(available_classes):
        ax = axes[i]
        indices = np.where(y == cls)[0]

        if len(indices) == 0:
            continue

        idx = rng.choice(indices)
        img = X[idx]

        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{mapping_rev[cls]} (ID: {cls})", fontsize=10)
        ax.axis("off")

    plt.suptitle("Wafer Map Samples by Class", fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    verify_data(FILE_PATH)