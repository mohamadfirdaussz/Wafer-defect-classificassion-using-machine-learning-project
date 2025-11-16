"""
verify_cleaned_data.py
────────────────────────────────────────────
Verify structure and integrity of cleaned_data.npz,
convert normalized labels to integers (0–7),
and visualize one random wafer from each defect class.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

# Path to your cleaned data file
path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_data.npz"

print("🔹 Loading cleaned dataset...")
data = np.load(path, allow_pickle=True)

# Extract arrays
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Convert normalized float labels (0.0–1.0) back to integer classes (0–7)
y_train = np.round(y_train * 7).astype(int)
y_test = np.round(y_test * 7).astype(int)

print("✅ Labels converted to integer classes (0–7).")

# Display shapes
print("\n📏 Array Shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

# Show one example wafer shape
print("\n🧩 Example wafer map shape:", X_train[0].shape)

# Print label distribution
print("\n📊 Label Distribution (Train):")
print(dict(sorted(Counter(y_train).items())))

print("\n📊 Label Distribution (Test):")
print(dict(sorted(Counter(y_test).items())))

# Visualization: one random sample from each class
unique_classes = sorted(set(y_train))
print(f"\n🎨 Visualizing one random wafer per class ({len(unique_classes)} classes)...")

plt.figure(figsize=(12, 6))
for i, cls in enumerate(unique_classes):
    indices = np.where(y_train == cls)[0]
    if len(indices) == 0:
        continue
    idx = random.choice(indices)
    wafer = X_train[idx]
    
    plt.subplot(2, 4, i + 1)
    plt.imshow(wafer, cmap="Greens")
    plt.title(f"Class {cls}", fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\n✅ Verification complete — all classes visualized successfully.")