# # ================================================================
# # Convert legacy LSWMD.pkl → LSWMD_1500.pkl (first 1500 rows)
# # Uses pd.read_pickle() for safer loading
# # ================================================================
#
# import pandas as pd
#
# # --- Input and Output paths ---
# input_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD.pkl"
# output_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\LSWMD_1500.pkl"
#
# # --- Load PKL safely using pandas ---
# print(f"[STEP 1] Loading dataset from: {input_path}")
#
# try:
#     df = pd.read_pickle(input_path)
#     print(f"[INFO] Dataset loaded successfully. Total rows: {len(df)}")
# except FileNotFoundError:
#     raise FileNotFoundError(f"[ERROR] File not found: {input_path}")
# except Exception as e:
#     raise RuntimeError(f"[ERROR] Failed to read pickle file: {e}")
#
# # --- Validate type ---
# if not isinstance(df, pd.DataFrame):
#     df = pd.DataFrame(df)
#     print("[WARN] Pickle content was not a DataFrame. Converted manually.")
#
# # --- Trim dataset to 1500 rows ---
# df_small = df.head(1500)
# print(f"[INFO] Trimmed dataset to {len(df_small)} rows")
#
# # --- Save to new PKL file ---
# try:
#     df_small.to_pickle(output_path)
#     print(f"[DONE] Saved successfully to: {output_path}")
# except Exception as e:
#     raise RuntimeError(f"[ERROR] Failed to save new pickle file: {e}")



# npz_to_csv.py
# ─────────────────────────────────────────────
# Convert cleaned wafer map dataset (.npz) to CSV files
# X_train, X_test, y_train, y_test → CSV
# Each wafer map will be flattened into a single row (for traditional ML)

import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
# 1️⃣ Load .npz file
# ─────────────────────────────────────────────

npz_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\cleaned_balanced_wm811k.npz"
save_dir = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\data_loader_results\csv_files"

os.makedirs(save_dir, exist_ok=True)

data = np.load(npz_path, allow_pickle=True)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# ─────────────────────────────────────────────
# 2️⃣ Flatten wafer maps for CSV
# ─────────────────────────────────────────────

def flatten_wafer_maps(X):
    """
    Convert each 2D wafer map into a 1D array.
    """
    return np.array([x.flatten() for x in X])

X_train_flat = flatten_wafer_maps(X_train)
X_test_flat = flatten_wafer_maps(X_test)

# ─────────────────────────────────────────────
# 3️⃣ Convert to DataFrame
# ─────────────────────────────────────────────

df_train = pd.DataFrame(X_train_flat)
df_train["label"] = y_train

df_test = pd.DataFrame(X_test_flat)
df_test["label"] = y_test

# ─────────────────────────────────────────────
# 4️⃣ Save to CSV
# ─────────────────────────────────────────────

train_csv_path = os.path.join(save_dir, "X_train.csv")
test_csv_path = os.path.join(save_dir, "X_test.csv")

df_train.to_csv(train_csv_path, index=False)
df_test.to_csv(test_csv_path, index=False)

print(f"✅ CSV files saved to: {save_dir}")
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
