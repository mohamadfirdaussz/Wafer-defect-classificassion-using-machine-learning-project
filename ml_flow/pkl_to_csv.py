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

import pandas as pd
import os

# === Configuration ===
input_path = "C:/Users/user/OneDrive - ums.edu.my/FYP 1/LSWMD_1500.pkl"   # 🔹 Change to your actual .pkl file path
output_path = "C:/Users/user/OneDrive - ums.edu.my/FYP 1/LSWMD_1500.csv"

# === Step 1: Load PKL file safely ===
print("[STEP 1] Loading dataset using pandas...")
try:
    data = pd.read_pickle(input_path)
    print(f"[INFO] Dataset loaded successfully. Shape: {data.shape}")
except Exception as e:
    print(f"[ERROR] Failed to load PKL file: {e}")
    exit()

# === Step 2: Convert to DataFrame (if not already) ===
if not isinstance(data, pd.DataFrame):
    print("[INFO] Data is not a DataFrame. Attempting conversion...")
    try:
        data = pd.DataFrame(data)
        print("[INFO] Conversion to DataFrame successful.")
    except Exception as e:
        print(f"[ERROR] Could not convert data to DataFrame: {e}")
        exit()

# === Step 3: Take only first 1500 rows ===
subset = data.head(1500)
print(f"[STEP 2] Selected first 1500 rows. Shape: {subset.shape}")

# === Step 4: Save as CSV ===
try:
    subset.to_csv(output_path, index=False)
    print(f"[STEP 3] Successfully saved CSV: {os.path.abspath(output_path)}")
except Exception as e:
    print(f"[ERROR] Failed to save CSV: {e}")

