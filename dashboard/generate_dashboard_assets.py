import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(os.getcwd())
CLEANED_DATA_FILE = PROJECT_ROOT / "data_loader_results" / "cleaned_full_wm811k.npz"
CSV_PATH = PROJECT_ROOT / "model_artifacts" / "master_model_comparison.csv"
ASSETS_DIR = PROJECT_ROOT / "dashboard" / "assets"

# Label mapping
MAPPING_TYPE = {
    0: "Center", 1: "Donut", 2: "Edge-Loc", 3: "Edge-Ring",
    4: "Loc", 5: "Random", 6: "Scratch", 7: "none"
}

def generate_assets():
    print("🚀 Generating dashboard assets...")
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # 1. Generate Defect Class Images
    if CLEANED_DATA_FILE.exists():
        print(f"🔹 Loading data from {CLEANED_DATA_FILE}")
        data = np.load(CLEANED_DATA_FILE)
        wafer_maps = data['waferMap']
        labels = data['labels']

        for label_idx, label_name in MAPPING_TYPE.items():
            # Find the first occurrence of this label
            indices = np.where(labels == label_idx)[0]
            if len(indices) > 0:
                idx = indices[0]
                wafer = wafer_maps[idx]
                
                # Plot and save
                plt.figure(figsize=(4, 4))
                plt.imshow(wafer, cmap='viridis')
                plt.axis('off')
                plt.title(label_name, color='white', fontsize=12)
                
                # Save with transparency and tight layout
                # Use a specific style for dashboard
                img_path = ASSETS_DIR / f"defect_{label_idx}.png"
                plt.savefig(img_path, bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
                print(f"   [OK] Saved image for {label_name} to {img_path}")
    else:
        print(f"⚠️ Cleaned data file not found at {CLEANED_DATA_FILE}")

    # 2. Extract Metrics to JSON
    if CSV_PATH.exists():
        print(f"🔹 Loading metrics from {CSV_PATH}")
        try:
            df = pd.read_csv(CSV_PATH)
            # Filter columns we want
            potential_cols = ['Track', 'Model', 'Test_F1_Macro', 'Test_Recall_Macro', 'Test_Precision_Macro', 'Overfit_Gap']
            available_cols = [c for c in potential_cols if c in df.columns]
            
            # Take top 10 models by Test_F1_Macro
            if 'Test_F1_Macro' in df.columns:
                df = df.sort_values(by='Test_F1_Macro', ascending=False)
            
            top_models = df[available_cols].head(10).to_dict(orient='records')
            
            with open(ASSETS_DIR / "metrics.json", "w") as f:
                json.dump(top_models, f, indent=4)
            print(f"   [OK] Saved metrics to {ASSETS_DIR / 'metrics.json'}")
        except Exception as e:
            print(f"❌ Error processing CSV: {e}")
    else:
        print(f"⚠️ Metrics CSV not found at {CSV_PATH}")

    print("✅ Asset generation complete.")

if __name__ == "__main__":
    generate_assets()
