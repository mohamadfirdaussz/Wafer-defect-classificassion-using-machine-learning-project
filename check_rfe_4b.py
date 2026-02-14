import numpy as np
import os

def check_rfe_features():
    path = 'feature_selection_results/data_track_4B_RFE.npz'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    data = np.load(path, allow_pickle=True)
    if 'feature_names' not in data:
        print(f"Key 'feature_names' not found in {path}")
        return

    feature_names = data['feature_names']
    print(f"Total Features: {len(feature_names)}")
    print("-" * 30)
    for i, name in enumerate(feature_names, 1):
        # Convert to string and clean up any whitespace/newlines
        clean_name = str(name).replace('\n', ' ').strip()
        # Collapse multiple spaces
        clean_name = ' '.join(clean_name.split())
        print(f"{i}. {clean_name}")

if __name__ == "__main__":
    check_rfe_features()
