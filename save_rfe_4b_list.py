import numpy as np
import os

def save_rfe_features():
    path = 'feature_selection_results/data_track_4B_RFE.npz'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    data = np.load(path, allow_pickle=True)
    if 'feature_names' not in data:
        print(f"Key 'feature_names' not found in {path}")
        return

    feature_names = data['feature_names']
    with open('rfe_features_list.txt', 'w', encoding='utf-8') as f:
        for i, name in enumerate(feature_names, 1):
            # Replace any control characters with a space
            clean_name = ''.join(c if ord(c) >= 32 else ' ' for c in str(name))
            clean_name = ' '.join(clean_name.split())
            f.write(f"{i}. {clean_name}\n")
    print(f"Saved {len(feature_names)} features to rfe_features_list.txt")

if __name__ == "__main__":
    save_rfe_features()
