import numpy as np
import os

# 1. SET THE PATH to the file you want to inspect
# -----------------------------------------------------------------
# (Example: 'model_ready_data.npz' or 'cleaned_balanced_wm811k.npz')
#
file_path = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\feature_selection_results\data_track_4D_Embedded_L1.npz"
# -----------------------------------------------------------------


# 2. Check if file exists
if not os.path.exists(file_path):
    print(f"❌ ERROR: File not found at {file_path}")
    print("Please check the path and try again.")
else:
    print(f"Inspecting file: {file_path}\n")

    # 3. Load the .npz file
    # We use 'with' so it automatically closes the file
    with np.load(file_path) as data:
        
        # --- 4. List all arrays stored in the file ---
        print("--- 1. Arrays in this file: ---")
        print(data.files)
        
        # --- 5. Inspect a specific array ---
        # Let's check the first array in the list
        if data.files:
            array_name = data.files[0]
            print(f"\n--- 2. Inspecting the first array ('{array_name}'): ---")
            
            # Load the array into memory
            my_array = data[array_name]
            
            # Print its shape
            print(f"Shape: {my_array.shape}")
            
            # Print its data type
            print(f"Data type: {my_array.dtype}")
            
            # Print a small sample of the data
            if len(my_array.shape) == 1:
                print(f"Sample data (first 5 values): {my_array[:5]}")
            elif len(my_array.shape) == 2:
                print(f"Sample data (first row, first 5 values): {my_array[0, :5]}")
            else:
                print("Array has more than 2 dimensions, showing first item.")
                print(my_array[0])