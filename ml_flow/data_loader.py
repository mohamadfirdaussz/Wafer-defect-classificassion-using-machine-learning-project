# === FILENAME: data_loader.py ===
import pandas as pd

def load_wafer_data(file_path):
    """
    Loads wafer data from a specified pickle file.

    Args:
        file_path (str): The full path to the .pkl data file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_pickle(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

if __name__ == '__main__':
    # Example usage (optional, for testing the loader directly)
    # Replace with an actual path if you run this file directly
    # conceptual_file_path = "path/to/your/LSWMD.pkl"
    conceptual_file_path = "C:/Users/user/IdeaProjects/Projek-mcgogo/ml_flow/your_file_20rows.pkl" # Using the path from the original script

    print(f"Attempting to load data from: {conceptual_file_path}")
    wafer_df = load_wafer_data(conceptual_file_path)

    if wafer_df is not None:
        print("\nDataFrame Information:")
        wafer_df.info()
        print("\nFirst 5 rows of the DataFrame:")
        print(wafer_df.head())
    else:
        print("\nFailed to load data.")