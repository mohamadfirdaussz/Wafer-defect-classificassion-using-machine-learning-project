# === FILENAME: data_loader.py ===

import pandas as pd
def load_wafer_data(file_path):
    """
    Loads wafer data from a specified pickle (.pkl) file.

    Args:
        file_path (str): The full path to the .pkl data file.

    Returns:
        pandas.DataFrame: The loaded DataFrame if successful, or None if an error occurs.
    """

    try:
        # Try to load the pickle file into a pandas DataFrame
        df = pd.read_pickle(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df

    except FileNotFoundError:
        # Handle case where the file path is invalid or file is missing
        print(f"Error: The file was not found at {file_path}")
        return None

    except Exception as e:
        # Catch any other exceptions during loading and print the error message
        print(f"An error occurred while loading the data: {e}")
        return None

# Optional test code block: This only runs if the file is executed directly (not when imported)
if __name__ == '__main__':
    # Example file path – replace with your actual file path to test the loader
    conceptual_file_path = "C:/Users/user/IdeaProjects/Projek-mcgogo/ml_flow/your_file_20rows.pkl"

    # Try loading the data and print status
    print(f"Attempting to load data from: {conceptual_file_path}")
    wafer_df = load_wafer_data(conceptual_file_path)

    # If loading was successful, display DataFrame info and preview
    if wafer_df is not None:
        print("\nDataFrame Information:")
        wafer_df.info()  # Show summary of DataFrame (columns, types, nulls, etc.)

        print("\nFirst 5 rows of the DataFrame:")
        print(wafer_df.head())  # Show the first few rows for inspection
    else:
        print("\nFailed to load data.")  # Loading failed, already reported in function
