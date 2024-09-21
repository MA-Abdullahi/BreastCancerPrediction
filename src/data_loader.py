import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
    - file_path: str, the path to the data file.
    
    Returns:
    - DataFrame: the loaded pandas dataframe.
    """
    df = pd.read_csv(file_path)
    return df
