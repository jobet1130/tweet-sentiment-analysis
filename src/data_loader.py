import pandas as pd
from pathlib import Path

def load_raw_data(path: str = '../data/raw/sentiment140.csv') -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f'Raw data file not found at {path}')
    df = pd.read_csv(file, encoding='latin-1', header=None)
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    return df[['target', 'text']]

def load_processed_data(path: str = 'data/processed/sentiment140_clean.csv') -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f'Processed data file not found at {path}')
    return pd.read_csv(file)
