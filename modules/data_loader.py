# modules/data_loader.py
import pandas as pd
import json
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _parse_json_string(json_string: str) -> Optional[dict]:
    """Safely parses a JSON string into a dictionary, returning None on failure."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None

def load_and_prepare_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads data from a CSV file, parses JSON columns, and cleans invalid rows.
    """
    logging.info(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        df['compliance_data'] = df['compliance'].apply(_parse_json_string)
        df['llm_output_data'] = df['LLM_output'].apply(_parse_json_string)
        
        initial_rows = len(df)
        df.dropna(subset=['compliance_data', 'llm_output_data'], inplace=True)
        cleaned_rows = len(df)
        
        if initial_rows > cleaned_rows:
            logging.warning(f"Dropped {initial_rows - cleaned_rows} rows due to JSON parsing errors.")
            
        logging.info(f"Successfully loaded and prepared {cleaned_rows} rows.")
        return df
        
    except FileNotFoundError:
        logging.error(f"Data file not found at the specified path: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return None
