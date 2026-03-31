"""
Data loader module for PharmaDrishti.
Handles loading persona data from JSON files.
"""

import json
import pandas as pd
from pathlib import Path


def load_personas(filepath='data/indian_healthcare_personas.json'):
    """
    Load personas from JSON file and return as pandas DataFrame.
    
    Args:
        filepath (str): Path to the JSON file containing persona data.
                       Defaults to 'data/indian_healthcare_personas.json'
    
    Returns:
        pd.DataFrame: DataFrame containing persona data
    
    Raises:
        FileNotFoundError: If the specified file does not exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        # Convert to Path object for better path handling
        file_path = Path(filepath)

        if not file_path.exists():
            fallback_candidates = [
                Path(__file__).parent / 'indian_healthcare_personas.json',
                Path(__file__).parent / 'data' / 'indian_healthcare_personas.json',
            ]
            for candidate in fallback_candidates:
                if candidate.exists():
                    file_path = candidate
                    break
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Persona data file not found: {filepath}\n"
                f"Please ensure the file exists at the specified location."
            )
        
        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # Return empty DataFrame
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filepath}")
        print(f"Details: {e}")
        return pd.DataFrame()  # Return empty DataFrame
    
    except Exception as e:
        print(f"Unexpected error loading personas: {e}")
        return pd.DataFrame()  # Return empty DataFrame
