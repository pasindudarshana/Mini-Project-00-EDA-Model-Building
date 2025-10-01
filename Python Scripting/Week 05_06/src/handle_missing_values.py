import groq
import logging
import pandas as pd
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class ReplaceValuesStrategy(MissingValueHandlingStrategy):
    """
    Replaces blanks/whitespace in specified columns with NaN,
    converts columns to numeric (if applicable),
    and fills NaNs with 0.
    """
    def __init__(self, replace_columns=[]):
        self.replace_columns = replace_columns
        logging.info(f'Initializing ReplaceValuesStrategy for columns: {self.replace_columns}')

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.replace_columns:
            if col not in df.columns:
                logging.warning(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            # Replace blank or whitespace-only strings with NaN
            df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)

            # Try converting to numeric (non-numeric -> NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill NaNs with 0
            df[col] = df[col].fillna(0)

            logging.info(f"Column '{col}' cleaned. Nulls after cleaning: {df[col].isnull().sum()}")

        return df