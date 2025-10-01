import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) ->pd.DataFrame:
        pass

class CustomBinningStratergy(FeatureBinningStrategy):
    def __init__(self, bin_definitions):
        self.bin_definitions = bin_definitions 

    def bin_feature(self, df, column):
        def assign_bin(value):      
            for bin_label, bin_range in self.bin_definitions.items():
                if len(bin_range) == 2:
                    if bin_range[0] <= value <= bin_range[1]:
                        return bin_label
                elif len(bin_range) == 1:
                    if value >= bin_range[0]:
                        return bin_label  # Loyal has no upper limit 
            
            return "Invalid"
        
        df[f'{column}Bins'] = df[column].apply(assign_bin)
        del df[column]

        return df
    
    def bin_boolean(self, df, column):
        """
        Converts 'Yes'/'No' values in specified columns to 1/0.
        """
        for col in column:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        return df