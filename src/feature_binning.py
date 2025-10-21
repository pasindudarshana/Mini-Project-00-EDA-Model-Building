import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) ->pd.DataFrame:
        pass

class CustomBinningStratergy(FeatureBinningStrategy):
    def __init__(self, bin_definitions):
        self.bin_definitions = bin_definitions
        logger.info(f"CustomBinningStrategy initialized with bins: {list(bin_definitions.keys())}") 

    def bin_feature(self, df, column):
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE BINNING - {column.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Starting binning for column: {column}")
        
        # Handle potential non-numeric or NA values gracefully before logging range
        numeric_col = pd.to_numeric(df[column], errors='coerce')
        if numeric_col.isnull().any():
            logger.warning(f"  Non-numeric or NA values found in '{column}'. They will be binned as 'Invalid'.")
        
        initial_unique = df[column].nunique()
        value_range = (numeric_col.min(), numeric_col.max())
        
        logger.info(f"  Unique values: {initial_unique}, Numeric Range: [{value_range[0]:.2f}, {value_range[1]:.2f}]")
        
        # --- MODIFIED FUNCTION ---
        # This function is now generic and driven entirely by self.bin_definitions
        def assign_bin(value):
            # Handle non-numeric types that can't be compared
            try:
                # Attempt to convert to float for comparison
                val = float(value)
            except (ValueError, TypeError):
                return "Invalid"
            
            # Check for NaN values explicitly
            if pd.isna(val):
                return "Invalid"

            for bin_label, bin_range in self.bin_definitions.items():
                if len(bin_range) == 2:
                    # Case: [min, max] (e.g., "New Customers": [0, 12])
                    # This is inclusive on both ends
                    if bin_range[0] <= val <= bin_range[1]:
                        return bin_label
                elif len(bin_range) == 1:
                    # Case: [min] (e.g., "Loyal": [48])
                    # This means >= min
                    if val >= bin_range[0]:
                        return bin_label
            
            # If no bin was matched after checking all definitions
            return "Invalid"
        # --- END OF MODIFICATION ---
        
        df[f'{column}Bins'] = df[column].apply(assign_bin)
        
        # Log binning results
        bin_counts = df[f'{column}Bins'].value_counts().sort_index()
        logger.info(f"\nBinning Results:")
        total_count = len(df)
        for bin_name, count in bin_counts.items():
            logger.info(f"  ✓ {bin_name}: {count} ({count/total_count*100:.2f}%)")
        
        invalid_count = (df[f'{column}Bins'] == "Invalid").sum()
        if invalid_count > 0:
            logger.warning(f"  ⚠ Found {invalid_count} 'Invalid' values in column '{column}'")
            
        del df[column]
        logger.info(f"✓ Original column '{column}' removed, replaced with '{column}Bins'")
        logger.info(f"{'='*60}\n")

        return df

# class CustomBinningStratergy(FeatureBinningStrategy):
#     def __init__(self, bin_definitions):
#         self.bin_definitions = bin_definitions 
#         logger.info(f"CustomBinningStrategy initialized with bins: {list(bin_definitions.keys())}") 

#     def bin_feature(self, df, column):
#         def assign_bin(value):      
#             for bin_label, bin_range in self.bin_definitions.items():
#                 if len(bin_range) == 2:
#                     if bin_range[0] <= value <= bin_range[1]:
#                         return bin_label
#                 elif len(bin_range) == 1:
#                     if value >= bin_range[0]:
#                         return bin_label  # Loyal has no upper limit 
            
#             return "Invalid"
        
#         df[f'{column}Bins'] = df[column].apply(assign_bin)
#         del df[column]

#         return df
    
    # def bin_boolean(self, df, column):
    #     """
    #     Converts 'Yes'/'No' values in specified columns to 1/0.
    #     """
    #     for col in column:
    #         df[col] = df[col].map({'Yes': 1, 'No': 0})
    #     return df