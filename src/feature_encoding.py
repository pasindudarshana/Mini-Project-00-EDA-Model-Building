import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass


class VariableType(str, Enum):
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'

class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns):
        self.nominal_columns = nominal_columns
        self.encoder_mappings = {}  # Store category mappings for inference
        os.makedirs('artifacts/encode', exist_ok=True)
        logger.info(f"NominalEncodingStrategy initialized for ONE-HOT encoding of columns: {nominal_columns}")

    def encode(self, df):
        logger.info(f"\n{'='*60}")
        logger.info(f"ONE-HOT ENCODING (NOMINAL)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting one-hot encoding for {len(self.nominal_columns)} columns")
        
        for column in self.nominal_columns:
            logger.info(f"\n--- Processing column: {column} ---")
            unique_values = sorted(df[column].dropna().unique())  # Sort for consistency
            logger.info(f"  Unique values ({len(unique_values)}): {unique_values}")
            
            # Store mapping for inference
            self.encoder_mappings[column] = unique_values
            
            # Save encoder mapping
            encoder_path = os.path.join('artifacts/encode', f"{column}_encoder.json")
            with open(encoder_path, "w") as f:
                json.dump({'categories': unique_values, 'encoding_type': 'one_hot'}, f)
            logger.info(f"  ✓ Saved one-hot encoder mapping to {encoder_path}")
            
            # Check for missing values before encoding
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                logger.warning(f"  ⚠ Column has {missing_count} missing values before encoding")
            
            # Create one-hot encoded columns
            for value in unique_values:
                new_col_name = f"{column}_{value}"
                df[new_col_name] = (df[column] == value).astype(int)
                logger.info(f"    ✓ Created binary column: {new_col_name}")
            
            # Drop the original column
            df = df.drop(columns=[column])
            logger.info(f"  ✓ Dropped original column '{column}'")
            logger.info(f"  ✓ One-hot encoding complete for '{column}' -> {len(unique_values)} binary columns")
            
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ ONE-HOT ENCODING COMPLETE")
        logger.info(f"{'='*60}\n")
        return df
    
    def get_encoder_mappings(self):
        return self.encoder_mappings
    
class OrdinalEncodingStratergy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings
        logger.info(f"OrdinalEncodingStrategy initialized for columns: {list(ordinal_mappings.keys())}")

    def encode(self,df):
        logger.info(f"\n{'='*60}")
        logger.info(f"ORDINAL ENCODING")
        logger.info(f"{'='*60}")
        logger.info(f"Starting ordinal encoding for {len(self.ordinal_mappings)} columns")
        
        for column, mapping in self.ordinal_mappings.items():
            logger.info(f"\n--- Processing column: {column} ---")
            initial_values = df[column].value_counts()
            logger.info(f"  Mapping: {mapping}")
            logger.info(f"  Before encoding: {initial_values.to_dict()}")
            
            # Check for missing values before encoding
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                logger.warning(f"  ⚠ Column has {missing_count} missing values before encoding")
            
            df[column] = df[column].map(mapping)
            
            # Check for unmapped values
            unmapped_count = df[column].isnull().sum()
            if unmapped_count > missing_count:
                logger.warning(f"  ⚠ Column has {unmapped_count - missing_count} unmapped values after encoding")
                
            # Log encoded value distribution
            encoded_values = df[column].value_counts()
            logger.info(f"  ✓ Encoded with {len(mapping)} categories")
            logger.info(f"  After encoding: {encoded_values.to_dict()}")
            
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ ORDINAL ENCODING COMPLETE")
        logger.info(f"{'='*60}\n")
        return df

class ServiceFeatureEncodingStrategy(FeatureEncodingStrategy):
    """
    Encodes PhoneServiceBins and InternetServiceBins, then drops the original columns.
    """
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        # PhoneServiceBins
        df['PhoneServiceBins'] = (
            df['PhoneService'].map({'No': 0, 'Yes': 1}) +
            df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 0})
        )

        # # InternetServiceBins
        # df['InternetServiceBins'] = (
        #     df['InternetService'].map({True: 0, False: 1}) +
        #     df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        #         'TechSupport', 'StreamingTV', 'StreamingMovies']]
        #     .apply(lambda col: col.map({'No': 0, 'Yes': 1, 'No internet service': 0}))
        #     .sum(axis=1)
        # )
        df['InternetServiceBins'] = (
            # InternetService_No is boolean, so map with True/False not strings
            # changed the code so that InterbetService will not be binned and encoded
            # since fiber optic customers are less likely to leave we can give that feature a higher score
            df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2}) +
            df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']]
            .apply(lambda col: col.map({'No internet service': 0, 'No': 1, 'Yes': 2, }))
            .sum(axis=1)
        )

        # Drop original columns
        df.drop(columns=[
            'PhoneService', 'MultipleLines',
            'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'InternetService'
        ], inplace=True)

        return df