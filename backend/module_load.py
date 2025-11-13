"""
module_load.py
=================
This module provides a DataLoader class for loading datasets and processing variable types.
The class handles dataset loading, determining variable types, splitting data into feature,
target, and protected attributes, and encoding categorical variables for machine learning tasks.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional

# Import configuration
from core_config import VERBOSE, DATASET, PARAMS_NUM_TO_CAT_METHOD, PARAMS_NUM_TO_CAT_CUTS


class DataLoader:
    """
        Data loader class for loading datasets and processing variable types
    """
    
    def __init__(self):
        """Initialize DataLoader class"""
        self.df = None  # Complete dataset
        self.X = None  # Feature data (excluding Y and O)
        self.Y = None  # Target variable
        self.O = None  # Protected attributes
        self.categorical_columns = []  # Categorical variables
        self.numerical_columns = []  # Continuous variables
        self.label_encoders = {}  # Store encoders for potential inverse transformation
        

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str], List[str]]:
        """
        Load dataset and process according to requirements
        
        Returns:
            tuple: (X feature data, Y target variable, O protected attributes, 
                   list of categorical variables, list of continuous variables)
        """
        # Read dataset
        self._read_dataset()
        
        # Determine variable types
        self._determine_variable_types()
        
        # Split data into Y, O, and X
        self._split_data()
        
        # Process numerical O variables
        self._process_numerical_o()
        
        # Encode categorical variables to integers starting from 0
        self._encode_categorical_variables()
        
        # Output detailed information (if VERBOSE is enabled)
        if VERBOSE:
            self._print_data_info()
        
        return self.X, self.Y, self.O, self.categorical_columns, self.numerical_columns
    

    def _read_dataset(self) -> None:
        """Read dataset file, choose between name and path"""
        from datasets_info import DATASET_INFO
        
        data_path = DATASET.get('path', '')
        data_name = DATASET.get('name', '')
        
        # Choose between name and path, prefer name if both provided
        if data_name:
            # Use name to load dataset from DATASET_INFO
            if data_name in DATASET_INFO:
                dataset_info = DATASET_INFO[data_name]
                data_file = dataset_info.get('data', '')
                
                # Construct full path
                full_path = data_file
                if not os.path.exists(full_path):
                    # Try data directory
                    full_path = os.path.join('data', data_file)
                    if not os.path.exists(full_path):
                        raise ValueError(f"Dataset file not found: {full_path}")
                
                # Load data based on file extension
                if full_path.endswith('.csv'):
                    self.df = pd.read_csv(full_path)
                elif full_path.endswith('.xlsx') or full_path.endswith('.xls'):
                    self.df = pd.read_excel(full_path)
                elif full_path.endswith('.json'):
                    self.df = pd.read_json(full_path)
                elif full_path.endswith('.parquet'):
                    self.df = pd.read_parquet(full_path)
                else:
                    raise ValueError(f"Unsupported file format: {full_path}")
                
                if VERBOSE:
                    print(f"Loaded dataset '{data_name}' from DATASET_INFO")
            else:
                raise ValueError(f"Dataset '{data_name}' not found in DATASET_INFO")
        elif data_path:
            # Use path to load dataset
            if os.path.exists(data_path):
                # Load data based on file extension
                if data_path.endswith('.csv'):
                    self.df = pd.read_csv(data_path)
                elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                    self.df = pd.read_excel(data_path)
                elif data_path.endswith('.json'):
                    self.df = pd.read_json(data_path)
                elif data_path.endswith('.parquet'):
                    self.df = pd.read_parquet(data_path)
                else:
                    raise ValueError(f"Unsupported file format: {data_path}")
                
                if VERBOSE:
                    print(f"Loaded dataset from path: {data_path}")
            else:
                raise ValueError(f"Dataset file not found: {data_path}")
        else:
            raise ValueError("Neither 'name' nor 'path' provided in DATASET configuration")
    

    def _determine_variable_types(self) -> None:
        """Determine variable types (categorical or continuous)"""
        # Get predefined variable types from DATASET
        self.categorical_columns = DATASET.get('categorical', [])
        self.numerical_columns = DATASET.get('numerical', [])
        
        # If not enough type information provided, auto-detect
        if not self.categorical_columns or not self.numerical_columns:
            auto_categorical, auto_numerical = self._auto_identify_column_types()
            
            if not self.categorical_columns:
                self.categorical_columns = auto_categorical
            if not self.numerical_columns:
                self.numerical_columns = auto_numerical
        
        # Validate column names
        all_columns = set(self.df.columns)
        self.categorical_columns = [col for col in self.categorical_columns if col in all_columns]
        self.numerical_columns = [col for col in self.numerical_columns if col in all_columns]
    

    def _auto_identify_column_types(self) -> Tuple[List[str], List[str]]:
        """
        Auto-identify column types
        
        Returns:
            tuple: (list of categorical variables, list of continuous variables)
        """
        categorical = []
        numerical = []
        
        for col in self.df.columns:
            # Skip columns with all NaN values
            if self.df[col].isna().all():
                continue
            
            # Check data type
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                numerical.append(col)  # Datetime treated as continuous
            elif pd.api.types.is_bool_dtype(self.df[col]):
                categorical.append(col)  # Boolean treated as categorical
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                # Numerical - check unique value ratio, might be categorical
                unique_num = self.df[col].nunique()
                if unique_num < 20:  # Treat as categorical if unique values < 20
                    categorical.append(col)
                else:
                    numerical.append(col)
            else:
                # Object type - check if can be converted to numerical
                try:
                    _ = pd.to_numeric(self.df[col])
                    # Can be converted to numerical - check unique value ratio
                    unique_ratio = self.df[col].nunique() / len(self.df[col])
                    if unique_ratio < 0.1:
                        categorical.append(col)
                    else:
                        numerical.append(col)
                except (ValueError, TypeError):
                    # Cannot be converted to numerical - treat as categorical
                    categorical.append(col)
        
        return categorical, numerical
    

    def _split_data(self) -> None:
        """Split data into Y, O, and X"""
        target_col = DATASET.get('target')
        if not target_col or target_col not in self.df.columns:
            raise ValueError("Target variable 'target' is not defined in DATASET or not in the dataset")
        
        # Get Y
        self.Y = self.df[target_col].copy()
        
        # Get O (protected attributes)
        protected_cols = DATASET.get('protected', [])
        if not protected_cols:
            raise ValueError("Protected attribute 'protected' is not defined in DATASET")
        
        # Handle case where O can be single or multiple attributes
        if isinstance(protected_cols, str):
            protected_cols = [protected_cols]
        
        # Validate O columns exist
        valid_protected_cols = [col for col in protected_cols if col in self.df.columns]
        if not valid_protected_cols:
            raise ValueError(f"Protected attributes {protected_cols} do not exist in the dataset")
        
        # Get O
        self.O = self.df[valid_protected_cols].copy()
        
        # Get X (all columns except Y and O)
        exclude_cols = [target_col] + valid_protected_cols
        self.X = self.df.drop(columns=exclude_cols).copy()
        
        # Update categorical and numerical variable lists (remove Y and O)
        self.categorical_columns = [col for col in self.categorical_columns if col not in exclude_cols]
        self.numerical_columns = [col for col in self.numerical_columns if col not in exclude_cols]
    

    def _process_numerical_o(self) -> None:
        """Process numerical protected attributes O, converting them to categorical variables"""
        # If O has multiple columns, process each column
        for col in self.O.columns:
            # Check if column is numerical
            if pd.api.types.is_numeric_dtype(self.O[col]) and col not in self.categorical_columns:
                # Convert according to configured method
                if PARAMS_NUM_TO_CAT_METHOD == 'median':
                    # Binarize using median
                    median_val = self.O[col].median()
                    self.O[col] = (self.O[col] > median_val).astype(str)
                elif PARAMS_NUM_TO_CAT_METHOD == 'quartile':
                    # Quartile binning into 4 categories
                    try:
                        self.O[col] = pd.qcut(self.O[col], q=4, labels=False)
                        self.O[col] = self.O[col].astype(str)
                    except ValueError:
                        # If quartile binning fails, use custom binning
                        self.O[col] = pd.cut(self.O[col], bins=PARAMS_NUM_TO_CAT_CUTS, labels=False)
                        self.O[col] = self.O[col].astype(str)
                else:
                    # Custom binning
                    self.O[col] = pd.cut(self.O[col], bins=PARAMS_NUM_TO_CAT_CUTS, labels=False)
                    self.O[col] = self.O[col].astype(str)
    

    def _encode_categorical_variables(self) -> None:
        """Encode categorical variables to integers starting from 0 and process target variable Y"""
        # Encode categorical variables in X
        for col in self.categorical_columns:
            if col in self.X.columns:
                encoder = LabelEncoder()
                # Handle NaN values by converting them to a string representation
                X_col = self.X[col].fillna('NaN')
                self.X[col] = encoder.fit_transform(X_col)
                self.label_encoders[col] = encoder
        
        # Encode categorical variables in O
        for col in self.O.columns:
            if not pd.api.types.is_numeric_dtype(self.O[col]):
                encoder = LabelEncoder()
                # Handle NaN values by converting them to a string representation
                O_col = self.O[col].fillna('NaN')
                self.O[col] = encoder.fit_transform(O_col)
                self.label_encoders[col] = encoder
        
        # Process target variable Y according to requirements
        # Handle NaN values first
        self.Y = self.Y.fillna('NaN') if self.Y.dtype == 'object' else self.Y.fillna(self.Y.median())
        
        # Get unique values count
        unique_values = self.Y.unique()
        num_unique = len(unique_values)
        
        # Check if it's a binary classification problem
        if num_unique == 2:
            # Binary classification - convert to 0/1 series while keeping Series type
            if pd.api.types.is_object_dtype(self.Y):
                # Encode object type binary target
                encoder = LabelEncoder()
                self.Y = pd.Series(encoder.fit_transform(self.Y), index=self.Y.index)
                self.label_encoders['target'] = encoder
            else:
                # For numerical binary target, ensure it's 0/1 encoded as Series
                min_val, max_val = self.Y.min(), self.Y.max()
                if min_val != 0 or max_val != 1:
                    encoder = LabelEncoder()
                    self.Y = pd.Series(encoder.fit_transform(self.Y), index=self.Y.index)
                    self.label_encoders['target'] = encoder
            if VERBOSE:
                print("Processing binary classification target Y as Series.")
        
        # Check if it's a continuous variable
        elif pd.api.types.is_numeric_dtype(self.Y) and num_unique > 2:
            # Continuous variable - binarize using median
            median_val = self.Y.median()
            self.Y = pd.Series((self.Y > median_val).astype(int), index=self.Y.index)
            if VERBOSE:
                print(f"Converting continuous target Y to binary using median {median_val}.")
        
        # Check if it's a multi-class classification problem
        elif num_unique > 2:
            # Multi-class classification - raise error
            raise ValueError(f"Target variable Y is multi-class with {num_unique} classes. This implementation only supports binary classification.")
        
        # This should not happen with proper data
        else:
            if VERBOSE:
                print("Warning: Target variable Y has unusual characteristics.")
        

    def _print_data_info(self) -> None:
        """Print detailed information about the dataset"""
        print("\nDataset loaded:")
        print(f"Original data shape: {self.df.shape}")
        print(f"Features X shape: {self.X.shape}")
        print(f"Target Y shape: {self.Y.shape}")
        print(f"Protected attributes O shape: {self.O.shape}")
        print(f"Categorical variables: {self.categorical_columns}")
        print(f"Numerical variables: {self.numerical_columns}")
        print(f"First 5 samples of Y:\n{self.Y.head()}")
        print(f"First 5 samples of O:\n{self.O.head()}")
        print(f"First 5 samples of X:\n{self.X.head()}")
    

    def get_encoding_map(self, column: str) -> Optional[Dict[int, str]]:
        """
        Get encoding map for a categorical column
        
        Parameters:
            column: Name of the categorical column
        
        Returns:
            Dictionary mapping encoded values to original labels, or None if column is not encoded
        """
        if column not in self.label_encoders:
            return None
        
        encoder = self.label_encoders[column]
        return {i: label for i, label in enumerate(encoder.classes_)}
    

    def get_column_types(self) -> Dict[str, List[str]]:
        """Get variable type information for the current dataset"""
        return {
            "categorical": self.categorical_columns,
            "numerical": self.numerical_columns
        }
        

    def inverse_encode(self, column: str, encoded_values) -> np.ndarray:
        """
        Convert encoded values back to original labels
        
        Parameters:
            column: Name of the column to inverse encode
            encoded_values: Encoded values to convert back
        
        Returns:
            Original labels corresponding to the encoded values
        """
        if column not in self.label_encoders:
            raise ValueError(f"Column '{column}' is not encoded or does not exist")
        
        encoder = self.label_encoders[column]
        return encoder.inverse_transform(encoded_values)

