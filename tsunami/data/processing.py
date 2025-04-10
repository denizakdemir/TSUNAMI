import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Union, Optional, Tuple
import math


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Advanced data preprocessing module for TSUNAMI
    
    Handles:
    - Missing data imputation
    - Categorical encoding
    - Temporal feature processing
    - Feature normalization
    
    Parameters
    ----------
    num_impute_strategy : str, default='mean'
        Strategy for imputing missing numeric values.
        Options: 'mean', 'median', 'most_frequent', 'constant'
    
    cat_impute_strategy : str, default='most_frequent'
        Strategy for imputing missing categorical values.
        Options: 'most_frequent', 'constant'
        
    normalize : str, default='robust'
        Method for normalizing numeric features.
        Options: 'robust', 'standard', 'minmax', None
        
    time_features : List[str], default=None
        List of time encoding methods to apply.
        Options: 'sin', 'cos', 'hour', 'day', 'month', 'year', 'weekday'
        
    cat_embed_dim : Union[str, Dict[str, int]], default='auto'
        Dimensions for categorical embeddings.
        If 'auto', uses min(50, (cardinality+1)//2)
        Or provide a dictionary mapping feature names to dimensions
        
    handle_unknown : str, default='embed'
        Strategy for handling unknown categorical values.
        Options: 'error', 'embed'
        
    create_missing_indicators : bool, default=True
        Whether to create binary indicators for missing values
        
    mice_iterations : int, default=0
        Number of MICE iterations for advanced imputation.
        If 0, MICE is not used
    """
    
    def __init__(self, 
                 num_impute_strategy: str = 'mean',
                 cat_impute_strategy: str = 'most_frequent',
                 normalize: str = 'robust', 
                 time_features: Optional[List[str]] = None,
                 cat_embed_dim: Union[str, Dict[str, int]] = 'auto',
                 handle_unknown: str = 'embed',
                 create_missing_indicators: bool = True,
                 mice_iterations: int = 0):
        
        # Validate handle_unknown parameter
        if handle_unknown not in ['error', 'embed']:
            raise ValueError("handle_unknown must be either 'error' or 'embed'")
            
        # Initialize parameters
        self.num_impute_strategy = num_impute_strategy
        self.cat_impute_strategy = cat_impute_strategy
        self.normalize = normalize
        self.time_features = time_features or []
        self.cat_embed_dim = cat_embed_dim
        self.handle_unknown = handle_unknown
        self.create_missing_indicators = create_missing_indicators
        self.mice_iterations = mice_iterations
        
        # Initialize components (will be set during fit)
        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.ordinal_encoder = None
        self.cat_embed_info = {}
        self.time_encoders = {}
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the data processor to the input data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit the processor
            
        y : ignored
            Not used, present for API consistency
            
        Returns
        -------
        self : DataProcessor
            Fitted processor
        """
        # Separate numeric, categorical, and datetime features
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Store original missing patterns
        self.missing_patterns = {col: X[col].isna() for col in X.columns}
        
        # Initialize imputers
        self.num_imputer = SimpleImputer(strategy=self.num_impute_strategy)
        self.cat_imputer = SimpleImputer(strategy=self.cat_impute_strategy)
        
        # Fit imputers
        if self.num_cols:
            self.num_imputer.fit(X[self.num_cols])
        
        if self.cat_cols:
            # Encode categorical features first for imputation
            self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            
            # Fit the encoder on non-missing values only
            for col in self.cat_cols:
                non_missing_values = X[col].dropna().values.reshape(-1, 1)
                if len(non_missing_values) > 0:  # Only fit if there are non-missing values
                    self.ordinal_encoder.fit(non_missing_values)
            
            # Apply encoder to categorical features (only for imputation)
            encoded_cats = np.zeros((X.shape[0], len(self.cat_cols)))
            for i, col in enumerate(self.cat_cols):
                # Handle categorical data properly
                if X[col].dtype.name == 'category':
                    # For categorical data, we need to handle missing values differently
                    # Create a temporary series with missing values filled
                    temp_series = X[col].copy()
                    
                    # Make sure we can add a missing category
                    if '__MISSING__' not in temp_series.cat.categories:
                        temp_series = temp_series.cat.add_categories(['__MISSING__'])
                    
                    # Fill NA values with our special missing value
                    temp_series = temp_series.fillna('__MISSING__')
                    
                    # Transform the data
                    encoded_cats[:, i] = self.ordinal_encoder.transform(temp_series.values.reshape(-1, 1)).flatten()
                else:
                    # For non-categorical data, we can use the original approach
                    # Convert to strings first to avoid issues
                    temp_series = X[col].astype(str).fillna('__MISSING__')
                    encoded_cats[:, i] = self.ordinal_encoder.transform(temp_series.values.reshape(-1, 1)).flatten()
            
            # Fit imputer on encoded categories
            self.cat_imputer.fit(pd.DataFrame(encoded_cats, columns=self.cat_cols))
            
            # Compute embedding dimensions for categorical features
            for col in self.cat_cols:
                # Get unique values excluding NA
                cardinality = X[col].dropna().nunique()
                
                # Set embedding dimension
                if isinstance(self.cat_embed_dim, dict) and col in self.cat_embed_dim:
                    embed_dim = self.cat_embed_dim[col]
                elif self.cat_embed_dim == 'auto':
                    # Formula: min(50, (cardinality+1)//2)
                    embed_dim = min(50, (cardinality + 1) // 2)
                else:
                    embed_dim = 4  # Default
                
                # Store embedding info with original category labels
                unique_vals = X[col].dropna().unique()
                self.cat_embed_info[col] = {
                    'cardinality': cardinality + 1,  # +1 for unknown/missing
                    'embed_dim': embed_dim,
                    'mapping': {val: i for i, val in enumerate(unique_vals)},
                    'reverse_mapping': {i: val for i, val in enumerate(unique_vals)},
                    'original_name': col  # Store original variable name
                }
                
        # Initialize and fit scaler
        if self.normalize and self.num_cols:
            if self.normalize == 'robust':
                self.scaler = RobustScaler()
            elif self.normalize == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            elif self.normalize == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
                
            # Fit the scaler with feature names to avoid warnings
            num_data = X[self.num_cols].copy()
            self.scaler.fit(num_data)
            
        # Setup time feature encoders
        for col in self.date_cols:
            self.time_encoders[col] = self._create_time_encoder(X[col])
            
        # Determine output feature names
        self.feature_names_out_ = []
        
        # Numeric features (normalized)
        self.feature_names_out_.extend(self.num_cols)
        
        # Missing indicators
        if self.create_missing_indicators:
            self.missing_indicator_cols = [f"{col}_missing" for col in X.columns if X[col].isna().any()]
            self.feature_names_out_.extend(self.missing_indicator_cols)
        
        # Categorical embeddings
        for col, info in self.cat_embed_info.items():
            for i in range(info['embed_dim']):
                self.feature_names_out_.append(f"{col}_embed_{i}")
        
        # Time features
        for col in self.date_cols:
            for method in self.time_features:
                self.feature_names_out_.append(f"{col}_{method}")
                
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted processor.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform
            
        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        result = pd.DataFrame(index=X.index)
        
        # Create missing indicators first (before imputation)
        if self.create_missing_indicators and hasattr(self, 'missing_indicator_cols'):
            for col in X.columns:
                if f"{col}_missing" in self.missing_indicator_cols:
                    result[f"{col}_missing"] = X[col].isna().astype(float)
        
        # Process numeric features
        if self.num_cols:
            # Handle case where some columns in self.num_cols might not exist in X
            num_data = X.reindex(columns=self.num_cols).copy()
            
            # Impute missing numeric values
            num_imputed = self.num_imputer.transform(num_data)
            
            # Apply scaling if needed
            if self.scaler:
                # Create a DataFrame with column names to avoid the feature names warning
                num_df = pd.DataFrame(num_imputed, columns=self.num_cols, index=X.index)
                num_transformed = self.scaler.transform(num_df)
            else:
                num_transformed = num_imputed
                
            # Add to result dataframe
            for i, col in enumerate(self.num_cols):
                result[col] = num_transformed[:, i]
        
        # Process categorical features for embeddings
        for col in self.cat_cols:
            if col in X.columns:
                embed_info = self.cat_embed_info.get(col, {})
                
                if embed_info:
                    # Map categorical values to indices
                    indices = np.zeros(len(X), dtype=int)
                    
                    # Handle categorical data type properly
                    if X[col].dtype.name == 'category':
                        # Create a temporary series with missing values handled
                        temp_series = X[col].copy()
                        
                        # Make sure we can add the missing category if needed
                        if '__MISSING__' not in temp_series.cat.categories:
                            temp_series = temp_series.cat.add_categories(['__MISSING__'])
                        
                        # Fill NA values 
                        temp_series = temp_series.fillna('__MISSING__')
                        cat_vals = temp_series.values
                    else:
                        cat_vals = X[col].values
                    
                    for i, val in enumerate(cat_vals):
                        if pd.isna(val) or val == '__MISSING__':
                            # Use last index for missing values
                            indices[i] = embed_info['cardinality'] - 1
                        else:
                            # Map known values or use unknown index
                            indices[i] = embed_info['mapping'].get(val, embed_info['cardinality'] - 1)
                    
                    # Create simple one-hot encoding as placeholder for embeddings
                    # (actual embeddings will be learned by the model)
                    for j in range(embed_info['embed_dim']):
                        result[f"{col}_embed_{j}"] = 0.0
                        
                    # This is just placing initial values - the neural network
                    # will learn proper embeddings from these indices
                    for i, idx in enumerate(indices):
                        if idx < embed_info['cardinality'] - 1:  # Not missing
                            # Initialize with simple normalized index value
                            val = idx / embed_info['cardinality']
                            result.loc[X.index[i], f"{col}_embed_0"] = val
        
        # Process datetime features
        for col in self.date_cols:
            if col in X.columns:
                # Apply different time encodings
                for method in self.time_features:
                    result[f"{col}_{method}"] = self._encode_time_feature(X[col], method)
        
        # Ensure all columns from feature_names_out_ exist
        for col in self.feature_names_out_:
            if col not in result.columns:
                result[col] = 0.0  # Initialize missing columns
                
        # Return only the expected output columns, in the right order
        return result[self.feature_names_out_]
    
    def _create_time_encoder(self, datetime_series: pd.Series):
        """Create appropriate time feature encoding functions"""
        # Could be extended with more complex time encodings
        return {'min_time': datetime_series.min(), 'max_time': datetime_series.max()}
    
    def _encode_time_feature(self, datetime_series: pd.Series, method: str):
        """Encode datetime features using various methods"""
        if method == 'sin':
            # Calculate period and normalize
            min_time = self.time_encoders[datetime_series.name]['min_time']
            max_time = self.time_encoders[datetime_series.name]['max_time']
            total_seconds = (max_time - min_time).total_seconds()
            
            if total_seconds == 0:  # Avoid division by zero
                normalized = np.zeros(len(datetime_series))
            else:
                normalized = ((datetime_series - min_time).dt.total_seconds() / total_seconds) * 2 * np.pi
                
            return np.sin(normalized)
            
        elif method == 'cos':
            # Calculate period and normalize
            min_time = self.time_encoders[datetime_series.name]['min_time']
            max_time = self.time_encoders[datetime_series.name]['max_time']
            total_seconds = (max_time - min_time).total_seconds()
            
            if total_seconds == 0:  # Avoid division by zero
                normalized = np.zeros(len(datetime_series))
            else:
                normalized = ((datetime_series - min_time).dt.total_seconds() / total_seconds) * 2 * np.pi
                
            return np.cos(normalized)
            
        elif method == 'hour':
            return np.sin(datetime_series.dt.hour * (2 * np.pi / 24))
            
        elif method == 'day':
            return np.sin(datetime_series.dt.day * (2 * np.pi / 31))
            
        elif method == 'month':
            return np.sin((datetime_series.dt.month - 1) * (2 * np.pi / 12))
            
        elif method == 'year':
            # Normalize year to the range of years in the dataset
            min_year = datetime_series.dt.year.min()
            max_year = datetime_series.dt.year.max()
            
            if min_year == max_year:  # Avoid division by zero
                return np.zeros(len(datetime_series))
            else:
                normalized_year = (datetime_series.dt.year - min_year) / (max_year - min_year)
                return normalized_year * 2 - 1  # Scale to [-1, 1]
                
        elif method == 'weekday':
            return np.sin(datetime_series.dt.weekday * (2 * np.pi / 7))
            
        else:
            raise ValueError(f"Unknown time encoding method: {method}")
    
    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data
            
        y : ignored
            Not used, present for API consistency
            
        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def get_feature_names_out(self):
        """Return feature names of the output DataFrame"""
        return self.feature_names_out_


class TabularDataBatch:
    """
    Container for batched tabular data with different feature types.
    
    Attributes
    ----------
    continuous : torch.Tensor
        Tensor of continuous features [batch_size, num_continuous]
        
    categorical : Optional[torch.Tensor]
        Tensor of categorical indices [batch_size, num_categorical]
        
    categorical_info : List[Dict]
        Information about each categorical feature
        
    missing_mask : Optional[torch.Tensor]
        Binary mask of missing values [batch_size, num_features]
        
    time_features : Optional[torch.Tensor]
        Tensor of time features [batch_size, num_time_features]
    """
    
    def __init__(self, 
                 continuous: torch.Tensor,
                 categorical: Optional[torch.Tensor] = None,
                 categorical_info: Optional[List[Dict]] = None,
                 missing_mask: Optional[torch.Tensor] = None,
                 time_features: Optional[torch.Tensor] = None):
        
        self.continuous = continuous
        self.categorical = categorical
        self.categorical_info = categorical_info or []
        self.missing_mask = missing_mask
        self.time_features = time_features
        
    def to(self, device):
        """Move all tensors to the specified device"""
        self.continuous = self.continuous.to(device)
        
        if self.categorical is not None:
            self.categorical = self.categorical.to(device)
            
        if self.missing_mask is not None:
            self.missing_mask = self.missing_mask.to(device)
            
        if self.time_features is not None:
            self.time_features = self.time_features.to(device)
            
        return self
    
    @property
    def batch_size(self):
        """Get the batch size"""
        return self.continuous.size(0)


def build_category_info(processor: DataProcessor) -> List[Dict]:
    """
    Build categorical feature information from a fitted DataProcessor.
    
    Parameters
    ----------
    processor : DataProcessor
        Fitted data processor
        
    Returns
    -------
    List[Dict]
        List of dictionaries with information about each categorical feature
    """
    cat_info = []
    
    for col, info in processor.cat_embed_info.items():
        cat_info.append({
            'name': col,
            'cardinality': info['cardinality'],
            'embed_dim': info['embed_dim'],
            'original_name': info.get('original_name', col),
            'reverse_mapping': info.get('reverse_mapping', {})
        })
        
    return cat_info


def dataframe_to_torch(df: pd.DataFrame, 
                       processor: DataProcessor, 
                       device: str = 'cpu') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert preprocessed DataFrame to PyTorch tensors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe (already transformed by DataProcessor)
        
    processor : DataProcessor
        Fitted data processor
        
    device : str, default='cpu'
        PyTorch device to create tensors on
        
    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        Tuple containing:
        - Tensor of all features
        - Tensor of missing indicators (if available)
    """
    # Extract features and convert to torch tensors
    X_tensor = torch.tensor(df.values, dtype=torch.float32, device=device)
    
    # Extract missing indicators if they exist
    missing_indicators = None
    if processor.create_missing_indicators and hasattr(processor, 'missing_indicator_cols'):
        missing_cols = [col for col in df.columns if col.endswith('_missing')]
        if missing_cols:
            missing_indicators = torch.tensor(df[missing_cols].values, dtype=torch.float32, device=device)
    
    return X_tensor, missing_indicators