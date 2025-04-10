import pytest
import numpy as np
import pandas as pd
from tsunami.data.processing import DataProcessor

def test_missing_data_handling():
    """Test missing data imputation strategies"""
    data = pd.DataFrame({
        'numeric': [1, np.nan, 3, 4, np.nan],
        'categorical': ['A', 'B', np.nan, 'C', np.nan]
    })
    
    processor = DataProcessor(
        num_impute_strategy='mean',
        cat_impute_strategy='most_frequent'
    )
    
    processed = processor.fit_transform(data)
    
    # Check numeric imputation
    assert processed['numeric'].isna().sum() == 0
    # The mean will be different after robust scaling
    # Just check if all values are finite
    assert np.all(np.isfinite(processed['numeric']))
    
    # Check that we have missing indicators
    assert 'numeric_missing' in processed.columns
    assert 'categorical_missing' in processed.columns
    
    # Check missing indicators are correct
    assert processed['numeric_missing'].sum() == 2
    assert processed['categorical_missing'].sum() == 2

def test_categorical_encoding():
    """Test categorical feature encoding with embeddings"""
    data = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B'],
        'numeric': [1, 2, 3, 4, 5]
    })
    
    processor = DataProcessor(
        cat_embed_dim='auto',
        handle_unknown='embed'
    )
    
    processed = processor.fit_transform(data)
    
    # Check embedding dimensions
    assert 'category_embed_0' in processed.columns
    # Check that we have the right number of embedding dimensions
    embed_cols = [col for col in processed.columns if col.startswith('category_embed')]
    # For auto, should be min(50, (cardinality+1)//2) = min(50, 2) = 2
    assert len(embed_cols) == 2
    
    # Check we have original numeric column plus embeddings
    assert processed.shape[1] == 1 + 2  # numeric + 2 embed dimensions

def test_temporal_processing():
    """Test time encoding with sinusoidal features"""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    data = pd.DataFrame({'event_date': dates})
    
    processor = DataProcessor(
        time_features=['sin']
    )
    
    processed = processor.fit_transform(data)
    
    # Check sinusoidal encoding
    assert 'event_date_sin' in processed.columns
    assert processed['event_date_sin'].between(-1, 1).all()

def test_feature_normalization():
    """Test adaptive feature normalization"""
    data = pd.DataFrame({
        'normal': [1, 2, 3, 4, 5],
        'skewed': [1, 2, 3, 4, 100]
    })
    
    processor = DataProcessor(
        normalize='robust'
    )
    
    processed = processor.fit_transform(data)
    
    # Check robust scaling - after scaling, median should be close to 0
    assert np.isclose(processed['skewed'].median(), 0)
    
    # The maximum value for 'skewed' will still be large due to the outlier,
    # but it should be scaled compared to the original (100)
    assert processed['skewed'].max() != 100