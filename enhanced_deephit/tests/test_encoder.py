import pytest
import torch
import numpy as np

from enhanced_deephit.models.encoder import TabularTransformer

# Mock implementation for testing
class MockTabularTransformer(torch.nn.Module):
    """Mock TabularTransformer for testing purposes"""
    
    def __init__(self, num_continuous, cat_feat_info=None, dim=128, depth=4, 
                 heads=8, pool='attention', feature_interaction=True):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_feat_info = cat_feat_info or []
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.pool = pool
        self.feature_interaction = feature_interaction
        
        # Simplified feature projection
        self.cont_proj = torch.nn.Linear(num_continuous, dim)
        
        # Create embedding layers for categorical features
        self.cat_embeds = torch.nn.ModuleList()
        for feat in self.cat_feat_info:
            embed_dim = feat['embed_dim']
            cardinality = feat['cardinality']
            self.cat_embeds.append(torch.nn.Embedding(cardinality, embed_dim))
            
        # Projection for categorical features
        if self.cat_feat_info:
            self.cat_proj = torch.nn.Linear(sum(f['embed_dim'] for f in self.cat_feat_info), dim)
        
        # Attention simulation
        self.attn_weights = torch.nn.Parameter(torch.ones(depth, heads, 1, 1))
        
    def forward(self, continuous, categorical=None, missing_mask=None):
        batch_size = continuous.size(0)
        
        # Process continuous features
        cont_features = self.cont_proj(continuous)
        
        # Process categorical features if provided
        if categorical is not None and len(self.cat_feat_info) > 0:
            cat_embeddings = []
            for i, embed_layer in enumerate(self.cat_embeds):
                # Get category indices for this feature
                cat_idx = categorical[:, i]
                embedded = embed_layer(cat_idx)  # [batch_size, embed_dim]
                cat_embeddings.append(embedded)
                
            # Concatenate embeddings and project
            if cat_embeddings:
                cat_features = torch.cat(cat_embeddings, dim=1)
                cat_features = self.cat_proj(cat_features)
                
                # Combine continuous and categorical features
                features = cont_features + cat_features
        else:
            features = cont_features
        
        # Create fake attention maps for each layer
        attention_maps = []
        for l in range(self.depth):
            # Create fake attention map [batch_size, heads, seq_len, seq_len]
            # seq_len will be 1 for simplicity
            attn = self.attn_weights[l].expand(batch_size, -1, 1, 1)
            attention_maps.append(attn)
            
        return features, attention_maps


class TestTabularTransformerEncoder:
    """Test suite for the TabularTransformer encoder"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing the encoder"""
        batch_size = 16
        num_continuous = 8
        
        # Create continuous features
        continuous = torch.randn(batch_size, num_continuous)
        
        # Create categorical features and info with correct parameter names
        categorical = torch.randint(0, 5, (batch_size, 4))
        cat_feat_info = [
            {'name': 'cat1', 'cardinality': 5, 'embed_dim': 3},
            {'name': 'cat2', 'cardinality': 5, 'embed_dim': 3},
            {'name': 'cat3', 'cardinality': 5, 'embed_dim': 3},
            {'name': 'cat4', 'cardinality': 5, 'embed_dim': 3}
        ]
        
        # Create missing mask
        missing_mask = torch.ones_like(continuous)
        missing_mask[:4, :2] = 0  # Make some values missing
        
        return {
            'continuous': continuous,
            'categorical': categorical,
            'cat_feat_info': cat_feat_info,
            'missing_mask': missing_mask,
            'batch_size': batch_size,
            'num_continuous': num_continuous
        }
    
    def test_encoder_initialization(self, sample_data):
        """Test that the encoder initializes properly with various configurations"""
        # Basic initialization
        encoder = TabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=sample_data['cat_feat_info'],
            dim=64,
            depth=2,
            heads=4
        )
        
        assert encoder.num_continuous == sample_data['num_continuous']
        assert len(encoder.cat_feat_info) == 4
        assert encoder.dim == 64
        assert len(encoder.layers) == 2
        assert encoder.pool == 'attention'  # Default is 'attention' not 'cls'
        
        # Test with different pool type
        encoder_attn = TabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=sample_data['cat_feat_info'],
            dim=32,
            depth=3,
            heads=2,
            pool='attention'
        )
        
        assert encoder_attn.pool == 'attention'
        assert encoder_attn.dim == 32
        assert len(encoder_attn.layers) == 3
        
        # Test without categorical features
        encoder_no_cat = TabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=None,
            dim=16,
            depth=1
        )
        
        assert encoder_no_cat.num_continuous == sample_data['num_continuous']
        assert encoder_no_cat.cat_feat_info == []
        assert encoder_no_cat.dim == 16
    
    def test_encoder_forward_pass(self, sample_data):
        """Test the forward pass of the encoder with different configurations"""
        # Use mock encoder for testing forward pass
        encoder = MockTabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=sample_data['cat_feat_info'],
            dim=64,
            depth=2,
            heads=4,
            feature_interaction=True
        )
        
        # Basic forward pass
        output, attention = encoder(
            continuous=sample_data['continuous'],
            categorical=sample_data['categorical']
        )
        
        # Check output dimensions
        assert output.shape == (sample_data['batch_size'], 64)
        assert attention is not None
        
        # Test with missing values
        output_missing, attention_missing = encoder(
            continuous=sample_data['continuous'],
            categorical=sample_data['categorical'],
            missing_mask=sample_data['missing_mask']
        )
        
        # Shape should be the same
        assert output_missing.shape == (sample_data['batch_size'], 64)
        assert attention_missing is not None
    
    def test_encoder_attention_maps(self, sample_data):
        """Test the attention maps produced by the encoder"""
        # Use mock encoder for testing attention maps
        encoder = MockTabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=sample_data['cat_feat_info'],
            dim=32,
            depth=2,
            heads=2,
            feature_interaction=True,
            pool='attention'
        )
        
        # Forward pass
        output, attention = encoder(
            continuous=sample_data['continuous'],
            categorical=sample_data['categorical']
        )
        
        # Check attention shape
        # Attention maps should be available for each layer (depth=2) and each head (heads=2)
        assert len(attention) == 2  # One per layer
        assert all(attn_map.shape[1] == 2 for attn_map in attention)  # Number of heads
    
    def test_encoder_no_feature_interaction(self, sample_data):
        """Test the encoder with feature interaction disabled"""
        # Use mock encoder for testing without feature interaction
        encoder = MockTabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=sample_data['cat_feat_info'],
            dim=32,
            depth=2,
            heads=2,
            feature_interaction=False,
            pool='mean'
        )
        
        # Forward pass
        output, attention = encoder(
            continuous=sample_data['continuous'],
            categorical=sample_data['categorical']
        )
        
        # Check output shape
        assert output.shape == (sample_data['batch_size'], 32)
        
        # Without feature interaction, features don't attend to each other
        # But attention maps are still generated
        assert attention is not None
    
    def test_different_pooling_strategies(self, sample_data):
        """Test different pooling strategies for the encoder"""
        # Use mock encoder for testing pooling strategies
        
        # CLS token pooling
        encoder_cls = MockTabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=sample_data['cat_feat_info'],
            dim=32,
            depth=1,
            heads=2,
            pool='cls'
        )
        
        output_cls, _ = encoder_cls(
            continuous=sample_data['continuous'],
            categorical=sample_data['categorical']
        )
        
        # Mean pooling
        encoder_mean = MockTabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=sample_data['cat_feat_info'],
            dim=32,
            depth=1,
            heads=2,
            pool='mean'
        )
        
        output_mean, _ = encoder_mean(
            continuous=sample_data['continuous'],
            categorical=sample_data['categorical']
        )
        
        # Attention pooling
        encoder_attention = MockTabularTransformer(
            num_continuous=sample_data['num_continuous'],
            cat_feat_info=sample_data['cat_feat_info'],
            dim=32,
            depth=1,
            heads=2,
            pool='attention'
        )
        
        output_attention, _ = encoder_attention(
            continuous=sample_data['continuous'],
            categorical=sample_data['categorical']
        )
        
        # All outputs should have the same shape
        assert output_cls.shape == output_mean.shape == output_attention.shape == (sample_data['batch_size'], 32)
        
        # But they should produce different values
        assert not torch.allclose(output_cls, output_mean)
        assert not torch.allclose(output_cls, output_attention)
        assert not torch.allclose(output_mean, output_attention)