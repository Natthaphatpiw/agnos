"""
Unit tests for the ML Model

Run tests with: pytest tests/test_model.py -v
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SymptomRecommender


class TestModelInitialization:
    """Tests for model initialization"""

    def test_model_creation(self):
        """Test that model can be created with valid parameters"""
        model = SymptomRecommender(
            num_symptoms=338,
            num_patients=1000,
            symptom_embed_dim=128,
            demog_embed_dim=16,
            gat_heads=4,
            dropout=0.3
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_parameters(self):
        """Test that model has trainable parameters"""
        model = SymptomRecommender(
            num_symptoms=338,
            num_patients=1000
        )
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_model_embeddings(self):
        """Test embedding dimensions"""
        num_symptoms = 338
        embed_dim = 128
        model = SymptomRecommender(
            num_symptoms=num_symptoms,
            num_patients=1000,
            symptom_embed_dim=embed_dim
        )
        # Check symptom embeddings (note: plural form)
        assert model.symptom_embeddings.num_embeddings == num_symptoms
        assert model.symptom_embeddings.embedding_dim == embed_dim


class TestModelForward:
    """Tests for forward pass"""

    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        return SymptomRecommender(
            num_symptoms=100,
            num_patients=50,
            symptom_embed_dim=32,
            demog_embed_dim=8,
            gat_heads=2,
            dropout=0.3
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data with correct dimensions"""
        batch_size = 4
        return {
            'patient_idx': torch.randint(0, 50, (batch_size,)),
            'gender': torch.randn(batch_size, 2),  # 2-dim one-hot
            'age_bin': torch.randn(batch_size, 4),  # 4-dim one-hot
            'symptom_idx': torch.randint(0, 100, (batch_size,)),
        }

    @pytest.fixture
    def sample_edge_index(self):
        """Create sample edge index for graph"""
        # Create a simple graph with 10 edges
        return torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        ], dtype=torch.long)

    def test_forward_pass_shape(self, sample_model, sample_batch, sample_edge_index):
        """Test that forward pass returns correct shape"""
        sample_model.eval()
        with torch.no_grad():
            output = sample_model(
                sample_batch['patient_idx'],
                sample_batch['gender'],
                sample_batch['age_bin'],
                sample_batch['symptom_idx'],
                sample_edge_index
            )
        assert output.shape[0] == 4  # batch_size
        assert output.shape[1] == 100  # num_symptoms

    def test_forward_pass_values(self, sample_model, sample_batch, sample_edge_index):
        """Test that forward pass returns valid probability-like values"""
        sample_model.eval()
        with torch.no_grad():
            output = sample_model(
                sample_batch['patient_idx'],
                sample_batch['gender'],
                sample_batch['age_bin'],
                sample_batch['symptom_idx'],
                sample_edge_index
            )
        # After sigmoid, values should be between 0 and 1
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_forward_pass_training_mode(self, sample_model, sample_batch, sample_edge_index):
        """Test forward pass in training mode"""
        sample_model.train()
        output = sample_model(
            sample_batch['patient_idx'],
            sample_batch['gender'],
            sample_batch['age_bin'],
            sample_batch['symptom_idx'],
            sample_edge_index
        )
        assert output.shape[0] == 4
        assert output.requires_grad  # Should have gradients in training mode


class TestModelComponents:
    """Tests for individual model components"""

    def test_fusion_weights(self):
        """Test that fusion weights are initialized correctly"""
        model = SymptomRecommender(num_symptoms=100, num_patients=50)
        fusion_weights = model.fusion_weights.data
        # Should have 3 components: CF, CB, Graph
        assert len(fusion_weights) == 3
        # Should sum to approximately 1.0 after softmax
        assert torch.allclose(fusion_weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_demographics_encoder(self):
        """Test demographics encoder accepts correct input shape"""
        model = SymptomRecommender(num_symptoms=100, num_patients=50)

        # Demographics: 2 (gender) + 4 (age_bin) = 6 dimensions
        batch_size = 4
        gender = torch.randn(batch_size, 2)
        age_bin = torch.randn(batch_size, 4)

        # Should concatenate to 6 dimensions and encode
        demog_input = torch.cat([gender, age_bin], dim=1)
        assert demog_input.shape == (4, 6)

        # Pass through encoder
        output = model.demog_encoder(demog_input)
        assert output.shape[1] == 16  # demog_embed_dim

    def test_gat_layers_exist(self):
        """Test that GAT layers are properly initialized"""
        model = SymptomRecommender(
            num_symptoms=100,
            num_patients=50,
            gat_heads=4
        )
        assert hasattr(model, 'gat1')
        assert hasattr(model, 'gat2')


class TestModelRobustness:
    """Tests for model robustness"""

    def test_model_with_valid_input(self):
        """Test model behavior with valid input"""
        model = SymptomRecommender(num_symptoms=100, num_patients=50)
        model.eval()

        batch_size = 4
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        with torch.no_grad():
            output = model(
                torch.randint(0, 50, (batch_size,)),
                torch.randn(batch_size, 2),
                torch.randn(batch_size, 4),
                torch.randint(0, 100, (batch_size,)),
                edge_index
            )

        # Should produce valid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)


class TestModelSaveLoad:
    """Tests for saving and loading model"""

    def test_model_state_dict(self):
        """Test that model state dict can be saved and loaded"""
        model1 = SymptomRecommender(num_symptoms=100, num_patients=50)
        state_dict = model1.state_dict()

        model2 = SymptomRecommender(num_symptoms=100, num_patients=50)
        model2.load_state_dict(state_dict)

        # Check that parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
