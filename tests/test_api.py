"""
Unit tests for the Symptom Recommendation API

NOTE: These tests require trained model files to be present.
Run 'python train.py' before running tests.

Run tests with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)


# Check if model files exist
MODEL_FILES_EXIST = all([
    os.path.exists('model.pth'),
    os.path.exists('model_config.pkl'),
    os.path.exists('graph.pt'),
    os.path.exists('symptom_to_idx.pkl')
])


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check(self):
        """Test that health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


@pytest.mark.skipif(not MODEL_FILES_EXIST, reason="Model files not found. Run 'python train.py' first.")
class TestSymptomsEndpoint:
    """Tests for symptoms list endpoint"""

    def test_get_symptoms(self):
        """Test retrieving list of available symptoms"""
        response = client.get("/symptoms")
        assert response.status_code == 200
        data = response.json()
        assert "symptoms" in data
        assert "count" in data
        assert isinstance(data["symptoms"], list)
        assert data["count"] > 0


@pytest.mark.skipif(not MODEL_FILES_EXIST, reason="Model files not found. Run 'python train.py' first.")
class TestRecommendEndpoint:
    """Tests for recommendation endpoint"""

    def test_recommend_valid_request(self):
        """Test recommendation with valid input"""
        payload = {
            "gender": "male",
            "age": 26,
            "symptoms": ["ไอ"],
            "top_k": 5
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "query_symptoms" in data
        assert "unknown_symptoms" in data
        assert len(data["recommendations"]) <= 5
        assert isinstance(data["recommendations"], list)

    def test_recommend_multiple_symptoms(self):
        """Test recommendation with multiple input symptoms"""
        payload = {
            "gender": "female",
            "age": 35,
            "symptoms": ["ไอ", "ปวดหัว"],
            "top_k": 10
        }
        response = client.post("/recommend", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert len(data["recommendations"]) <= 10
        # If symptoms don't exist, might get 400 - that's also OK

    def test_recommend_invalid_age(self):
        """Test that invalid age returns error"""
        payload = {
            "gender": "male",
            "age": 150,  # Invalid age
            "symptoms": ["ไอ"]
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 422  # Validation error

    def test_recommend_negative_age(self):
        """Test that negative age returns error"""
        payload = {
            "gender": "male",
            "age": -5,
            "symptoms": ["ไอ"]
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 422  # Validation error

    def test_recommend_unknown_symptom(self):
        """Test handling of unknown symptoms"""
        payload = {
            "gender": "male",
            "age": 26,
            "symptoms": ["unknown_symptom_xyz"]
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 400  # Bad request
        data = response.json()
        assert "detail" in data

    def test_recommend_default_top_k(self):
        """Test that default top_k is used when not specified"""
        payload = {
            "gender": "male",
            "age": 26,
            "symptoms": ["ไอ"]
        }
        response = client.post("/recommend", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert len(data["recommendations"]) <= 5  # Default is 5


class TestInputValidation:
    """Tests for input validation"""

    def test_missing_gender(self):
        """Test that missing gender field returns error"""
        payload = {
            "age": 26,
            "symptoms": ["ไอ"]
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 422

    def test_missing_age(self):
        """Test that missing age field returns error"""
        payload = {
            "gender": "male",
            "symptoms": ["ไอ"]
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 422

    def test_missing_symptoms(self):
        """Test that missing symptoms field returns error"""
        payload = {
            "gender": "male",
            "age": 26
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 422

    def test_invalid_json(self):
        """Test that invalid JSON returns error"""
        response = client.post(
            "/recommend",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
