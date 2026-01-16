"""
Test quantized model loading and inference.
"""
import os
import pytest
from unittest.mock import patch

# Skip GPU tests in CI
SKIP_GPU = os.getenv("CI") == "true" or os.getenv("USE_MOCK") == "true"


@pytest.mark.skipif(SKIP_GPU, reason="Skipping GPU tests in CI")
def test_load_quantized_model():
    """Test that quantized model loads successfully."""
    from app.models.quantized import load_model
    
    model, tokenizer = load_model()
    assert model is not None
    assert tokenizer is not None


@pytest.mark.skipif(SKIP_GPU, reason="Skipping GPU tests in CI")
def test_generate_response():
    """Test that model can generate a response."""
    from app.models.quantized import generate_response
    
    response = generate_response("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_model_info():
    """Test model info endpoint (works in mock mode)."""
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "mode" in data
