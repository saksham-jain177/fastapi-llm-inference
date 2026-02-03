import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
from app.models.confidence import ConfidenceEstimator, compute_agreement

# Set environment variable for hardware-independent inference
os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

pytestmark = pytest.mark.anyio


@pytest.fixture
def estimator():
    return ConfidenceEstimator()


def test_compute_agreement_identical():
    # Identical embeddings should have agreement 1.0
    embeddings = np.array([[1, 0, 0], [1, 0, 0]])
    assert compute_agreement(embeddings) == pytest.approx(1.0)


def test_compute_agreement_orthogonal():
    # Orthogonal embeddings should have agreement 0.0
    embeddings = np.array([[1, 0, 0], [0, 1, 0]])
    assert compute_agreement(embeddings) == pytest.approx(0.0)


async def test_estimate_confidence_high_agreement():
    estimator = ConfidenceEstimator()
    from unittest.mock import AsyncMock

    # Deterministic generate_fn returning identical responses
    mock_gen = AsyncMock(return_value="The capital of France is Paris.")

    response, confidence = await estimator.estimate_confidence(
        "What is the capital of France?", mock_gen
    )

    assert response == "The capital of France is Paris."
    assert confidence > 0.9
    assert mock_gen.call_count >= 4  # 3 low-temp + 1 perturbed


async def test_estimate_confidence_low_agreement():
    estimator = ConfidenceEstimator()
    from unittest.mock import AsyncMock

    # Generate_fn returning different responses
    responses = ["Response A", "Response B", "Response C", "Response D"]
    mock_gen = AsyncMock(side_effect=responses)

    response, confidence = await estimator.estimate_confidence("Test query", mock_gen)

    assert confidence < 0.8


async def test_estimate_confidence_perturbation_refusal():
    estimator = ConfidenceEstimator()

    # Phase 1 consistent, Phase 2 (perturbed) refuses
    async def mock_gen(query, temperature=0.1, max_new_tokens=50):
        if "don't know" in query:
            return "I don't know the answer."
        return "Confident response."

    response, confidence = await estimator.estimate_confidence("Adversarial query", mock_gen)

    assert response == "Confident response."
    assert confidence == 0.5  # Downgraded due to refusal in phase 2


async def test_orchestrator_decision_flow():
    """
    Test orchestrator routing using hardware-independent reasoner.
    No quantized model or Ollama required.
    """
    from app.routing.orchestrator import Orchestrator
    from app.reasoners.factory import reset_reasoner
    from unittest.mock import AsyncMock

    # Ensure deterministic reasoner is used
    reset_reasoner()
    os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

    # Mock dependencies
    with patch("app.models.calibration.get_confidence_threshold", return_value=0.7), \
         patch("app.routing.retrieval_gate.get_retrieval_gate") as mock_gate, \
         patch("app.routing.orchestrator.get_data_collector") as mock_get_collector:

        # Mock retrieval gate
        mock_gate().should_retrieve.return_value = False
        
        # Mock collector (ensure cache miss)
        mock_collector = AsyncMock()
        mock_collector.get_cached_response.return_value = None
        mock_get_collector.return_value = mock_collector

        orch = Orchestrator()
        result = await orch.route_and_execute("What is the capital of France?")

        assert "response" in result
        assert result["mode"] in ["internal-confident", "abstained", "rag", "adapter"]
        mock_collector.log_interaction.assert_called_once()

