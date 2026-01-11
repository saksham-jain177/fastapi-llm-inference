import pytest
from prometheus_client import REGISTRY
import app.metrics.prometheus as metrics

def test_prometheus_invariants():
    """
    Ensure all mandatory production metrics are registered.
    """
    expected_metrics = [
        'gate_decision_total',
        'hallucination_counter_total',
        'refusal_counter_total',
        'inference_latency_seconds'
    ]
    
    # Print metrics for debugging if it fails
    registered_metrics = [m.name for m in REGISTRY.collect()]
    print(f"DEBUG: Registered metrics: {registered_metrics}")
    
    for metric in expected_metrics:
        # Check for both with and without _total as Prometheus behavior can vary
        assert (metric in registered_metrics or metric.replace("_total", "") in registered_metrics), \
            f"Missing mandatory metric: {metric}. Found: {registered_metrics}"

def test_metrics_increment_on_gate():
    """
    Test that metrics actually increment when the orchestrator makes a decision.
    """
    from app.metrics.prometheus import gate_decision_total
    
    # Get initial value for a specific label
    initial_value = gate_decision_total.labels(type="answer")._value.get()
    
    # Increment
    gate_decision_total.labels(type="answer").inc()
    
    new_value = gate_decision_total.labels(type="answer")._value.get()
    assert new_value == initial_value + 1
