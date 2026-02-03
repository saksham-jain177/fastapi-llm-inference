"""
Confidence estimation for small language models using sampling disagreement.
Implements perturbation check to catch false confidence.
"""

from sentence_transformers import SentenceTransformer
from typing import Tuple, Callable
import numpy as np


def compute_agreement(embeddings: np.ndarray) -> float:
    """
    Compute semantic agreement via pairwise cosine similarities.
    
    Returns:
        Average similarity score ∈ [0, 1]
    """
    if len(embeddings) < 2:
        return 1.0
    
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities.append(sim)
    
    return float(np.mean(similarities))


class ConfidenceEstimator:
    """
    Estimates model confidence via multi-sample disagreement.
    Includes perturbation check to detect brittle false confidence.
    """
    
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    async def estimate_confidence(
        self, 
        query: str, 
        generate_fn: Callable,
        num_samples: int = 3
    ) -> Tuple[str, float]:
        """
        Two-phase confidence estimation:
        1. Low-temperature sampling (check basic agreement)
        2. Perturbation check (catch false confidence)
        """
        # Phase 1: Low-temperature sampling
        low_temp_samples = []
        for _ in range(num_samples):
            # generate_fn is now expected to be async
            sample = await generate_fn(query, temperature=0.7, max_new_tokens=50)
            low_temp_samples.append(sample)
        
        # Check low-temp agreement (encoding is CPU bound, keep sync for now per 'discipline')
        embeddings = self.encoder.encode(low_temp_samples)
        low_temp_agreement = compute_agreement(embeddings)
        
        print(f"  Phase 1: Low-temp agreement = {low_temp_agreement:.3f}")
        
        # Early exit if low agreement
        if low_temp_agreement < 0.75:
            return low_temp_samples[0], low_temp_agreement
        
        # Phase 2: Perturbation check (refusal-biased high-temp sample)
        refusal_prompt = (
            f"{query}\n\n"
            "If you are uncertain or this is outside your knowledge, "
            "say 'I don't know' instead of guessing."
        )
        
        perturbed_sample = await generate_fn(refusal_prompt, temperature=1.2, max_new_tokens=50)
        
        print(f"  Phase 2: Perturbed = '{perturbed_sample[:60]}...'")
        
        # Check if perturbation triggered refusal
        refusal_keywords = ["don't know", "uncertain", "not sure", "cannot answer"]
        if any(kw in perturbed_sample.lower() for kw in refusal_keywords):
            print(f"  → Perturbation triggered refusal, downgrading confidence")
            return low_temp_samples[0], 0.5  # Downgrade
        
        # Compare all samples (including perturbed)
        all_samples = low_temp_samples + [perturbed_sample]
        all_embeddings = self.encoder.encode(all_samples)
        final_confidence = compute_agreement(all_embeddings)
        
        print(f"  → Final confidence = {final_confidence:.3f}")
        
        return low_temp_samples[0], final_confidence


# Singleton instance
_estimator_instance = None

def get_confidence_estimator() -> ConfidenceEstimator:
    """Get singleton confidence estimator."""
    global _estimator_instance
    if _estimator_instance is None:
        _estimator_instance = ConfidenceEstimator()
    return _estimator_instance
