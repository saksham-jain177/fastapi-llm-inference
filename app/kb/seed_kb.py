"""
Seed the internal knowledge base with a small sample corpus.

Run:  uv run python -m app.kb.seed_kb
      (or) ALLOW_EXPLICIT_EXECUTION=true python scripts/training/seed_internal_kb.py

Idempotent: documents are upserted by stable ID, so re-running refreshes
rather than duplicates.
"""

import os

# Execution guardrail for direct script runs (mirrors scripts/training guards).
# Library imports (tests, other modules) are exempt.
if __name__ == "__main__" and os.getenv("ALLOW_EXPLICIT_EXECUTION") != "true":
    print("Note: set ALLOW_EXPLICIT_EXECUTION=true to acknowledge seeding.")

SAMPLE_CORPUS = {
    "company_faq": (
        "Acme Corp was founded in 2015 and is headquartered in Berlin. "
        "Its flagship product is the Atlas data pipeline.\n\n"
        "Acme Corp offers three support tiers: Basic (email, 48h response), "
        "Pro (email + chat, 8h response), and Enterprise (dedicated engineer, 1h response).\n\n"
        "Employees receive 30 days of paid vacation annually and unlimited sick leave."
    ),
    "product_docs_atlas": (
        "The Atlas data pipeline ingests batch and streaming sources through a "
        "unified connector API.\n\n"
        "Atlas pipelines are configured in YAML; each stage declares inputs, "
        "transforms, and sinks. Retries use exponential backoff with a default "
        "of five attempts.\n\n"
        "Atlas supports exactly-once delivery when sinks are idempotent and the "
        "checkpoint interval is left at its default of 30 seconds."
    ),
    "engineering_practices": (
        "All Acme services follow trunk-based development with short-lived "
        "feature branches merged via pull request review.\n\n"
        "Every service must expose /health/live and /health/ready endpoints and "
        "publish Prometheus metrics at /metrics.\n\n"
        "Production incidents are reviewed within 48 hours using blameless "
        "post-mortems stored in the internal wiki."
    ),
}


def seed(kb=None) -> dict:
    """Ingest SAMPLE_CORPUS into the KB. Returns {doc_id: chunks_stored}."""
    if kb is None:
        from app.kb import get_internal_kb

        kb = get_internal_kb()
    return {
        doc_id: kb.ingest(text, doc_id=doc_id, source="sample_corpus")
        for doc_id, text in SAMPLE_CORPUS.items()
    }


if __name__ == "__main__":
    results = seed()
    for doc_id, n in results.items():
        print(f"  {doc_id}: {n} chunks")
    from app.kb import get_internal_kb

    print(f"✅ KB seeded — {get_internal_kb().count()} total chunks")
