# Internal Knowledge Base

The Orchestrator's truth-first pipeline requires **evidence** before the model
may answer a general query. The internal KB (`app/kb/`) is the first evidence
source consulted — a local chromadb collection of curated documents — before
the pipeline falls back to external search (Tavily/DuckDuckGo).

## How it fits into routing

```
query → semantic classify → cache? → adapter?
      → KnowledgeGate:
            has_evidence = internal_kb.retrieve(query)   ← this module
            no evidence  → external RAG
            evidence + high epistemic confidence → model answers (mode: "model")
```

Responses served via the KB path include `"evidence": "internal_kb"` and
`"kb_sources": [...]` so provenance is auditable.

## Adding documents

### Option A — edit the sample corpus (code)

Append an entry to `SAMPLE_CORPUS` in `app/kb/seed_kb.py` and re-run:

```bash
ALLOW_EXPLICIT_EXECUTION=true uv run python -m app.kb.seed_kb
```

Seeding is idempotent — documents are upserted by stable ID.

### Option B — programmatic ingestion

```python
from app.kb import get_internal_kb

kb = get_internal_kb()
kb.ingest(text, doc_id="runbook_incident_response", source="wiki")
```

Documents are chunked on blank lines (default max 1000 chars/chunk).
Re-ingesting the same `doc_id` replaces its old chunks. `kb.remove(doc_id)`
deletes a document; `kb.count()` reports total stored chunks.

## Configuration (env vars)

| Variable            | Default            | Meaning                                          |
|---------------------|--------------------|--------------------------------------------------|
| `KB_PERSIST_DIR`    | `data/chroma_kb`   | chromadb persistence directory                   |
| `KB_COLLECTION`     | `internal_kb`      | collection name                                  |
| `KB_MIN_SIMILARITY` | `0.25`             | min similarity (1 − cosine distance) for evidence |

## Threshold relationships

Two independent gates decide whether the model may answer. They measure
**different things** and must be tuned separately:

| Gate | Env/config | Default | Unit | What it measures |
|------|-----------|---------|------|------------------|
| Evidence gate (this module) | `KB_MIN_SIMILARITY` | `0.25` | cosine-derived embedding similarity (0–1) | Does the query resemble curated KB content? |
| Epistemic gate (`app/models/calibration.py`) | `data/confidence_calibration.json` | `0.75` | model self-consistency confidence (0–1) | Is the model internally consistent about its draft answer? |

Routing consequence: a query first needs evidence (`has_evidence=True`, gated
by `KB_MIN_SIMILARITY`) **and then** epistemic confidence ≥ threshold to reach
the `model` path. Lowering `KB_MIN_SIMILARITY` widens what counts as
*evidence*; lowering the epistemic threshold widens when a *confident* answer
is trusted. Neither implies the other.

The offline report (`uv run python -m scripts.calibration_report`) shows both
values side by side but only ever SUGGESTS the epistemic threshold — changing
the evidence bar changes what counts as truth and stays a human decision.
Feedback labels (`incorrect`, `should_have_refused`) feed the epistemic
suggestion; they carry no embedding-similarity signal.

## Degradation

If chromadb or the embedding model is unavailable, retrieval returns
`has_evidence=False` and routing falls back to external RAG exactly as before
— the KB can never hard-fail a request.
