# Technical Walkthrough: Adaptive Inference & Logic Flow

This document describes the technical execution flow, decision logic, and metadata semantics of the FastAPI LLM Inference service as of Phase 4.

## 1. Request Execution Flow (/infer-adaptive)

The `/infer-adaptive` endpoint implements an agentic routing strategy that prioritizes speed and cost efficiency by layer-gating expensive LLM calls.

### Decision Order (Priority Stack)

1.  **Redis (Hot Cache Layer)**:
    - The system first checks Redis for an exact query match.
    - If found, it returns immediately with `source: redis`.
    - This layer bypasses all semantic analysis and database lookups.

2.  **MongoDB / Semantic Memory (Durable Layer)**:
    - If Redis misses, the system performs a semantic search via `FeedbackRetriever` (backed by ChromaDB and MongoDB logs).
    - If a match meets the `MEMORY_ACCEPT_THRESHOLD` (default 0.85), it returns with `source: memory`.
    - The hit is then cached in Redis for future requests.

3.  **Active Routing (Analysis Layer)**:
    - If memory misses, the `QueryAnalyzer` classifies the intent into:
      - `external_search`: Triggers the RAG pipeline.
      - `complex_reasoning`: Triggers the Chain-of-Thought (CoT) reasoning path.
      - `simple_internal`: Triggers the domain-adapter or base model path.

4.  **RAG / Reasoning (Execution Layer)**:
    - **RAG**: Performs external web search (Tavily), synthesizes context, and returns with `source: rag`.
    - **Reasoning**: Executes internal reasoning via the `Reasoner` interface and returns with `source: model`.
    - **Adapter**: Routes to a specific fine-tuned LoRA adapter if applicable.

5.  **Epistemic Gating (Safety Layer)**:
    - For `simple_internal` queries with low model confidence:
      - The system checks query novelty.
      - If the query is novel, it triggers a RAG fallback.
      - If the query is _not_ novel but the model is unsure, it triggers a forced abstention.

6.  **Forced Abstention (Refusal)**:
    - If epistemic gating blocks the response, it returns `refused: true` with a canonical refusal message and `source: refused`.

---

## 2. Response Metadata Semantics

The API Response contract includes four key semantic signals:

- **`confidence`**: A floating-point value [0.0 - 1.0]. Note that this is **epistemic confidence** (internal consistency/perturbation agreement), not a probabilistic guarantee of accuracy.
- **`intent`**: The classified objective of the query (`simple_internal`, `complex_reasoning`, `external_search`).
- **`source`**: The physical layer that fulfilled the request (`redis`, `memory`, `rag`, `model`, `refused`).
- **`refused`**: A boolean flag indicating whether the safety/epistemic guardrail blocked a generative response.

---

## 3. Deterministic vs. Runtime Behavior

The system supports a hardware-independent **Deterministic Mode** for verification and CI/CD:

- **Runtime Mode**: Uses quantized models (GGUF), GPU acceleration (llama-cpp-python), and live infrastructure (Redis/Mongo).
- **Deterministic Mode** (`USE_DETERMINISTIC_INFERENCE=true`):
  - Bypasses heavy weights/GPU requirements.
  - Uses keyword-based routing and length-based heuristics for semantic components.
  - Ensures consistent execution paths for non-functional verification.
