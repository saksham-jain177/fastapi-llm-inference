# Architecture Overview

This document outlines the core technical invariants and design patterns used to ensure hardware independence and system stability.

## Core Invariants

### 1. Import-Time Dependency Isolation

The system enforces a strict isolation rule for heavy or optional dependencies (e.g., `torch`, `transformers`, `ollama`, `better-profanity`).

- **Rule**: No optional or hardware-specific library may be imported at the module level.
- **Implementation**: All such dependencies are lazy-loaded within method or function scopes. This allows the application to be imported, analyzed, and tested in environments without specialized hardware or specific libraries.

### 2. Interface → Factory → Backend Pattern

To decouple orchestration logic from specific inference providers, the system utilizes a tiered abstraction:

- **Interface**: Defines the abstract contract (e.g., `Reasoner`, `Judge`, `Moderator`).
- **Factory**: Resolves the concrete implementation at runtime based on environment variables.
- **Backend**: The specific implementation (e.g., `OllamaReasoner`, `DeterministicModerator`).

## Deterministic CI Rationale

The system is designed to be fully testable without specialized hardware (GPUs) or external daemons.

- **Deterministic Backends**: Each component provides a rule-based or keyword-based implementation for CI.
- **Hardware Independence**: Setting `USE_DETERMINISTIC_INFERENCE=true` forces the factories to select these isolated backends. This ensures that CI validates logic flows, orchestration, and API integrity without infrastructure-induced failures.

## Developer Fixtures & Tooling

To ensure the reliability of complex data pipelines (e.g., RLHF, feedback loops), the repository contains guarded fixture generators under `scripts/fixtures/`.

- **Role**: These tools are used strictly for integration testing and pipeline verification.
- **Safety**: All fixture generators are environment-guarded (`ALLOW_FIXTURE_GENERATION=true`) to prevent accidental execution in non-development contexts.
- **Separation**: Fixtures are isolated from the core `app/` runtime and are not used in production or standard CI flows.
