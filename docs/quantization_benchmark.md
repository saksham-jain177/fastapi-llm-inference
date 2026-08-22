# Quantization Benchmark Harness

Measures the real cost of 4-bit quantization (`bitsandbytes` nf4 — the config
in `app/models/quantized.py`) against the fp16 base model on
**Qwen2.5-0.5B-Instruct**, across three axes:

1. **Perplexity** — teacher-forced language-modeling loss over a fixed eval set.
2. **Latency** — end-to-end generation wall time and tokens/sec.
3. **Quality proxy** — ROUGE-L of outputs vs references, token-level agreement
   between base and quantized outputs, and output-drift rate.

## Usage

```bash
# CI-safe deterministic mode (no GPU/torch needed; pseudo-metrics)
uv run python -m scripts.benchmark_quantization

# Real benchmark (loads both models; requires torch + transformers + GPU)
uv run python -m scripts.benchmark_quantization --real

# Machine-readable
uv run python -m scripts.benchmark_quantization --json
```

The tool is read-only: it never writes runtime state or model artifacts.

## Interpreting results

| Metric | Meaning | Healthy nf4 range (0.5B models) |
|--------|---------|-------------------------------|
| perplexity delta | % change in LM loss vs fp16 | < +5% |
| latency ratio (nf4/base) | generation speed factor | ~0.8–2.0× (nf4 saves VRAM but can be slower on some GPUs) |
| agreement with base | ROUGE-L overlap between backends' answers | > 0.6 |
| drift rate | fraction of samples with any output difference | informational |

Perplexity is computed on the fixed `EVAL_SET` in
`scripts/benchmark_quantization.py` (small factual/reasoning prompts). It is a
directional signal, not an absolute quality verdict — extend `EVAL_SET` with
domain-representative prompts before drawing conclusions.

## Architecture notes

- Mirrors the repo's backend pattern: vendor imports (`torch`,
  `transformers`, bitsandbytes) happen only inside execution branches, never
  at module top level, so importing the script is always side-effect-free.
- Deterministic mode derives stable pseudo-metrics from prompt hashing — same
  philosophy as `DeterministicReasoner`. It exists to keep harness logic,
  report shape, and exit codes testable in CI.
- Metric primitives (`perplexity_from_losses`, `rouge_l_f1`) are pure
  functions with no dependencies — see `tests/test_quantization_benchmark.py`.
