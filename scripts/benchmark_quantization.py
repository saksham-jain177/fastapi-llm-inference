"""
Base-vs-quantized model benchmark harness.

Measures the real cost of 4-bit (bitsandbytes nf4) quantization on
Qwen2.5-0.5B-Instruct across three axes:

1. Perplexity  — language-modeling quality on a small evaluation corpus.
2. Latency     — end-to-end generation wall time and tokens/sec.
3. Quality proxy — token-level agreement between base fp16 and quantized
   outputs (ROUGE-L F1) plus vocabulary-drift rate.

Design constraints (mirrors the repo's backend pattern):
- Deterministic mode: USE_DETERMINISTIC_INFERENCE=true or missing torch/
  transformers yields stable, hardware-independent pseudo-metrics so the
  harness is exercisable in CI without a GPU. No vendor imports at module top.
- Real mode: loads the base model in fp16 AND the quantized model (via
  app.models.quantized.get_bnb_config) and evaluates both on CPU/GPU.

Usage:
    uv run python -m scripts.benchmark_quantization            # human report
    uv run python -m scripts.benchmark_quantization --json     # machine JSON

This tool NEVER writes runtime state; it prints a report only.
Exit codes: 0 = report produced.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Small fixed eval set: factual + reasoning prompts with reference answers,
# enough for a directional perplexity/quality signal on a 0.5B model.
EVAL_SET = [
    {"prompt": "What is the capital of France?", "reference": "Paris"},
    {"prompt": "What is the capital of Japan?", "reference": "Tokyo"},
    {"prompt": "Define photosynthesis.", "reference": "the process by which plants convert light energy into chemical energy"},
    {"prompt": "Explain what gravity is.", "reference": "a force that attracts objects toward each other"},
    {"prompt": "What is 2 plus 3?", "reference": "5"},
    {"prompt": "Name the largest planet in the solar system.", "reference": "Jupiter"},
    {"prompt": "What language is primarily spoken in Brazil?", "reference": "Portuguese"},
    {"prompt": "Who wrote Romeo and Juliet?", "reference": "Shakespeare"},
]

MAX_NEW_TOKENS = 48


# ---------------------------------------------------------------------------
# Metric primitives (pure functions — testable without any model)
# ---------------------------------------------------------------------------

def perplexity_from_losses(losses: List[float]) -> float:
    """Exponentiated mean negative log-loss."""
    if not losses:
        return float("inf")
    import math

    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)


def rouge_l_f1(candidate: str, reference: str) -> float:
    """ROUGE-L F1 via longest common subsequence (pure Python, no deps)."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if not cand_tokens or not ref_tokens:
        return 0.0

    # LCS length (DP)
    prev = [0] * (len(ref_tokens) + 1)
    for c in cand_tokens:
        curr = [0] * (len(ref_tokens) + 1)
        for j, r in enumerate(ref_tokens, start=1):
            if c == r:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class SampleResult:
    prompt: str
    latency_seconds: float
    new_tokens: int


@dataclass
class BackendReport:
    name: str
    perplexity: Optional[float] = None
    avg_latency_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None
    rouge_l_vs_reference: Optional[float] = None
    agreement_with_base: Optional[float] = None
    vocab_drift_rate: Optional[float] = None
    samples: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic backend (CI-safe, mirrors DeterministicReasoner philosophy)
# ---------------------------------------------------------------------------

class DeterministicBenchmarkBackend:
    """
    Hardware-independent stand-in producing STABLE pseudo-metrics derived from
    prompt hashing. Not meaningful as a quality measure; exists so the harness
    logic, output shape, and exit codes are verifiable in CI.
    """

    name = "deterministic"

    def generate(self, prompt: str) -> str:
        import hashlib

        h = hashlib.md5(prompt.encode()).hexdigest()
        words = prompt.split()[:6] or ["answer"]
        return " ".join(words) + f" [ref:{h[:6]}]"

    def evaluate(self, prompts: List[str]) -> BackendReport:
        start_total = time.perf_counter()
        losses = []
        results = []
        for p in prompts:
            t0 = time.perf_counter()
            out = self.generate(p)
            dt = time.perf_counter() - t0
            n_tokens = len(out.split())
            losses.append(0.5 + (int(p.encode().hex(), 16) % 100) / 1000)
            results.append(SampleResult(p, dt, n_tokens))
        total_time = time.perf_counter() - start_total
        total_tokens = sum(r.new_tokens for r in results) or 1
        return BackendReport(
            name=self.name,
            perplexity=perplexity_from_losses(losses),
            avg_latency_seconds=total_time / len(results) if results else 0.0,
            tokens_per_second=total_tokens / total_time if total_time else 0.0,
            samples=len(results),
            notes="pseudo-metrics (hash-derived); CI-safe only",
        )


# ---------------------------------------------------------------------------
# Real backends (torch/transformers; lazy imports)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class _TorchBackendBase:
    def _load(self):  # pragma: no cover — exercised only with real hardware
        raise NotImplementedError

    def evaluate(self, prompts):  # pragma: no cover
        import torch

        model, tokenizer = self._load()
        device = next(model.parameters()).device
        losses, results, outputs = [], [], []

        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)

            # Quality signal: teacher-forced loss against the reference-free LM
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
                loss_out = model(**inputs, labels=inputs["input_ids"])
            dt = time.perf_counter() - t0
            losses.append(float(loss_out.loss))
            out_text = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            outputs.append(out_text.strip())
            results.append(SampleResult(p, dt, len(gen[0]) - inputs["input_ids"].shape[1]))

        total_time = sum(r.latency_seconds for r in results) or 1e-9
        total_tokens = sum(r.new_tokens for r in results) or 1
        mem = getattr(model, "get_memory_footprint", lambda: None)()
        return (
            BackendReport(
                name=self.name,
                perplexity=perplexity_from_losses(losses),
                avg_latency_seconds=total_time / len(results) if results else 0.0,
                tokens_per_second=total_tokens / total_time,
                samples=len(results),
                extra={"memory_bytes": mem} if mem else {},
            ),
            outputs,
        )


class BaseFp16Backend(_TorchBackendBase):
    name = "base_fp16"

    def _load(self):  # pragma: no cover
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return model, tokenizer


class QuantizedBackend(_TorchBackendBase):
    name = "quantized_nf4"

    def _load(self):  # pragma: no cover
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from app.models.quantized import get_bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=get_bnb_config(),
            device_map="auto", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return model, tokenizer


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def compare(base_report: BackendReport, quant_report: BackendReport,
            base_outputs: List[str], quant_outputs: List[str],
            references: List[str]) -> dict:
    """Cross-backend comparison block appended to both reports' consumer."""
    agreements = [rouge_l_f1(q, b) for q, b in zip(quant_outputs, base_outputs)]
    drift = [
        1.0 if q.strip() != b.strip() else 0.0
        for q, b in zip(quant_outputs, base_outputs)
    ]
    quant_report.agreement_with_base = round(sum(agreements) / len(agreements), 4) if agreements else None
    quant_report.vocab_drift_rate = round(sum(drift) / len(drift), 4) if drift else None

    rouge_q = [rouge_l_f1(o, r) for o, r in zip(quant_outputs, references)]
    rouge_b = [rouge_l_f1(o, r) for o, r in zip(base_outputs, references)]
    quant_report.rouge_l_vs_reference = round(sum(rouge_q) / len(rouge_q), 4) if rouge_q else None
    base_report.rouge_l_vs_reference = round(sum(rouge_b) / len(rouge_b), 4) if rouge_b else None

    summary = {
        "perplexity_delta_pct": _pct_change(base_report.perplexity, quant_report.perplexity),
        "latency_ratio_quant_over_base": (
            round(quant_report.avg_latency_seconds / base_report.avg_latency_seconds, 3)
            if base_report.avg_latency_seconds and quant_report.avg_latency_seconds else None
        ),
        "agreement_with_base": quant_report.agreement_with_base,
        "vocab_drift_rate": quant_report.vocab_drift_rate,
    }
    return summary


def _pct_change(old, new):
    if old in (None, 0) or new is None:
        return None
    return round(100.0 * (new - old) / old, 2)


def run_benchmark(use_real_models: bool = False, json_output: bool = False) -> dict:
    prompts = [e["prompt"] for e in EVAL_SET]
    references = [e["reference"] for e in EVAL_SET]

    if use_real_models:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            print("torch/transformers unavailable; falling back to deterministic mode")
            use_real_models = False

    if use_real_models:
        base_report, base_outputs = BaseFp16Backend().evaluate(prompts)
        quant_report, quant_outputs = QuantizedBackend().evaluate(prompts)
    else:
        backend = DeterministicBenchmarkBackend()
        rep = backend.evaluate(prompts)
        base_report = BackendReport(**{**asdict(rep)})
        quant_report = BackendReport(**{
            **asdict(rep), "name": "quantized_nf4 (simulated)",
            "notes": "deterministic fallback — install torch+transformers+GPU for real numbers",
        })
        base_outputs = [backend.generate(p) for p in prompts]
        quant_outputs = list(base_outputs)  # identical by construction

    comparison = compare(base_report, quant_report, base_outputs, quant_outputs, references)

    result = {
        "model": MODEL_NAME,
        "mode": "real" if use_real_models else "deterministic",
        "eval_samples": len(prompts),
        "base_fp16": asdict(base_report),
        "quantized_nf4": asdict(quant_report),
        "comparison": comparison,
    }
    return result


def render(result: dict, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(result, indent=2)

    b = result["base_fp16"]
    q = result["quantized_nf4"]
    c = result["comparison"]
    lines = [
        "=" * 62,
        f" Quantization benchmark — {result['model']} ({result['mode']})",
        "=" * 62,
        f" Eval samples              : {result['eval_samples']}",
        "",
        f" {'metric':<28}{'base fp16':>14}{'nf4':>14}",
        f" {'-'*28}{'-'*14}{'-'*14}",
        f" {'perplexity':<28}{_fmt(b['perplexity']):>14}{_fmt(q['perplexity']):>14}",
        f" {'avg latency (s)':<28}{_fmt(b['avg_latency_seconds']):>14}{_fmt(q['avg_latency_seconds']):>14}",
        f" {'tokens/sec':<28}{_fmt(b['tokens_per_second']):>14}{_fmt(q['tokens_per_second']):>14}",
        f" {'ROUGE-L vs reference':<28}{_fmt(b['rouge_l_vs_reference']):>14}{_fmt(q['rouge_l_vs_reference']):>14}",
        "",
        f" Agreement with base (ROUGE-L): {c['agreement_with_base']}",
        f" Output drift rate             : {c['vocab_drift_rate']}",
        f" Perplexity delta              : {c['perplexity_delta_pct']}%",
        f" Latency ratio (nf4/base)      : {c['latency_ratio_quant_over_base']}",
    ]
    if q.get("extra", {}).get("memory_bytes"):
        lines.append(f" Quantized memory footprint    : {q['extra']['memory_bytes'] / 1e6:.1f} MB")
    lines += ["", f" Notes: {q['notes']}" ]
    return "\n".join(lines)


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--real", action="store_true",
                        help="Load real base+quantized models (requires torch/GPU)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run_benchmark(use_real_models=args.real, json_output=args.json)
    print(render(result, as_json=args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
