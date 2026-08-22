"""
Tests for the base-vs-quantized benchmark harness (scripts/benchmark_quantization).

Covers the pure metric primitives, deterministic-mode end-to-end run, and
report rendering. Real-model paths are intentionally NOT exercised here
(GPU-gated; use `uv run python -m scripts.benchmark_quantization --real`).
"""

import json

from scripts import benchmark_quantization as bq


class TestMetricPrimitives:
    def test_perplexity_of_uniform_loss(self):
        # exp(1.0) ~ 2.718
        assert bq.perplexity_from_losses([1.0, 1.0]) == 2.718281828459045

    def test_perplexity_empty_is_inf(self):
        assert bq.perplexity_from_losses([]) == float("inf")

    def test_lower_loss_means_lower_perplexity(self):
        low = bq.perplexity_from_losses([0.5] * 10)
        high = bq.perplexity_from_losses([2.0] * 10)
        assert low < high

    def test_rouge_l_identical_strings(self):
        assert bq.rouge_l_f1("the capital is paris", "the capital is paris") == 1.0

    def test_rouge_l_disjoint_strings(self):
        assert bq.rouge_l_f1("alpha beta", "gamma delta") == 0.0

    def test_rouge_l_partial_overlap(self):
        score = bq.rouge_l_f1("paris is the capital of france", "the capital of france is paris")
        assert 0.0 < score <= 1.0

    def test_rouge_l_case_insensitive(self):
        assert bq.rouge_l_f1("Paris", "paris") == 1.0


class TestDeterministicRun:
    def test_run_benchmark_deterministic_shape(self):
        result = bq.run_benchmark(use_real_models=False)
        assert result["mode"] == "deterministic"
        assert result["eval_samples"] == len(bq.EVAL_SET)
        for section in ("base_fp16", "quantized_nf4"):
            rep = result[section]
            for key in ("perplexity", "avg_latency_seconds", "tokens_per_second", "samples"):
                assert key in rep
            assert rep["samples"] == len(bq.EVAL_SET)

    def test_deterministic_metrics_are_stable_across_runs(self):
        r1 = bq.run_benchmark(use_real_models=False)
        r2 = bq.run_benchmark(use_real_models=False)
        assert r1["base_fp16"]["perplexity"] == r2["base_fp16"]["perplexity"]
        assert (
            r1["quantized_nf4"]["agreement_with_base"]
            == r2["quantized_nf4"]["agreement_with_base"]
        )

    def test_comparison_block_present(self):
        result = bq.run_benchmark(use_real_models=False)
        c = result["comparison"]
        assert "perplexity_delta_pct" in c
        assert "latency_ratio_quant_over_base" in c
        assert c["agreement_with_base"] == 1.0  # simulated outputs identical


class TestRenderAndCli:
    def test_render_text_contains_key_sections(self):
        result = bq.run_benchmark(use_real_models=False)
        text = bq.render(result)
        assert "Quantization benchmark" in text
        assert "perplexity" in text
        assert "Agreement with base" in text

    def test_render_json_round_trips(self):
        result = bq.run_benchmark(use_real_models=False)
        parsed = json.loads(bq.render(result, as_json=True))
        assert parsed["mode"] == result["mode"]
        assert parsed["comparison"] == result["comparison"]

    def test_cli_main_exit_code_zero(self, capsys):
        rc = bq.main(["--json"])
        assert rc == 0
        out = capsys.readouterr().out
        assert json.loads(out)["mode"] == "deterministic"


class TestEvalSet:
    def test_every_entry_has_prompt_and_reference(self):
        for entry in bq.EVAL_SET:
            assert entry["prompt"].strip()
            assert entry["reference"].strip()

    def test_eval_set_size_reasonable(self):
        assert 4 <= len(bq.EVAL_SET) <= 32
