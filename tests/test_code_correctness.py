"""Tests for benchmarks/code_correctness.py (Sylveste-b7j).

Exercises the dry-run path end-to-end and proves:
  1. Dry-run does not import mlx (swe-bench-lite stub path).
  2. LCB executor actually runs subprocess python (correct stub → pass@1 = 1.0).
  3. CLI entrypoint returns non-zero on bad args and zero on success.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Allow `from benchmarks.code_correctness import ...` when pytest cwd is elsewhere.
sys.path.insert(0, str(REPO_ROOT))

from benchmarks import code_correctness as cc  # noqa: E402


def test_resolve_config_known_alias():
    name, cfg = cc.resolve_config("local:qwen3.6-35b")
    assert name == "35b-3.6"
    assert cfg["backend"] == "mlx"


def test_resolve_config_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        cc.resolve_config("local:does-not-exist")


def test_resolve_config_deepseek_v4_aliases():
    """Sylveste-bvh: DeepSeek V4 cloud aliases route via OpenRouter OpenAI-compat."""
    for alias, expected_model in [
        ("cloud:deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
        ("cloud:deepseek-v4-pro", "deepseek/deepseek-v4-pro"),
    ]:
        name, cfg = cc.resolve_config(alias)
        assert cfg["backend"] == "cloud"
        assert cfg["provider"] == "openai"
        assert cfg["model"] == expected_model
        assert cfg["base_url"] == "https://openrouter.ai/api/v1"
        assert cfg["api_key_env"] == "OPENROUTER_API_KEY"
        assert cfg["reasoning_effort"] == "high"


def test_generate_cloud_dispatch_unknown_provider():
    """An unrecognized provider raises ValueError before any network call."""
    from benchmarks.holistic_benchmark import _generate_cloud

    with pytest.raises(ValueError, match="Unknown cloud provider"):
        _generate_cloud(
            {"provider": "fictional", "model": "x"},
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )


def test_generate_cloud_openai_compat_missing_key(monkeypatch):
    """OpenAI-compat path raises a clear error when api_key_env is missing."""
    from benchmarks.holistic_benchmark import _generate_cloud_openai_compat

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = {
        "provider": "openai",
        "model": "deepseek/deepseek-v4-flash",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    }
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _generate_cloud_openai_compat(
            cfg,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            timeout=30.0,
        )


def test_generate_cloud_string_back_compat():
    """Old call sites that pass a bare model string still default to Anthropic."""
    from benchmarks.holistic_benchmark import _generate_cloud

    # Don't actually call out — just verify the dispatch decision by inspecting
    # what _generate_cloud_anthropic would receive. We use a sentinel that will
    # blow up inside the anthropic client if reached, proving we routed there
    # (and not to the openai path which would raise a different error).
    import pytest as _pytest

    # An unknown anthropic model id will fail inside the SDK, but the dispatch
    # itself shouldn't raise ValueError("Unknown cloud provider").
    with _pytest.raises(Exception) as exc_info:
        _generate_cloud(
            "claude-fake-model-for-test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            timeout=1.0,
        )
    assert "Unknown cloud provider" not in str(exc_info.value)


def test_generate_cloud_openai_compat_records_reasoning_tokens(monkeypatch):
    """Sylveste-uk3: reasoning tokens are captured from the stream usage block
    and returned as a field separate from tokens_generated (output-only).

    OpenRouter's final usage-only event carries
    completion_tokens_details.reasoning_tokens; without plumbing it through we
    undercount billable output for reasoning-mode cloud models.
    """
    import types

    from benchmarks import holistic_benchmark as hb

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Build a fake SSE stream: one content delta, then a usage-only event.
    content_delta = types.SimpleNamespace(content="42", reasoning=None)
    content_event = types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=content_delta)],
        usage=None,
    )
    usage_details = types.SimpleNamespace(reasoning_tokens=1234)
    usage_event = types.SimpleNamespace(
        choices=[],
        usage=types.SimpleNamespace(
            completion_tokens=7,
            completion_tokens_details=usage_details,
        ),
    )

    class _FakeCompletions:
        def create(self, **kwargs):
            return iter([content_event, usage_event])

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = _FakeChat()

    # Patch the OpenAI symbol imported lazily inside the function.
    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    cfg = {
        "provider": "openai",
        "model": "deepseek/deepseek-v4-flash",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "reasoning_effort": "high",
    }
    result = hb._generate_cloud_openai_compat(
        cfg,
        messages=[{"role": "user", "content": "what is 6*7?"}],
        max_tokens=64,
        timeout=30.0,
    )

    assert result["tokens_generated"] == 7  # output only
    assert result["reasoning_tokens"] == 1234  # not conflated into output
    assert result["output_text"] == "42"

    # And the field is carried on GenerationResult (default 0 elsewhere).
    gr = hb.GenerationResult(
        config="cloud:deepseek-v4-flash",
        prompt_id="p1",
        category="",
        difficulty="",
        output_text=result["output_text"],
        tokens_generated=result["tokens_generated"],
        elapsed_s=result["elapsed_s"],
        ttft_s=result["ttft_s"],
        gen_tps=result["gen_tps"],
        peak_mem_gb=result["peak_mem_gb"],
        thermal_start="",
        thermal_end="",
        timestamp="2026-05-06T00:00:00Z",
        reasoning_tokens=result["reasoning_tokens"],
    )
    assert gr.to_dict()["reasoning_tokens"] == 1234


def test_swebench_lite_dry_run_emits_zero_pass(tmp_path):
    # Matches the bead's verification command precisely.
    sc = cc.run_suite(
        suite="swe-bench-lite",
        model_name="local:qwen3.6-35b",
        output_dir=tmp_path,
        dry_run=True,
    )
    assert sc.n_problems == 2
    assert sc.n_passed == 0
    assert sc.pass_at_1 == 0.0
    assert sc.errors == 0
    assert (tmp_path / "code_correctness.jsonl").exists()

    lines = (tmp_path / "code_correctness.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert rec["suite"] == "swe-bench-lite"
        assert rec["model"] == "local:qwen3.6-35b"
        assert rec["passed"] is False


def _correct_lcb_generator(prompt: str):
    """Synthesise a correct solution for whichever dry-run LCB problem this is.

    Both dry-run problems read a single stdin line; we discriminate by prompt
    body keywords and emit the minimal correct program.
    """
    if "verbatim" in prompt:
        code = "import sys\nprint(sys.stdin.read().rstrip('\\n'))"
    elif "sum" in prompt or "Sum" in prompt:
        code = "a, b = map(int, input().split())\nprint(a + b)"
    else:
        raise AssertionError(f"Unexpected LCB prompt: {prompt[:120]}")
    return {
        "output_text": f"```python\n{code}\n```",
        "tokens_generated": 20,
        "elapsed_s": 0.001,
        "ttft_s": 0.0,
        "gen_tps": 20000.0,
        "peak_mem_gb": 0.0,
        "timed_out": False,
    }


def test_livecodebench_real_executor_scores_correct_solution(tmp_path):
    """If the generator emits correct Python, LCB should score 2/2."""
    sc = cc.run_suite(
        suite="livecodebench-v6",
        model_name="local:qwen3.6-35b",
        output_dir=tmp_path,
        dry_run=True,  # uses fixture problems
        generator=_correct_lcb_generator,
    )
    assert sc.n_problems == 2
    assert sc.n_passed == 2
    assert sc.pass_at_1 == 1.0
    assert sc.errors == 0


def test_livecodebench_stub_executor_scores_incorrect_solution(tmp_path):
    """The built-in stub emits `__STUB__`, which must not match fixture output."""
    sc = cc.run_suite(
        suite="livecodebench-v6",
        model_name="local:qwen3.6-35b",
        output_dir=tmp_path,
        dry_run=True,
    )
    assert sc.n_problems == 2
    assert sc.n_passed == 0


def test_cli_dryrun_swebench(tmp_path, capsys):
    """End-to-end CLI: match the bead verification command and output dir."""
    rc = cc.main(
        [
            "--model=local:qwen3.6-35b",
            "--suite=swe-bench-lite",
            "--dry-run",
            f"--output={tmp_path}",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "pass@1 = 0/2" in captured
    summary = json.loads((tmp_path / "code_correctness_summary.json").read_text())
    assert any(s["suite"] == "swe-bench-lite" for s in summary)


def test_cli_rejects_missing_model():
    with pytest.raises(SystemExit):
        cc.main(["--suite=livecodebench-v6", "--dry-run"])


def test_swebench_non_dryrun_raises_not_implemented():
    from benchmarks.suites import swe_bench_lite

    with pytest.raises(NotImplementedError, match="Sylveste-r8g"):
        swe_bench_lite.load_problems(dry_run=False)
