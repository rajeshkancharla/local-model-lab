"""Tests for prompt loading."""

from local_model_lab.benchmarks.prompts import load_prompts


def test_load_benchmark_quick():
    prompts = load_prompts("benchmark_quick")
    assert len(prompts) == 8
    assert prompts[0].id == "factual_short_01"
    assert prompts[0].category == "factual"
    assert len(prompts[0].text) > 0


def test_prompt_fields():
    prompts = load_prompts("benchmark_quick")
    for p in prompts:
        assert p.id, "Each prompt must have an id"
        assert p.category, "Each prompt must have a category"
        assert p.text, "Each prompt must have text"
