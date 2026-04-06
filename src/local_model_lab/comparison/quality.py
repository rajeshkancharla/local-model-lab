"""Heuristic quality scoring for model responses.

Scores on 4 dimensions, each 0–5, giving a maximum of 20 points.
No LLM-as-judge — all scoring is deterministic and offline-safe.

Dimensions
----------
relevance        0–5  Keyword coverage: how many expected domain terms appear.
completeness     0–5  Response length and structural depth vs. min_length.
format_compliance 0–5  Adherence to requested output format (JSON, code, text).
coherence        0–5  Readability: no repetition loops, no refusals, no truncation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from local_model_lab.benchmarks.prompts import Prompt


@dataclass
class QualityScore:
    relevance: int          # 0-5
    completeness: int       # 0-5
    format_compliance: int  # 0-5
    coherence: int          # 0-5

    @property
    def total(self) -> int:
        return self.relevance + self.completeness + self.format_compliance + self.coherence


ZERO_SCORE = QualityScore(0, 0, 0, 0)


def score_response(prompt: Prompt, response: str) -> QualityScore:
    """Score a model response against the given prompt heuristically."""
    if not response or response.startswith("[TIMED OUT"):
        return ZERO_SCORE
    return QualityScore(
        relevance=_score_relevance(prompt.expected_keywords, response),
        completeness=_score_completeness(response, prompt.min_length, prompt.category),
        format_compliance=_score_format(response, prompt.category, prompt.expected_format),
        coherence=_score_coherence(response),
    )


# ── Individual dimension scorers ──────────────────────────────────────────────


def _score_relevance(expected_keywords: list[str], response: str) -> int:
    """Keyword coverage: proportion of expected terms found in the response."""
    if not expected_keywords:
        return 3  # neutral — no keywords defined for this prompt
    response_lower = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return min(5, round(hits / len(expected_keywords) * 5))


def _score_completeness(response: str, min_length: int, category: str) -> int:
    """Length and structural depth relative to expected minimum length."""
    text = response.strip()
    length = len(text)
    if length < 20:
        return 0
    if length < min_length // 3:
        return 1
    if length < (min_length * 2) // 3:
        return 2
    # Multi-step prompts reward numbered/bulleted structure
    if category == "multi_step":
        has_list = bool(re.search(r"(\n\s*\d+\.|\n\s*[-*•]|\bstep\s+\d)", text, re.IGNORECASE))
        if length >= min_length * 1.5 and has_list:
            return 5
        if length >= min_length and has_list:
            return 4
        return 3
    if length >= min_length * 2:
        return 5
    if length >= min_length:
        return 4
    return 3


def _score_format(response: str, category: str, expected_format: str) -> int:
    """Adherence to the requested output format."""
    stripped = response.strip()

    if category == "structured_output" or expected_format == "json":
        return _score_json(stripped)

    if category == "code_generation" or expected_format == "code":
        return _score_code(response)

    # Text response: penalise unexpected JSON for a plain-text prompt
    if stripped.startswith("{") and stripped.endswith("}"):
        return 2
    return 4  # plain text is appropriate for everything else


def _score_json(response: str) -> int:
    """Score JSON output quality (0/2/4/5)."""
    # Ideal: whole response is valid JSON
    try:
        json.loads(response)
        return 5
    except (json.JSONDecodeError, ValueError):
        pass
    # Acceptable: valid JSON embedded in prose
    match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", response)
    if match:
        try:
            json.loads(match.group())
            return 4
        except (json.JSONDecodeError, ValueError):
            return 2  # JSON-like but malformed
    return 0  # no JSON found


def _score_code(response: str) -> int:
    """Score code output quality (1/3/5)."""
    if re.search(r"```[\s\S]*?```", response):
        return 5  # proper fenced code block
    if re.search(r"\bdef \w+\(|\bclass \w+:|\bfunction\s+\w+\(", response):
        return 3  # contains code but no fence
    return 1  # no recognisable code structure


def _score_coherence(response: str) -> int:
    """Readability: penalise refusals, repetition loops, and truncation."""
    stripped = response.strip()
    if not stripped:
        return 0

    lower = stripped.lower()

    # Detect outright refusals in the opening
    refusals = ("i cannot", "i'm unable to", "i am unable to", "i don't have the ability")
    if any(r in lower[:200] for r in refusals):
        return 1

    # Detect repetition loops (same sentence duplicated > 50 % of the time)
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", stripped) if len(s.strip()) > 15]
    if len(sentences) >= 4:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.5:
            return 1

    # Detect potential truncation: long response ending without terminal punctuation
    last_char = stripped[-1] if stripped else ""
    if len(stripped) > 150 and last_char not in ".!?:)'\"]>-_":
        return 3

    if len(sentences) >= 3:
        return 5
    if len(sentences) >= 2:
        return 4
    return 3
