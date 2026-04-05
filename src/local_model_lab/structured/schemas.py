"""Pydantic schemas for structured output experiments.

Each schema represents a real-world extraction task. Models are prompted to
respond with valid JSON matching the schema's JSON Schema definition.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class SentimentResult(BaseModel):
    """Sentiment classification with confidence and brief reasoning."""

    label: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0–1")
    reasoning: str = Field(description="One sentence explaining the classification")

    # Paired prompt for experiments
    PROMPT: ClassVar[str] = (
        "Analyse the sentiment of this review and respond with a JSON object: "
        '"The new coffee shop downtown has great ambiance but the service was painfully slow '
        'and the prices are way too high for what you get."'
    )


# ---------------------------------------------------------------------------
# Named Entity Recognition
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    person = "person"
    organization = "organization"
    location = "location"
    date = "date"
    product = "product"
    other = "other"


class Entity(BaseModel):
    text: str = Field(description="Exact span from the source text")
    type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("type", mode="before")
    @classmethod
    def normalise_entity_type(cls, v: Any) -> Any:
        """Lowercase and map unknown types to 'other'.

        Models invent types like 'event', 'product_name', 'ORGANIZATION'.
        Lowercasing handles casing; unknown values fall back to 'other' so
        the whole entity isn't rejected over one imaginative label.
        """
        if isinstance(v, str):
            v = v.lower()
            valid = {e.value for e in EntityType}
            return v if v in valid else "other"
        return v


class EntityList(BaseModel):
    """Named entities extracted from a piece of text."""

    entities: list[Entity] = Field(description="All entities found in the text")

    PROMPT: ClassVar[str] = (
        "Extract all named entities from this text and respond with a JSON object: "
        '"Apple CEO Tim Cook announced at the Worldwide Developers Conference in San Francisco '
        'on June 10 that the new M4 MacBook Pro will ship in November 2024."'
    )


# ---------------------------------------------------------------------------
# Code Review
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class CodeIssue(BaseModel):
    severity: Severity
    description: str
    line_hint: str | None = Field(None, description="Approximate location, e.g. 'line 3'")

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v: Any) -> Any:
        """Accept 'High', 'HIGH', 'high' — models rarely get the casing right."""
        return v.lower() if isinstance(v, str) else v


class CodeReview(BaseModel):
    """Structured code review with issues, suggestions, and an overall score."""

    issues: list[CodeIssue] = Field(description="List of problems found")
    suggestions: list[str] = Field(description="Improvement suggestions")
    overall_quality: int = Field(ge=1, le=10, description="Quality score 1–10")
    summary: str = Field(description="One sentence overall assessment")

    @field_validator("overall_quality", mode="before")
    @classmethod
    def coerce_quality_to_int(cls, v: Any) -> Any:
        """Coerce string scores and clamp out-of-range values.

        Models output '7' (string) or 0 (below ge=1 minimum). Normalise both
        so a structurally correct response isn't rejected on a trivial type issue.
        """
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                return v
        if isinstance(v, int) and v < 1:
            return 1
        return v

    PROMPT: ClassVar[str] = (
        "Review this Python function for bugs and security issues. "
        "Your JSON response must contain ALL FOUR of these fields:\n"
        "- issues: array of objects, each with severity (high/medium/low), description, line_hint\n"
        "- suggestions: array of improvement strings\n"
        "- overall_quality: integer from 1 to 10\n"
        "- summary: one sentence assessment\n\n"
        "Function to review:\n"
        "def get_user(id):\n"
        "    query = 'SELECT * FROM users WHERE id = ' + str(id)\n"
        "    result = db.execute(query)\n"
        "    return result[0]"
    )


# ---------------------------------------------------------------------------
# Structured Summary
# ---------------------------------------------------------------------------

class StructuredSummary(BaseModel):
    """Structured summary of a passage."""

    title: str = Field(description="A concise title for the content")
    key_points: list[str] = Field(min_length=1, description="2–5 key points")
    tone: str = Field(description="Overall tone, e.g. 'informative', 'critical', 'optimistic'")
    estimated_word_count: int = Field(ge=0, description="Approximate word count of the original")

    PROMPT: ClassVar[str] = (
        "Summarise this paragraph as a JSON object: "
        '"Renewable energy adoption has accelerated dramatically over the past decade. '
        "Solar panel costs have dropped by 90% since 2010, making rooftop installations "
        "accessible to ordinary homeowners. Wind energy now supplies over 10% of global "
        "electricity. However, grid-scale battery storage remains expensive, creating "
        'intermittency challenges that engineers are racing to solve."'
    )


# ---------------------------------------------------------------------------
# Registry — maps schema name → (class, prompt)
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "sentiment": SentimentResult,
    "entities": EntityList,
    "code_review": CodeReview,
    "summary": StructuredSummary,
}
