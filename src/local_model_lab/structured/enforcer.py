"""Schema enforcement: inject JSON schema into prompt, extract, validate, retry.

Usage::

    from local_model_lab.structured.enforcer import enforce_schema
    from local_model_lab.structured.schemas import SentimentResult

    result = await enforce_schema(client, "llama3.2:3b", prompt, SentimentResult)
    if result is not None:
        print(result.label, result.confidence)
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from local_model_lab.client import OllamaClient

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)

_SYSTEM_TEMPLATE = """\
You are a JSON API. Your only job is to output a single valid JSON object.

RULES (non-negotiable):
1. Output ONLY the JSON object. Nothing else.
2. No explanations, no prose, no markdown, no code fences.
3. Start your response with {{ and end with }}.
4. Every required field must be present.
5. Obey all enum constraints exactly as listed.

Required JSON schema:
{schema}

Example of correct output format (structure only, not real values):
{example}
"""

_RETRY_SUFFIX = """\


IMPORTANT: Your previous response was rejected. Error: {error}

You must output ONLY a JSON object starting with {{ and ending with }}. \
No other text whatsoever.\
"""


_EXAMPLES: dict[str, str] = {
    "SentimentResult": '{"label": "positive", "confidence": 0.92, "reasoning": "One sentence here."}',
    "EntityList": '{"entities": [{"text": "Apple", "type": "organization", "confidence": 0.95}]}',
    "CodeReview": (
        '{"issues": [{"severity": "high", "description": "SQL injection risk", "line_hint": "line 2"}], '
        '"suggestions": ["Use parameterised queries"], "overall_quality": 3, "summary": "One sentence."}'
    ),
    "StructuredSummary": (
        '{"title": "Short title", "key_points": ["Point one", "Point two"], '
        '"tone": "informative", "estimated_word_count": 120}'
    ),
}


def _build_system_prompt(schema_class: type[BaseModel]) -> str:
    schema_json = json.dumps(schema_class.model_json_schema(), indent=2)
    example = _EXAMPLES.get(schema_class.__name__, '{"field": "value"}')
    return _SYSTEM_TEMPLATE.format(schema=schema_json, example=example)


def _strip_json_comments(text: str) -> str:
    """Remove // line comments from a JSON string.

    Models like phi3.5 occasionally insert inline comments (e.g.
    ``"type": "event" // not in enum``). Standard json.loads rejects these.
    We strip everything from // to end-of-line, but only outside string literals.
    """
    result = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escape_next:
            result.append(ch)
            escape_next = False
            i += 1
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if not in_string and ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            # Skip to end of line
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _find_json_object(text: str, start: int = 0) -> str | None:
    """Brace-depth scan to extract a complete JSON object starting at text[start].

    Correctly handles nested objects and arrays, unlike a naive regex.
    """
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> str | None:
    """Extract a JSON object from model output.

    Handles three common model output patterns:
    1. Raw JSON (ideal case)
    2. ```json ... ``` code fence
    3. JSON embedded in prose — scans for the first { ... } block
    """
    text = text.strip()

    # Strip markdown code fences — grab content between ``` markers
    fence = re.search(r"```(?:json)?\s*", text)
    if fence:
        candidate = _find_json_object(text, fence.end())
        if candidate:
            return candidate

    # If it starts with { try to parse directly (handles trailing prose)
    if text.startswith("{"):
        return _find_json_object(text, 0)

    # Last resort: find the first { anywhere in the text
    brace_pos = text.find("{")
    if brace_pos != -1:
        return _find_json_object(text, brace_pos)

    return None


async def enforce_schema(
    client: OllamaClient,
    model: str,
    prompt: str,
    schema_class: type[T],
    *,
    temperature: float = 0.0,
) -> tuple[T | None, str, int, float]:
    """Attempt to get a valid structured response from the model.

    Injects the schema as a system prompt, calls generate_full, tries to parse
    and validate the response. On first failure, retries once with the
    validation error appended so the model can self-correct.

    Returns:
        (result, raw_response, attempts, latency_ms)
        result is None if both attempts fail.
    """
    system = _build_system_prompt(schema_class)
    user_suffix = "\n\nRespond with ONLY a JSON object. Start your response with {."

    # For qwen3: use the think=False API option rather than the /no_think prompt
    # directive, which proved unreliable. think=False tells Ollama to disable the
    # internal chain-of-thought at the API level — prompt directives are ignored
    # when text follows them. Also raise num_predict so thinking doesn't consume
    # the entire token budget before the JSON answer appears.
    # qwen3 generates very verbose JSON even with thinking disabled — entity lists
    # and key_points arrays routinely exceed 800 tokens. Use 1200 for qwen3.
    # Other models use 800, which handles all 4 schemas comfortably.
    extra_options: dict
    if "qwen3" in model.lower():
        extra_options = {"think": False, "num_predict": 1200}
    else:
        extra_options = {"num_predict": 800}

    start = time.perf_counter()
    attempts = 0
    raw_response = ""
    last_error = ""

    for attempt in range(1, 3):  # max 2 attempts
        attempts = attempt
        if attempt == 1:
            current_prompt = prompt + user_suffix
        else:
            current_prompt = prompt + user_suffix + _RETRY_SUFFIX.format(error=last_error)

        resp = await client.generate_full(
            model, current_prompt, temperature=temperature, system=system,
            extra_options=extra_options,
        )
        raw_response = resp.get("response", "")

        json_str = _extract_json(raw_response)
        if json_str is None:
            last_error = "No JSON object found in the response."
            continue

        try:
            data: Any = json.loads(_strip_json_comments(json_str))
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc}"
            continue

        try:
            validated = schema_class.model_validate(data)
            latency_ms = (time.perf_counter() - start) * 1000
            return validated, raw_response, attempts, round(latency_ms, 2)
        except ValidationError as exc:
            # Keep only the first error message to avoid overflowing the prompt
            first_error = exc.errors()[0]
            last_error = f"field '{'.'.join(str(l) for l in first_error['loc'])}': {first_error['msg']}"

    latency_ms = (time.perf_counter() - start) * 1000
    return None, raw_response, attempts, round(latency_ms, 2)


def parse_enforce_result(
    result: tuple[BaseModel | None, str, int, float],
) -> dict:
    """Convert enforce_schema return value to a plain dict for serialisation."""
    validated, raw, attempts, latency_ms = result
    return {
        "success": validated is not None,
        "result": validated.model_dump() if validated is not None else None,
        "raw_response": raw,
        "attempts": attempts,
        "latency_ms": latency_ms,
    }
