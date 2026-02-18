"""
llm_client.py
-------------
OpenAI API wrapper with:
  - Per-request structured logging to JSONL files
  - In-memory caching to avoid redundant API calls across restarts
  - Clean JSON-mode responses (no regex parsing)
  - Support for OSS models that separate thinking vs. final content
"""

from __future__ import annotations

import datetime
import json
import os
import re
import threading
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

import config

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_client: OpenAI | None = None
_lock = threading.Lock()

# Cache: request_key -> parsed dict (populated from disk + live calls)
_response_cache: dict[tuple, dict] = {}
# Set of log-file paths already scanned into the cache
_loaded_logs: set[str] = set()
# Set of log-file paths already cleared this run
_cleared_logs: set[str] = set()


def get_client() -> OpenAI:
    """Return (and lazily initialise) the shared OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.API_TIMEOUT,
            max_retries=0,  # We handle retries ourselves
        )
    return _client


def test_connection() -> bool:
    """Ping the API with a minimal request; return True if reachable."""
    print("\n[Testing API connection …]")
    try:
        get_client().chat.completions.create(
            model=config.JUDGE_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            temperature=0.0,
        )
        print(f"[OK] Connected – model '{config.JUDGE_MODEL}' is accessible.")
        return True
    except Exception as exc:
        print(f"[ERROR] API test failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "N/A"


def _split_oss_channels(text: str | None) -> tuple[str | None, str | None]:
    """
    Some OSS models embed thinking content and final content in a single string
    separated by a special token.  Split them if present.
    """
    separator = "<|channel|>final<|message|>"
    if isinstance(text, str) and separator in text:
        thinking, final = text.split(separator, 1)
        return thinking.strip(), final.strip()
    return None, text


def _cache_key(
    run_id: str,
    checkpoint: str,
    dataset: str,
    problem_id: str,
    metric: str,
    model: str,
) -> tuple:
    return (_safe(run_id), _safe(checkpoint), _safe(dataset), _safe(problem_id), _safe(metric), _safe(model))


def _log_path(dataset: str) -> str:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    return os.path.join(config.LOG_DIR, f"{dataset}_llm_responses.jsonl")


def _load_cache_from_disk(log_path: str) -> None:
    """Populate the in-memory cache from an existing JSONL log file (once per path)."""
    if log_path in _loaded_logs:
        return
    _loaded_logs.add(log_path)

    if not os.path.exists(log_path):
        return

    try:
        with open(log_path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if obj.get("parse_status", "").startswith("success") and isinstance(obj.get("parsed_data"), dict):
                    key = _cache_key(
                        obj.get("run_id", ""),
                        obj.get("checkpoint", ""),
                        obj.get("dataset", ""),
                        obj.get("problem_id", ""),
                        obj.get("metric_type", ""),
                        obj.get("model", ""),
                    )
                    _response_cache[key] = obj["parsed_data"]
    except Exception:
        pass  # Corrupt log – skip silently


def _append_log(log_path: str, entry: dict) -> None:
    with _lock:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    response_schema: type[BaseModel],
    dataset: str = "unknown",
    problem_id: str = "N/A",
    metric_type: str = "unknown",
    checkpoint: str = "N/A",
    run_id: str | None = None,
) -> dict:
    """
    Call the judge LLM and return a plain dict matching `response_schema`.

    The response is requested in JSON mode and validated against the Pydantic
    schema.  All calls are logged to JSONL and cached so that restarting the
    script does not re-bill completed items.

    Returns a dict on success.  Returns the schema's ``default()`` values on
    any irrecoverable failure so the pipeline can continue.
    """
    if run_id is None:
        run_id = config.RUN_ID

    model = config.JUDGE_MODEL
    log_path = _log_path(dataset)

    # Optional per-run log reset
    if config.CLEAR_PREVIOUS_OUTPUTS and log_path not in _cleared_logs:
        try:
            os.remove(log_path)
        except FileNotFoundError:
            pass
        _cleared_logs.add(log_path)

    _load_cache_from_disk(log_path)

    key = _cache_key(run_id, checkpoint, dataset, problem_id, metric_type, model)
    if key in _response_cache:
        return _response_cache[key]

    # More retries for large OSS models that are prone to transient failures
    max_tries = config.API_MAX_RETRIES if any(
        t in model.lower() for t in ("oss", "deepseek", "qwen")
    ) else 1

    last_error: str = ""
    for attempt in range(max_tries):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        try:
            response = get_client().chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            msg = response.choices[0].message
            raw_content: str | None = getattr(msg, "content", None)
            raw_reasoning: str | None = getattr(msg, "reasoning_content", None)

            oss_thinking, final_content = _split_oss_channels(raw_content)
            thinking = raw_reasoning or oss_thinking

            if not final_content:
                _append_log(log_path, {
                    "timestamp": ts, "run_id": run_id, "checkpoint": checkpoint,
                    "dataset": dataset, "problem_id": problem_id,
                    "metric_type": metric_type, "model": model,
                    "parse_status": "empty_response", "error_message": "Empty response",
                    "raw_response": None, "raw_reasoning": thinking,
                })
                return _default_payload(response_schema)

            # Parse XML returned by the model
            data = _parse_xml_response(final_content, response_schema)

            # Validate against schema
            validated = response_schema.model_validate(data).model_dump()

            _append_log(log_path, {
                "timestamp": ts, "run_id": run_id, "checkpoint": checkpoint,
                "dataset": dataset, "problem_id": problem_id,
                "metric_type": metric_type, "model": model,
                "parse_status": "success",
                "raw_response": final_content,
                "raw_reasoning": thinking,
                "parsed_data": validated,
            })
            _response_cache[key] = validated
            return validated

        except (ValueError, KeyError) as parse_err:
            last_error = str(parse_err)
            if attempt < max_tries - 1:
                print(f"  [LLM] JSON parse failed (attempt {attempt + 1}): {last_error[:120]} – retrying …")
                continue

        except Exception as api_err:
            last_error = str(api_err)
            err_lower = last_error.lower()
            if attempt < max_tries - 1:
                print(f"  [LLM] API error (attempt {attempt + 1}): {last_error[:120]} – retrying …")
                continue
            # Surface useful hints
            if "timeout" in err_lower:
                print("  [HINT] Increase API_TIMEOUT in config.py or check network.")
            elif "not found" in err_lower or "404" in err_lower:
                print(f"  [HINT] Model '{model}' not found – check JUDGE_MODEL in config.py.")

    _append_log(log_path, {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "run_id": run_id, "checkpoint": checkpoint,
        "dataset": dataset, "problem_id": problem_id,
        "metric_type": metric_type, "model": model,
        "parse_status": "failed", "error_message": last_error,
        "raw_response": None, "raw_reasoning": None,
    })
    print(f"  [LLM] All attempts failed for {metric_type}/{problem_id}: {last_error[:200]}")
    return _default_payload(response_schema)


def _parse_xml_response(text: str, schema: type[BaseModel]) -> dict:
    """
    Extract fields from an XML-formatted LLM response.

    Supports two response shapes:

    Binary  → <detected>, <reasoning>, <evidence>
    Counting → <analysis>, <matches> containing <match> blocks
                with <excerpt> and <explanation> children.
    """
    def _tag(tag: str, src: str) -> str:
        """Return the inner text of the first occurrence of <tag>…</tag>."""
        m = re.search(rf"<{tag}>(.*?)</{tag}>", src, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    fields = {f for f in schema.model_fields}

    # ---- Binary shape -------------------------------------------------------
    if "detected" in fields:
        raw_detected = _tag("detected", text).lower()
        detected = raw_detected in ("true", "1", "yes")
        return {
            "detected": detected,
            "reasoning": _tag("reasoning", text),
            "evidence": _tag("evidence", text),
        }

    # ---- Counting shape -----------------------------------------------------
    analysis = _tag("analysis", text)
    examples: list[dict] = []
    matches_block_m = re.search(r"<matches>(.*?)</matches>", text, re.DOTALL | re.IGNORECASE)
    if matches_block_m:
        matches_block = matches_block_m.group(1)
        for match in re.finditer(r"<match>(.*?)</match>", matches_block, re.DOTALL | re.IGNORECASE):
            block = match.group(1)
            examples.append({
                "excerpt": _tag("excerpt", block),
                "explanation": _tag("explanation", block),
            })
    return {
        "overall_analysis": analysis,
        "examples": examples,
    }


def _default_payload(schema: type[BaseModel]) -> dict:
    """Return a zeroed-out dict for the given Pydantic schema."""
    defaults: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        ann = field.annotation
        if ann is bool:
            defaults[name] = False
        elif ann is str or (hasattr(ann, "__origin__") is False and ann is str):
            defaults[name] = ""
        elif ann is int:
            defaults[name] = 0
        elif hasattr(ann, "__origin__") and ann.__origin__ is list:
            defaults[name] = []
        else:
            defaults[name] = None
    try:
        return schema(**defaults).model_dump()
    except Exception:
        return defaults
