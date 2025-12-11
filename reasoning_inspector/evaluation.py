from openai import OpenAI
import httpx
from pydantic import BaseModel
import json
from pydantic import BaseModel, Field
from llm_helper import ask_llm
import re
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

class Occurrence(BaseModel):
    text: str = Field(..., description="Exact substring found in the text")
    start: int = Field(..., description="Start character index (inclusive)") 
    end: int = Field(..., description="End character index (exclusive)") 


class ReasoningMetrics(BaseModel):
    branchiness: list[Occurrence] = []
    backtrack: list[Occurrence] = []
    self_verify: list[Occurrence] = []


SYSTEM_PROMPT = """
You are a precise text analysis tool.

Analyze a given text and extract explicit markers of the following reasoning behaviors:

1) Branchiness  
   Mark places where the text clearly considers multiple alternatives or cases.
   Examples include explicit case/option enumeration, conditional alternatives,
   and clear use of “or / otherwise / alternatively”.
   Do not mark simple sequential steps unless they introduce alternatives.

2) Backtrack  
   Mark places where the writer explicitly revises, corrects, or reverses a
   previous statement (e.g., “actually”, “wait”, “correction”, “instead”).
   Do not mark mere restatements.

3) Self-verification  
   Mark places where the writer explicitly checks or validates reasoning or
   intermediate results (e.g., plugging back into an equation, “let us verify”,
   sanity or consistency checks).
   Do not mark standard derivation steps unless a check is explicit.

For each occurrence:
- Return the exact substring from the original text.
- Return its character span as [start, end), using 0-based indexing.

Be conservative and do not infer intent.
If no occurrences exist for a category, return an empty list.
Return only data conforming to the output schema.

"""

def extract_reasoning_metrics_llm(text: str, model) -> ReasoningMetrics:
    return ask_llm(SYSTEM_PROMPT, text, model, ReasoningMetrics)

def extract_reasoning_metrics_llm_batch(texts: list[str], model, max_workers=10) -> list[ReasoningMetrics]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_reasoning_metrics_llm, text, model) for text in texts]
        wait(futures, return_when=ALL_COMPLETED)
        
        # Collect results, handling exceptions
        results = []
        for i, future in enumerate(futures):
            # Check if future has exception without raising it
            exc = future.exception()
            if exc is not None:
                print(f"Error processing text {i}: {exc}")
                # Return empty metrics for failed items
                results.append(exc)
            else:
                results.append(future.result())
        
        return results

BRANCH_CASE = re.compile(
    r"\b(case|option|scenario)\s*(\d+|one|two|three|four|[a-z])\b",
    flags=re.IGNORECASE,
)
BRANCH_ORDER = re.compile(
    r"\b(first|second|third|fourth|finally|next|then)\b",
    flags=re.IGNORECASE,
)
BRANCH_IF_THEN = re.compile(
    r"\bif\b.*?\bthen\b",
    flags=re.IGNORECASE | re.DOTALL,
)
BRANCH_ALT = re.compile(
    r"\b(otherwise|else|alternatively|another|either|neither)\b",
    flags=re.IGNORECASE,
)

BACKTRACK_CORR = re.compile(
    r"\b(actually|correction|i was wrong|that was incorrect)\b",
    flags=re.IGNORECASE,
)
BACKTRACK_WAIT = re.compile(
    r"\b(wait|hold on|let me reconsider)\b",
    flags=re.IGNORECASE,
)
BACKTRACK_REROUTE = re.compile(
    r"\b(instead|rather than|on second thought)\b",
    flags=re.IGNORECASE,
)

VERIFY_VERB = re.compile(
    r"\b(check|verify|confirm|validate|double[- ]?check)\b",
    flags=re.IGNORECASE,
)
VERIFY_CAUTION = re.compile(
    r"\b(to be sure|to make sure|ensure that)\b",
    flags=re.IGNORECASE,
)
VERIFY_CONSIST = re.compile(
    r"\b(does this match|is this consistent|this agrees with)\b",
    flags=re.IGNORECASE,
)

CODE_BLOCK = re.compile(r"```.*?```", flags=re.DOTALL)


def _strip_code_blocks(text: str) -> str:
    """Remove markdown code blocks to avoid noisy matches."""
    return re.sub(CODE_BLOCK, "", text)

def _find_occurrences(pattern: re.Pattern, text: str) -> list[Occurrence]:
    """
    Find all occurrences of a regex pattern in text and return as Occurrence objects.
    """
    occurrences = []
    for match in pattern.finditer(text):
        occurrences.append(Occurrence(
            text=match.group(),
            start=match.start(),
            end=match.end()
        ))
    return occurrences

def extract_reasoning_metrics_regex(text: str) -> ReasoningMetrics:
    """
    Compute heuristic behavioral metrics from a single completion:

    - branchiness: signs of exploring multiple cases/paths
    - backtrack: signs of correcting / reversing earlier statements
    - self_verify: signs of explicitly checking or verifying

    Returns ReasoningMetrics with actual occurrence details.
    """
    clean = _strip_code_blocks(text)

    # Collect all branchiness occurrences
    branchiness_occurrences = []
    branchiness_occurrences.extend(_find_occurrences(BRANCH_CASE, clean))
    branchiness_occurrences.extend(_find_occurrences(BRANCH_ORDER, clean))
    branchiness_occurrences.extend(_find_occurrences(BRANCH_IF_THEN, clean))
    branchiness_occurrences.extend(_find_occurrences(BRANCH_ALT, clean))

    # Collect all backtrack occurrences
    backtrack_occurrences = []
    backtrack_occurrences.extend(_find_occurrences(BACKTRACK_CORR, clean))
    backtrack_occurrences.extend(_find_occurrences(BACKTRACK_WAIT, clean))
    backtrack_occurrences.extend(_find_occurrences(BACKTRACK_REROUTE, clean))

    # Collect all self-verify occurrences
    self_verify_occurrences = []
    self_verify_occurrences.extend(_find_occurrences(VERIFY_VERB, clean))
    self_verify_occurrences.extend(_find_occurrences(VERIFY_CAUTION, clean))
    self_verify_occurrences.extend(_find_occurrences(VERIFY_CONSIST, clean))

    return ReasoningMetrics(
        branchiness=branchiness_occurrences,
        backtrack=backtrack_occurrences,
        self_verify=self_verify_occurrences
    )
    




