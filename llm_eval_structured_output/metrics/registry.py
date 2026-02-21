"""
metrics/registry.py
-------------------
Central registry of all metrics.

To add a new metric:
  1. Create its prompt file in prompts/binary/ or prompts/counting/.
  2. Import the prompt strings here.
  3. Instantiate BinaryMetric or CountingMetric.
  4. Add it to METRICS.

The pipeline resolves active metrics at runtime by filtering METRICS against
config.DISABLED_METRICS.
"""

from __future__ import annotations

from metrics.binary import BinaryMetric
from metrics.counting import CountingMetric
from metrics.base import BaseMetric

# ---------------------------------------------------------------------------
# Prompt imports – counting metrics
# ---------------------------------------------------------------------------
from prompts.counting.branchiness import (
    SYSTEM_PROMPT as BRANCH_SYS,
    USER_PROMPT_TEMPLATE as BRANCH_USR,
)
from prompts.counting.backtracking import (
    SYSTEM_PROMPT as BT_SYS,
    USER_PROMPT_TEMPLATE as BT_USR,
)
from prompts.counting.uncertainty_markers import (
    SYSTEM_PROMPT as UM_SYS,
    USER_PROMPT_TEMPLATE as UM_USR,
)

# ---------------------------------------------------------------------------
# Prompt imports – binary metrics
# ---------------------------------------------------------------------------
from prompts.binary.uncertainty_language import (
    SYSTEM_PROMPT as UL_SYS,
    USER_PROMPT_TEMPLATE as UL_USR,
)
from prompts.binary.detail_coverage import (
    SYSTEM_PROMPT as DC_SYS,
    USER_PROMPT_TEMPLATE as DC_USR,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRICS: dict[str, BaseMetric] = {
    # ── Counting metrics ────────────────────────────────────────────────────
    "branchiness": CountingMetric(
        name="branchiness",
        description="Exploring multiple distinct reasoning paths or hypotheses.",
        system_prompt=BRANCH_SYS,
        user_prompt_template=BRANCH_USR,
    ),
    "backtracking": CountingMetric(
        name="backtracking",
        description="Explicit self-correction or revision of a previous reasoning step.",
        system_prompt=BT_SYS,
        user_prompt_template=BT_USR,
    ),
    "uncertainty_markers": CountingMetric(
        name="uncertainty_markers",
        description="Count of individual probabilistic/hedging words and phrases in the reasoning trace.",
        system_prompt=UM_SYS,
        user_prompt_template=UM_USR,
    ),
    # ── Binary metrics ──────────────────────────────────────────────────────
    "uncertainty_language": BinaryMetric(
        name="uncertainty_language",
        description="Use of probabilistic language rather than absolute certainty.",
        system_prompt=UL_SYS,
        user_prompt_template=UL_USR,
    ),
    "detail_coverage": BinaryMetric(
        name="detail_coverage",
        description="Hypothesis accounts for all specific observation details, not just the main event.",
        system_prompt=DC_SYS,
        user_prompt_template=DC_USR,
    ),
}


def get_active_metrics(disabled: list[str] | None = None) -> dict[str, BaseMetric]:
    """Return only the metrics not present in ``disabled``."""
    disabled = disabled or []
    return {name: m for name, m in METRICS.items() if name not in disabled}
