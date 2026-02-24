"""
config.py
---------
All runtime settings for the LLM evaluation pipeline.
Edit this file to change models, sampling, API credentials, and which metrics to run.
"""

import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
RUN_ID: str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
N_SAMPLES: int = 3          # How many items to evaluate per dataset per checkpoint
MAX_WORKERS: int = 5        # Parallel threads for LLM calls
RANDOM_SEED: int = 42       # Fixed seed for reproducible sampling

# Fraction of N_SAMPLES drawn from correct items (0.0–1.0).
# Set to None to disable stratified sampling (pure random draw).
SAMPLE_CORRECT_RATIO: float | None = None

# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------
JUDGE_MODEL: str = "gpt-5-nano"
#JUDGE_MODEL: str = "openai/gpt-oss-120b"
# JUDGE_MODEL = "gpt-4o"
# JUDGE_MODEL = "Qwen/Qwen3-235B-A22B"

# Reasoning effort for GPT-5+ models ("low" | "medium" | "high").
# Ignored for older models that do not support this parameter.
REASONING_EFFORT: str = "low"

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.hyperbolic.xyz/v1")

API_TIMEOUT: float = 60.0   # seconds
API_MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_OUTPUT_DIR: str = "results"
LOG_DIR: str = os.path.join(BASE_OUTPUT_DIR, "llm_logs")
UNNORM_DIR: str = os.path.join(BASE_OUTPUT_DIR, "unnormalized")
NORM_DIR: str = os.path.join(BASE_OUTPUT_DIR, "normalized")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# If True, deletes existing JSONL log files at the start of the run so that
# all LLM calls are re-executed (useful when prompts change).
CLEAR_PREVIOUS_OUTPUTS: bool = False

# ---------------------------------------------------------------------------
# Metrics to run
# Controlled by the metric registry (metrics/registry.py).
# List only the metric names you want to evaluate.
# Available metrics: "backtracking", "branchiness", "uncertainty_markers",
#                    "uncertainty_language", "detail_coverage",
#                    "observation_coverage"
# An empty list activates ALL registered metrics.
# ---------------------------------------------------------------------------
ACTIVE_METRICS: list[str] = ["prior", "uncertainty_markers", "observation_coverage"]
# Example: ACTIVE_METRICS = ["uncertainty_markers", "backtracking"]
# Example (all metrics): ACTIVE_METRICS = []

# ---------------------------------------------------------------------------
# Datasets to evaluate
# List only the dataset names (folder names inside each checkpoint) to include.
# An empty list evaluates ALL datasets found in each checkpoint.
# ---------------------------------------------------------------------------
ACTIVE_DATASETS: list[str] = ["medqa", "copa_guess_effect"]
# Example (all datasets): ACTIVE_DATASETS = []

# ---------------------------------------------------------------------------
# Excluded checkpoints
# List checkpoint directory names (basenames) to skip entirely.
# Supports both trained checkpoints ("checkpoint-500") and "raw_model".
# An empty list means no checkpoints are excluded.
# ---------------------------------------------------------------------------
EXCLUDED_CHECKPOINTS: list[str] = []
# Example: EXCLUDED_CHECKPOINTS = ["checkpoint-500", "raw_model"]
