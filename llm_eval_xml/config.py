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
N_SAMPLES: int = 2          # How many items to evaluate per dataset per checkpoint
MAX_WORKERS: int = 5        # Parallel threads for LLM calls
RANDOM_SEED: int = 42       # Fixed seed for reproducible sampling

# Fraction of N_SAMPLES drawn from correct items (0.0–1.0).
# Set to None to disable stratified sampling (pure random draw).
SAMPLE_CORRECT_RATIO: float | None = 1.0

# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------
JUDGE_MODEL: str = "openai/gpt-oss-120b"
# JUDGE_MODEL = "gpt-4o"
# JUDGE_MODEL = "Qwen/Qwen3-235B-A22B"

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.hyperbolic.xyz/v1")

API_TIMEOUT: float = 60.0   # seconds
API_MAX_RETRIES: int = 2

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
# You can disable individual metrics here by name.
# ---------------------------------------------------------------------------
DISABLED_METRICS = ["backtracking", "branchiness"]
# DISABLED_METRICS: list[str] = []
# Example: DISABLED_METRICS = ["backtracking", "self_verification"]
