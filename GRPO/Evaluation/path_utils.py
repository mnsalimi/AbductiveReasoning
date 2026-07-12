"""
Path utilities for handling project-relative paths.
This module provides functions to get project root and construct relative paths.
"""
import os
import json
from pathlib import Path


EVALUATION_SUBSET_VERSION = "source_prefix_no_length_filter_v1"


def is_evaluation_cache_current(path, max_samples, split):
    """Return whether a cached result matches the current fixed evaluation subset."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    if cached.get("evaluation_subset_version") != EVALUATION_SUBSET_VERSION:
        return False
    if cached.get("max_samples") != max_samples or cached.get("split") != split:
        return False
    return max_samples is None or cached.get("total") == max_samples


def get_project_root():
    """
    Find the project root directory by looking for common markers.
    Returns the AbductiveReasoning directory.
    """
    # Start from this file's directory
    current = Path(__file__).resolve().parent
    
    # Go up until we find the project root (AbductiveReasoning directory)
    # Look for markers like GRPO/, datasets/, requirements.txt
    while current.parent != current:
        if (current / "GRPO").exists() and (current / "datasets").exists():
            return str(current)
        current = current.parent
    
    # Fallback: assume we're in AbductiveReasoning/GRPO/Evaluation
    # Go up 2 levels
    return str(Path(__file__).resolve().parent.parent.parent)


def get_datasets_dir():
    """Get the datasets directory path relative to project root."""
    return os.path.join(get_project_root(), "datasets")


def get_evaluation_dir():
    """Get the Evaluation directory path."""
    return str(Path(__file__).resolve().parent)


def get_grpo_dir():
    """Get the GRPO directory path relative to project root."""
    return os.path.join(get_project_root(), "GRPO")


def get_results_dir():
    """Get the GRPO/results directory path relative to project root."""
    return os.path.join(get_project_root(), "GRPO", "results")


def get_relative_path(*path_parts):
    """
    Construct a path relative to project root.
    
    Args:
        *path_parts: Path components to join
        
    Returns:
        Absolute path string
    """
    return os.path.join(get_project_root(), *path_parts)


def expand_user_path(path):
    """
    Expand ~ in paths and resolve to absolute path.
    Also handles paths that might be relative to project root.
    
    Args:
        path: Path string that may contain ~ or be relative
        
    Returns:
        Absolute path string
    """
    # Expand user home directory
    path = os.path.expanduser(path)
    
    # If it's already absolute, return as is
    if os.path.isabs(path):
        return path
    
    # Otherwise, treat as relative to project root
    return os.path.join(get_project_root(), path)
