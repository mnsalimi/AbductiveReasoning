import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_grouped_bars(
    data: list[dict],
    labels: Optional[list[str]] = None,
    title: str = "Grouped Bar Chart",
    xlabel: str = "Keys",
    ylabel: str = "Values",
    figsize: tuple = (12, 6),
    rotation: int = 45,
    show_values: bool = False
):
    if not data:
        raise ValueError("Data list cannot be empty")
    
    # Get common keys (assuming all dicts have same keys)
    keys = list(data[0].keys())
    num_groups = len(keys)
    num_bars = len(data)
    
    # Create labels if not provided
    if labels is None:
        labels = [f"Dict {i}" for i in range(num_bars)]
    
    # Set up the bar positions
    x = np.arange(num_groups)
    width = 0.8 / num_bars  # Width of each bar
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars for each dictionary
    for i, (d, label) in enumerate(zip(data, labels)):
        values = [d[key] for key in keys]
        offset = (i - num_bars / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width, label=label)
        
        # Add value labels on bars if requested
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
    
    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=rotation, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig, ax



