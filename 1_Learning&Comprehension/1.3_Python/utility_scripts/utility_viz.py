from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hist_distribution(
    df: pd.DataFrame,
    feature: str,
    figsize: tuple[int, int] = (9, 6),
    bins: int = 50,
    color: str = "teal",
    plot_central_ten: bool = True,
    log: bool = False,
    scientific: bool = False,
    grid: bool = False,
    zoom_x: bool = False,
    zoom_x_lim: tuple[None] = (None, None),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    if log and scientific:
        raise ValueError("You have set both scientific and log as True. Change.")

    if zoom_x and zoom_x_lim[0] is None:
        raise ValueError("zoom_x is set but xlim not given.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[feature].values, bins=bins, color=color, log=log)
    if plot_central_ten:
        central_vals = {
            "Mean": df[feature].mean(),
            "Median": df[feature].median(),
            "Mode": df[feature].mode()[0],
        }
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
        for _, value in zip(color, central_vals):
            ax.axvline(
                x=central_vals[value],
                label=value,
                linestyle="dashed",
                color=next(color),
                linewidth=1.5,
            )

    if not scientific:
        ax.ticklabel_format(useOffset=False, style="plain")
    if title is None:
        title = f"Distribution of {feature}"
    if xlabel is None:
        xlabel = f"{feature}"
    if ylabel is None:
        ylabel = f"Freq. of {feature}"

    ax.legend()
    ax.grid(grid)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if zoom_x:
        ax.set_xlim(zoom_x_lim)

    return fig, ax