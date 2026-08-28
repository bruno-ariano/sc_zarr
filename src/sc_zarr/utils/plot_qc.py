from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_qc(
    main_obs: pd.DataFrame,
    output_path: str | Path,
    batch_key: str | None = None,
    *,
    max_scatter_cells: int = 50_000,
    dpi: int = 150,
) -> Path:
    """Create cell-QC plots and save them as a PNG.

    Parameters
    ----------
    main_obs
        Cell metadata containing calculated QC metrics.
    output_path
        Location of the output PNG.
    batch_key
        Optional column used to display mitochondrial percentage by batch.
    max_scatter_cells
        Maximum number of cells included in the scatter plot.
    dpi
        Resolution of the saved image.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG.
    """
    required_columns = {
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
    }

    missing_columns = required_columns - set(main_obs.columns)

    if missing_columns:
        raise KeyError(f"Missing required QC columns: {sorted(missing_columns)}")

    if batch_key is not None and batch_key not in main_obs.columns:
        raise KeyError(f"Batch key {batch_key!r} is not present in main_obs.")

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".png":
        raise ValueError(f"output_path must end with '.png': {output_path}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_obs = main_obs.copy()

    plot_obs["log1p_total_counts"] = np.log1p(plot_obs["total_counts"])
    plot_obs["log1p_n_genes"] = np.log1p(plot_obs["n_genes_by_counts"])

    # Sample only for the scatter plot. Histograms and box plots
    # continue to use all cells.
    scatter_obs = plot_obs.sample(
        n=min(len(plot_obs), max_scatter_cells),
        random_state=0,
    )

    if batch_key is None:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 9),
        )

        batch_axis = None

    else:
        fig = plt.figure(figsize=(12, 13))

        grid = fig.add_gridspec(
            nrows=3,
            ncols=2,
            height_ratios=[1, 1, 0.8],
        )

        axes = np.empty((2, 2), dtype=object)
        axes[0, 0] = fig.add_subplot(grid[0, 0])
        axes[0, 1] = fig.add_subplot(grid[0, 1])
        axes[1, 0] = fig.add_subplot(grid[1, 0])
        axes[1, 1] = fig.add_subplot(grid[1, 1])

        # The batch plot spans the complete bottom row.
        batch_axis = fig.add_subplot(grid[2, :])

    sns.histplot(
        data=plot_obs,
        x="log1p_total_counts",
        bins=100,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Library size")
    axes[0, 0].set_xlabel("log1p(total counts)")

    sns.histplot(
        data=plot_obs,
        x="log1p_n_genes",
        bins=100,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Detected genes")
    axes[0, 1].set_xlabel("log1p(number of detected genes)")

    sns.histplot(
        data=plot_obs,
        x="pct_counts_mt",
        bins=100,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Mitochondrial percentage")
    axes[1, 0].set_xlabel("Mitochondrial counts (%)")

    sns.scatterplot(
        data=scatter_obs,
        x="log1p_total_counts",
        y="log1p_n_genes",
        hue="pct_counts_mt",
        palette="viridis",
        s=8,
        linewidth=0,
        alpha=0.7,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Counts, genes, and mitochondrial percentage")
    axes[1, 1].set_xlabel("log1p(total counts)")
    axes[1, 1].set_ylabel("log1p(number of detected genes)")

    if batch_axis is not None and batch_key is not None:
        sns.boxplot(
            data=plot_obs,
            x=batch_key,
            y="pct_counts_mt",
            showfliers=False,
            ax=batch_axis,
        )

        batch_axis.set_title("Mitochondrial percentage by batch")
        batch_axis.set_xlabel(batch_key)
        batch_axis.set_ylabel("Mitochondrial counts (%)")
        batch_axis.tick_params(
            axis="x",
            labelrotation=90,
        )

    fig.suptitle(
        f"Single-cell QC summary ({len(main_obs):,} cells)",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    # Prevent memory accumulation if this is called for many datasets.
    plt.close(fig)

    return output_path
