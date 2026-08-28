from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import zarr
from loguru import logger

from sc_zarr.utils.plot_qc import plot_qc
from sc_zarr.utils.zarr_stats import (
    # add_dataframe_columns,
    add_qc_gene_sets,
    choose_qc_matrix,
)

# ============================================================
# CONFIG
# ============================================================
ad.settings.allow_write_nullable_strings = True


@dataclass()
class ScZarrRunQc:
    """Parameters for a QC run on a Zarr AnnData store.

    Examples
    --------
    >>> params = ScZarrRunQc(
    ...     zarr_path=Path("data.zarr"),
    ...     batch_key=None,
    ...     covariate_keys=[],
    ... )
    >>> params.qc_source
    'auto'
    """

    zarr_path: Path
    batch_key: str | None
    covariate_keys: list[str]
    mad_n: float = 3
    mad_min_group_size: int | None = 100
    qc_source: str | None = "auto"
    gene_name_column: str | None = None
    output_dir: Path | None = None


def resolve_output_dir(sczarr_run_qc: ScZarrRunQc) -> Path:
    """Return the directory that receives QC outputs, creating it if needed.

    Defaults to a ``<zarr name>_qc`` folder next to the Zarr store so the
    store itself is never polluted with report files.

    Examples
    --------
    >>> params = ScZarrRunQc(Path("/data/pbmc.zarr"), None, [])
    >>> resolve_output_dir(params)
    PosixPath('/data/pbmc_qc')
    """
    if sczarr_run_qc.output_dir is not None:
        output_dir = Path(sczarr_run_qc.output_dir)
    else:
        zarr_path = Path(sczarr_run_qc.zarr_path)
        output_dir = zarr_path.parent / f"{zarr_path.stem}_qc"

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def attach_grouping_column(
    obs_qc: pd.DataFrame,
    main_obs: pd.DataFrame,
    batch_key: str | None,
) -> pd.DataFrame:
    """Copy a grouping column from the original obs table onto the QC metrics.

    Examples
    --------
    >>> obs_qc = pd.DataFrame({"total_counts": [10, 20]}, index=["c1", "c2"])
    >>> main_obs = pd.DataFrame({"batch": ["a", "b"]}, index=["c1", "c2"])
    >>> attach_grouping_column(obs_qc, main_obs, "batch")["batch"].tolist()
    ['a', 'b']
    """
    if batch_key is None:
        return obs_qc

    if batch_key not in main_obs.columns:
        available = ", ".join(map(str, main_obs.columns))
        raise KeyError(
            f"Batch key {batch_key!r} is not a column of obs. Available: {available}"
        )

    plot_obs = obs_qc.copy()
    plot_obs[batch_key] = main_obs[batch_key].reindex(plot_obs.index)

    return plot_obs


def zarr_qc_plot(sczarr_run_qc: ScZarrRunQc) -> Path:
    """Calculate cell and gene QC metrics and plot them as a PNG.

    No cell is filtered here. The metric tables are written next to the PNG so
    the later filtering step can reuse them without recomputing.

    Returns
    -------
    pathlib.Path
        Path of the saved QC summary PNG.

    Examples
    --------
    >>> zarr_qc_plot(ScZarrRunQc(zarr_path=Path("data.zarr"), batch_key=None, covariate_keys=[]))
    PosixPath('data_qc/qc_plots.png')
    """
    logger.info("\n# QC\n")
    logger.info(f"Zarr: {sczarr_run_qc.zarr_path}")

    # --------------------------------------------------------
    # Open Zarr
    # --------------------------------------------------------

    root = zarr.open_group(
        sczarr_run_qc.zarr_path,
        mode="r",
    )

    # obs and var are small metadata tables that I can read in memory.
    main_obs = ad.io.read_elem(root["obs"])

    main_var = ad.io.read_elem(root["var"])

    logger.info(f"Main AnnData: {len(main_obs):,} cells x {len(main_var):,} genes")

    # --------------------------------------------------------
    # Select counts matrix
    # --------------------------------------------------------
    # We can choose the expressionmatrix to use for QC and its associated var table based on the user's input.
    X_qc, var_qc = choose_qc_matrix(
        root,
        main_var,
        sczarr_run_qc.qc_source,
    )

    logger.info("\nLazy QC matrix:")
    logger.info(X_qc)

    logger.info("\nChunks:")
    logger.info(X_qc.chunks)

    logger.info("\nSparse chunk type:")
    logger.info(type(X_qc._meta))

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if X_qc.shape[0] != len(main_obs):
        raise ValueError(
            "QC matrix cell count does not match obs: "
            f"{X_qc.shape[0]} != {len(main_obs)}"
        )

    if X_qc.shape[1] != len(var_qc):
        raise ValueError(
            "QC matrix gene count does not match its var: "
            f"{X_qc.shape[1]} != {len(var_qc)}"
        )

    # --------------------------------------------------------
    # Gene categories
    # --------------------------------------------------------
    logger.info("\nAdding QC gene sets...")
    var_qc = add_qc_gene_sets(
        var_qc.copy(), gene_name_column=sczarr_run_qc.gene_name_column
    )

    # --------------------------------------------------------
    # Temporary AnnData used only for QC
    #
    # X remains lazy / Dask-backed.
    # --------------------------------------------------------
    logger.info("\nCreating temporary AnnData for QC...")
    adata_qc = ad.AnnData(
        X=X_qc,
        obs=main_obs.copy(),
        var=var_qc,
    )

    logger.info("\nQC AnnData:")
    logger.info(adata_qc)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    logger.info("\nCalculating QC metrics...")

    obs_qc, var_qc_results = sc.pp.calculate_qc_metrics(
        adata_qc,
        qc_vars=[
            "mt",
            "ribo",
            "hb",
        ],
        # Avoid expensive top-N-gene calculations
        # for the initial large-scale QC pass.
        percent_top=None,
        log1p=True,
        inplace=False,
    )

    logger.info("QC calculation complete.")

    # --------------------------------------------------------
    # Cell QC
    #
    # These always belong to main obs because the observation
    # axis is shared.
    # --------------------------------------------------------
    output_dir = resolve_output_dir(sczarr_run_qc)

    obs_qc.to_csv(output_dir / "obs_qc.csv")
    var_qc_results.to_csv(output_dir / "var_qc_results.csv")

    logger.info(f"\nQC tables written to {output_dir}")

    # `calculate_qc_metrics` only returns the calculated metrics, so the
    # grouping column has to be taken back from the original obs table.
    plot_obs = attach_grouping_column(
        obs_qc,
        main_obs,
        sczarr_run_qc.batch_key,
    )

    png_path = plot_qc(
        plot_obs,
        output_dir / "qc_plots.png",
        sczarr_run_qc.batch_key,
    )

    logger.info(f"QC plot written to {png_path}")

    return png_path
