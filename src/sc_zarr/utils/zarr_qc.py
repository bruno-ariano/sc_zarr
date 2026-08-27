from pathlib import Path

import anndata as ad
import click
import cloup
import scanpy as sc
import zarr
from dask.distributed import Client
from loguru import logger

from sc_zarr.utils.zarr_stats import (
    QcSource,
    add_dataframe_columns,
    add_qc_gene_sets,
    choose_qc_matrix,
)

# ============================================================
# CONFIG
# ============================================================
ad.settings.allow_write_nullable_strings = True
# ============================================================
# QC CONFIG
# ============================================================
# Number of MADs used to call a cell an outlier.
MAD_N = 3.0

# If batch/covariate grouping creates a group smaller than this,
# use global MAD thresholds for that group instead.
MAD_MIN_GROUP_SIZE = 100

# Gene filtering remains a prevalence rule rather than MAD.
MIN_CELLS_PER_GENE = 3

# If both are None/empty, MADs are calculated globally.
BATCH_KEY = None
COVARIATE_KEYS: list[str] = []


ZARR_PATH = Path("/Users/bruno.ariano/projects/hello_world/sc_practice/pbmc_adata.zarr")

# Which matrix should be used for QC?
#
# "auto":
#     1. layers["counts"] if present
#     2. raw.X if present
#     3. X otherwise
#
# You can also explicitly use:
#     "counts"
#     "raw"
#     "X"
QC_SOURCE: QcSource = "auto"


# Example QC thresholds.
#
# IMPORTANT:
# These are not universal biological thresholds.
MIN_GENES = 200
MAX_GENES = 10_000
MAX_MT = 20


@cloup.option(
    "--zarr-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input zarr file",
)
def zarr_qc(zarr_path: Path) -> None:
    logger.info("\n# QC\n")
    logger.info(f"Zarr: {zarr_path}")

    # --------------------------------------------------------
    # Open Zarr
    # --------------------------------------------------------

    root = zarr.open_group(
        zarr_path,
        mode="r",
    )

    # obs and var are small metadata tables.
    main_obs = ad.io.read_elem(root["obs"])

    main_var = ad.io.read_elem(root["var"])

    print()
    print(f"Main AnnData: {len(main_obs):,} cells x {len(main_var):,} genes")

    # --------------------------------------------------------
    # Select counts matrix
    # --------------------------------------------------------

    X_qc, qc_var, source = choose_qc_matrix(
        root,
        main_var,
        QC_SOURCE,
    )

    print("\nLazy QC matrix:")
    print(X_qc)

    print("\nChunks:")
    print(X_qc.chunks)

    print("\nSparse chunk type:")
    print(type(X_qc._meta))

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if X_qc.shape[0] != len(main_obs):
        raise ValueError(
            "QC matrix cell count does not match obs: "
            f"{X_qc.shape[0]} != {len(main_obs)}"
        )

    if X_qc.shape[1] != len(qc_var):
        raise ValueError(
            "QC matrix gene count does not match its var: "
            f"{X_qc.shape[1]} != {len(qc_var)}"
        )

    # --------------------------------------------------------
    # Gene categories
    # --------------------------------------------------------

    qc_var = add_qc_gene_sets(qc_var.copy())

    # --------------------------------------------------------
    # Temporary AnnData used only for QC
    #
    # X remains lazy / Dask-backed.
    # --------------------------------------------------------

    adata_qc = ad.AnnData(
        X=X_qc,
        obs=main_obs.copy(),
        var=qc_var,
    )

    print("\nQC AnnData:")
    print(adata_qc)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    print("\nCalculating QC metrics...")

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

    print("QC calculation complete.")

    # --------------------------------------------------------
    # Cell QC
    #
    # These always belong to main obs because the observation
    # axis is shared.
    # --------------------------------------------------------

    add_dataframe_columns(
        main_obs,
        obs_qc,
    )

    # --------------------------------------------------------
    # Cell pass/fail flag
    # --------------------------------------------------------

    main_obs["pass_qc"] = (
        (main_obs["n_genes_by_counts"] >= MIN_GENES)
        & (main_obs["n_genes_by_counts"] <= MAX_GENES)
        & (main_obs["pct_counts_mt"] <= MAX_MT)
    )

    # --------------------------------------------------------
    # Gene QC
    #
    # var_qc_results corresponds to whichever var was paired
    # with the QC matrix.
    # --------------------------------------------------------

    qc_var_output = qc_var.copy()

    add_dataframe_columns(
        qc_var_output,
        var_qc_results,
    )

    qc_var_output["pass_qc"] = qc_var_output["n_cells_by_counts"] >= MIN_CELLS_PER_GENE

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n# CELL QC SUMMARY\n")

    print(
        main_obs[
            [
                "total_counts",
                "n_genes_by_counts",
                "pct_counts_mt",
                "pct_counts_ribo",
                "pct_counts_hb",
            ]
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.50,
                0.95,
                0.99,
            ]
        )
    )

    print("\n# GENE QC SUMMARY\n")

    print(
        qc_var_output[
            [
                "total_counts",
                "n_cells_by_counts",
                "mean_counts",
            ]
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.50,
                0.95,
                0.99,
            ]
        )
    )

    print(f"\nCells passing QC: {main_obs['pass_qc'].sum():,} / {len(main_obs):,}")

    print(
        f"Genes passing QC: {qc_var_output['pass_qc'].sum():,} / {len(qc_var_output):,}"
    )

    # ========================================================
    # DECIDE WHERE GENE QC BELONGS
    # ========================================================

    write_raw_var = False

    if source in {"X", "counts"}:
        # X and layers always use the main AnnData var.
        main_var_output = main_var.copy()

        add_dataframe_columns(
            main_var_output,
            qc_var_output,
        )

    elif source == "raw":
        # raw.X belongs to raw.var.
        #
        # If raw.var and main var contain exactly the same genes,
        # we also copy QC metrics into the main var for convenience.
        same_gene_index = len(main_var) == len(qc_var_output) and main_var.index.equals(
            qc_var_output.index
        )

        print(
            "\nraw.var matches main var:",
            same_gene_index,
        )

        if same_gene_index:
            main_var_output = main_var.copy()

            add_dataframe_columns(
                main_var_output,
                qc_var_output,
            )

            print("Gene QC metrics will be stored in both var and raw.var.")

        else:
            main_var_output = main_var.copy()

            print("raw.var differs from main var.")
            print("Gene QC metrics will only be stored in raw.var.")

        write_raw_var = True

    else:
        raise RuntimeError(f"Unexpected QC source: {source}")

    # ========================================================
    # WRITE METADATA BACK
    # ========================================================

    print("\nSaving QC metadata...")

    # Reopen writable, bypassing possibly stale consolidated
    # metadata while editing.
    root_write = zarr.open_group(
        zarr_path,
        mode="a",
        use_consolidated=False,
    )

    # --------------------------------------------------------
    # obs
    # --------------------------------------------------------

    ad.io.write_elem(
        root_write,
        "obs",
        main_obs,
    )

    # --------------------------------------------------------
    # main var
    # --------------------------------------------------------

    ad.io.write_elem(
        root_write,
        "var",
        main_var_output,
    )

    # --------------------------------------------------------
    # raw.var if QC was calculated from raw.X
    #
    # raw.X itself is untouched.
    # --------------------------------------------------------

    if write_raw_var:
        ad.io.write_elem(
            root_write["raw"],
            "var",
            qc_var_output,
        )

    # --------------------------------------------------------
    # Refresh consolidated metadata
    # --------------------------------------------------------

    print("Consolidating Zarr metadata...")

    zarr.consolidate_metadata(root_write.store)

    print("\nQC metadata saved.")
    print("Expression matrices were not rewritten.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # One local Dask worker.
    #
    # processes=False is convenient on macOS and avoids
    # multiprocessing spawn complications.
    with Client(
        n_workers=1,
        threads_per_worker=4,
        processes=False,
        memory_limit="8GB",
    ) as client:
        print(
            "Dask dashboard:",
            client.dashboard_link,
        )

        zarr_qc(ZARR_PATH)

        print("\nDone. Closing Dask cluster.")
