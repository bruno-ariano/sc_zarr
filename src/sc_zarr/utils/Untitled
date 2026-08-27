from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import zarr
from dask.distributed import Client

# ============================================================
# HELPERS
# ============================================================


def choose_qc_matrix(root, main_var, qc_source):
    """
    Select the matrix used for QC.

    Returns
    -------
    X_qc
        Lazy Dask-backed expression matrix.

    qc_var
        Gene metadata matching X_qc.

    source
        One of:
            "counts"
            "raw"
            "X"
    """

    # --------------------------------------------------------
    # Explicit selection
    # --------------------------------------------------------

    if qc_source == "counts":
        if "layers" not in root or "counts" not in root["layers"]:
            raise ValueError('QC_SOURCE="counts", but layers["counts"] does not exist.')

        print("QC source: layers['counts']")

        X_qc = ad.experimental.read_elem_lazy(root["layers"]["counts"])

        return X_qc, main_var.copy(), "counts"

    if qc_source == "raw":
        if "raw" not in root or "X" not in root["raw"]:
            raise ValueError('QC_SOURCE="raw", but raw.X does not exist.')

        print("QC source: raw.X")

        X_qc = ad.experimental.read_elem_lazy(root["raw"]["X"])

        # raw.X must be paired with raw.var
        qc_var = ad.io.read_elem(root["raw"]["var"])

        return X_qc, qc_var, "raw"

    if qc_source == "X":
        print("QC source: X")

        X_qc = ad.experimental.read_elem_lazy(root["X"])

        return X_qc, main_var.copy(), "X"

    if qc_source != "auto":
        raise ValueError(f"Unknown QC_SOURCE: {qc_source!r}")

    # --------------------------------------------------------
    # Automatic selection
    # --------------------------------------------------------

    if "layers" in root and "counts" in root["layers"]:
        print("QC source: layers['counts']")

        X_qc = ad.experimental.read_elem_lazy(root["layers"]["counts"])

        return X_qc, main_var.copy(), "counts"

    if "raw" in root and "X" in root["raw"]:
        print("QC source: raw.X")

        X_qc = ad.experimental.read_elem_lazy(root["raw"]["X"])

        qc_var = ad.io.read_elem(root["raw"]["var"])

        return X_qc, qc_var, "raw"

    print("QC source: X")

    X_qc = ad.experimental.read_elem_lazy(root["X"])

    return X_qc, main_var.copy(), "X"


def get_gene_names(var):
    """
    Find the best available gene-symbol column.

    Many CellxGene-style AnnData objects use feature_name
    while var_names contains Ensembl IDs.
    """
    candidates = [
        "feature_name",
        "gene_symbol",
        "gene_name",
        "symbol",
    ]
    for column in candidates:
        if column in var.columns:
            print(f"Using var['{column}'] for gene symbols.")
            return var[column].astype(str)
    print("No gene-symbol column found; using var_names.")
    return var.index.astype(str)


def add_qc_gene_sets(var):
    """
    Add common human QC gene categories.
    """

    gene_names = get_gene_names(var)

    # Human mitochondrial genes
    var["mt"] = gene_names.str.startswith(
        "MT-",
        na=False,
    )

    # Ribosomal proteins
    var["ribo"] = gene_names.str.startswith(
        ("RPS", "RPL"),
        na=False,
    )

    # Hemoglobin genes, excluding pseudogenes such as HBP1
    var["hb"] = gene_names.str.match(
        r"^HB(?!P)",
        na=False,
    )

    print("\nQC gene sets:")
    print(f"  Mitochondrial: {var['mt'].sum():,}")
    print(f"  Ribosomal:     {var['ribo'].sum():,}")
    print(f"  Hemoglobin:    {var['hb'].sum():,}")

    if var["mt"].sum() == 0:
        print("\nWARNING: no mitochondrial genes detected.")
        print("Check the gene-symbol column and species.")

    return var


def add_dataframe_columns(target, source):
    """
    Copy all columns from source DataFrame into target,
    aligned by index.
    """

    for column in source.columns:
        target[column] = source[column]
