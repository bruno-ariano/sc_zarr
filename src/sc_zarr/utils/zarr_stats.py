from __future__ import annotations

from typing import Literal, TypeAlias

import anndata as ad
import pandas as pd
import zarr
from dask.array import Array as DaskArray

QcSource: TypeAlias = Literal["auto", "counts", "raw", "X"]
SelectedQcSource: TypeAlias = Literal["counts", "raw", "X"]

# ============================================================
# HELPERS
# ============================================================


def _require_group(parent: zarr.Group, key: str) -> zarr.Group:
    """Return a named Zarr subgroup.

    Zarr types ``__getitem__`` as ``Array | Group``. This helper narrows the
    result so nested key lookups are valid for mypy.

    Examples
    --------
    >>> layers = _require_group(root, "layers")
    >>> "counts" in layers
    True
    """
    if key not in parent:
        raise KeyError(f"Expected Zarr group {key!r} is missing.")

    child = parent[key]
    if not isinstance(child, zarr.Group):
        raise TypeError(f"Expected Zarr group at {key!r}, got {type(child).__name__}.")
    return child


def choose_qc_matrix(
    root: zarr.Group,
    main_var: pd.DataFrame,
    qc_source: QcSource,
) -> tuple[DaskArray, pd.DataFrame, SelectedQcSource]:
    """Select the matrix used for QC.

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

    Examples
    --------
    >>> X_qc, qc_var, source = choose_qc_matrix(root, main_var, "auto")
    >>> source
    'counts'
    """

    # --------------------------------------------------------
    # Explicit selection
    # --------------------------------------------------------

    if qc_source == "counts":
        if "layers" not in root:
            raise ValueError('QC_SOURCE="counts", but layers["counts"] does not exist.')

        layers = _require_group(root, "layers")
        if "counts" not in layers:
            raise ValueError('QC_SOURCE="counts", but layers["counts"] does not exist.')

        print("QC source: layers['counts']")

        X_qc = ad.experimental.read_elem_lazy(layers["counts"])

        return X_qc, main_var.copy(), "counts"

    if qc_source == "raw":
        if "raw" not in root or "X" not in _require_group(root, "raw"):
            raise ValueError('QC_SOURCE="raw", but raw.X does not exist.')

        print("QC source: raw.X")

        raw = _require_group(root, "raw")
        X_qc = ad.experimental.read_elem_lazy(raw["X"])

        # raw.X must be paired with raw.var
        qc_var: pd.DataFrame = ad.io.read_elem(raw["var"])

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

    if "layers" in root:
        layers = _require_group(root, "layers")
        if "counts" in layers:
            print("QC source: layers['counts']")

            X_qc = ad.experimental.read_elem_lazy(layers["counts"])

            return X_qc, main_var.copy(), "counts"

    if "raw" in root and "X" in _require_group(root, "raw"):
        print("QC source: raw.X")

        raw = _require_group(root, "raw")
        X_qc = ad.experimental.read_elem_lazy(raw["X"])

        qc_var = ad.io.read_elem(raw["var"])

        return X_qc, qc_var, "raw"

    print("QC source: X")

    X_qc = ad.experimental.read_elem_lazy(root["X"])

    return X_qc, main_var.copy(), "X"


def get_gene_names(var: pd.DataFrame) -> pd.Series:
    """Find the best available gene-symbol column.

    Many CellxGene-style AnnData objects use feature_name
    while var_names contains Ensembl IDs.

    Examples
    --------
    >>> var = pd.DataFrame({"feature_name": ["MT-ND1", "GAPDH"]})
    >>> get_gene_names(var).tolist()
    ['MT-ND1', 'GAPDH']
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
    return pd.Series(var.index.astype(str), index=var.index)


def add_qc_gene_sets(var: pd.DataFrame) -> pd.DataFrame:
    """Add common human QC gene categories.

    Examples
    --------
    >>> var = pd.DataFrame(index=["MT-ND1", "RPS3", "HBA1", "GAPDH"])
    >>> flagged = add_qc_gene_sets(var)
    >>> flagged.loc["MT-ND1", "mt"]
    True
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


def add_dataframe_columns(target: pd.DataFrame, source: pd.DataFrame) -> None:
    """Copy all columns from source DataFrame into target, aligned by index.

    Examples
    --------
    >>> target = pd.DataFrame({"a": [1, 2]}, index=["c1", "c2"])
    >>> source = pd.DataFrame({"b": [10, 20]}, index=["c1", "c2"])
    >>> add_dataframe_columns(target, source)
    >>> target["b"].tolist()
    [10, 20]
    """

    for column in source.columns:
        target[column] = source[column]
