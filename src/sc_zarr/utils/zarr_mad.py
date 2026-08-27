from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class MadLimits:
    """Median and MAD thresholds calculated for one numeric distribution."""

    median: float
    mad: float
    lower: float
    upper: float
    valid: bool


@dataclass(frozen=True)
class MetricSpec:
    """Names of the lower- and upper-tail flags for one QC metric."""

    lower_flag: str | None
    upper_flag: str | None


def calculate_mad_limits(
    values: npt.ArrayLike | pd.Series,
    nmads: float = 3.0,
) -> MadLimits:
    """Calculate robust median/MAD limits for a numeric distribution.

    Non-finite values are ignored. If no finite observations are available or
    the MAD is zero, ``valid`` is false and the limits are unbounded.
    """
    if nmads <= 0:
        raise ValueError("nmads must be greater than zero")

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return MadLimits(
            median=float("nan"),
            mad=float("nan"),
            lower=float("-inf"),
            upper=float("inf"),
            valid=False,
        )

    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))

    if not np.isfinite(mad) or mad == 0:
        return MadLimits(
            median=median,
            mad=mad,
            lower=float("-inf"),
            upper=float("inf"),
            valid=False,
        )

    return MadLimits(
        median=median,
        mad=mad,
        lower=median - nmads * mad,
        upper=median + nmads * mad,
        valid=True,
    )


def add_mad_cell_qc(
    obs_qc: pd.DataFrame,
    obs_metadata: pd.DataFrame,
    *,
    batch_key: str | None = None,
    covariate_keys: str | Sequence[str] | None = None,
    nmads: float = 3.0,
    min_group_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add MAD-based cell QC flags.

    Counts and detected-gene thresholds are calculated from the log1p metrics
    produced by ``scanpy.pp.calculate_qc_metrics(log1p=True)``. Mitochondrial
    percentage is evaluated on its original scale.

    If grouping columns are supplied, thresholds are calculated independently
    within their combined strata. Groups smaller than ``min_group_size``, and
    groups whose MAD is zero, use the corresponding global threshold.

    Parameters
    ----------
    obs_qc
        Cell-level output from ``scanpy.pp.calculate_qc_metrics``.
    obs_metadata
        Original AnnData observation metadata. It may be in a different order,
        but it must contain every index present in ``obs_qc``.
    batch_key
        Optional technical batch column.
    covariate_keys
        Optional categorical technical covariate, or a sequence of covariates.
    nmads
        Positive number of MADs defining an outlier.
    min_group_size
        Positive minimum number of cells needed for group-specific thresholds.

    Returns
    -------
    result
        A copy of ``obs_qc`` containing individual MAD flags,
        ``n_mad_outliers``, and ``pass_qc``.
    mad_stats
        One record per stratum and metric describing the applied thresholds.
    """
    if nmads <= 0:
        raise ValueError("nmads must be greater than zero")

    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1")

    if not obs_qc.index.is_unique:
        raise ValueError("obs_qc must have a unique index")

    if not obs_metadata.index.is_unique:
        raise ValueError("obs_metadata must have a unique index")

    missing_cells = obs_qc.index[~obs_qc.index.isin(obs_metadata.index)]
    if not missing_cells.empty:
        raise ValueError(
            f"obs_metadata is missing {len(missing_cells):,} cells present in obs_qc"
        )

    if covariate_keys is None:
        covariates: list[str] = []
    elif isinstance(covariate_keys, str):
        covariates = [covariate_keys]
    else:
        covariates = list(covariate_keys)

    group_keys: list[str] = []
    if batch_key is not None:
        group_keys.append(batch_key)
    group_keys.extend(covariates)
    group_keys = list(dict.fromkeys(group_keys))

    missing_columns = [key for key in group_keys if key not in obs_metadata.columns]
    if missing_columns:
        raise KeyError(
            f"MAD grouping columns not found in obs_metadata: {missing_columns}"
        )

    result = obs_qc.copy()

    metric_specs: dict[str, MetricSpec] = {
        "log1p_total_counts": MetricSpec(
            lower_flag="low_counts",
            upper_flag="high_counts",
        ),
        "log1p_n_genes_by_counts": MetricSpec(
            lower_flag="low_genes",
            upper_flag="high_genes",
        ),
        "pct_counts_mt": MetricSpec(
            lower_flag=None,
            upper_flag="high_mt",
        ),
    }

    missing_metrics = [
        metric for metric in metric_specs if metric not in result.columns
    ]
    if missing_metrics:
        raise KeyError(f"Expected QC columns are missing: {missing_metrics}")

    flag_names: list[str] = [
        "low_counts",
        "high_counts",
        "low_genes",
        "high_genes",
        "high_mt",
    ]
    for flag in flag_names:
        result[flag] = False

    global_limits: dict[str, MadLimits] = {
        metric: calculate_mad_limits(result[metric], nmads=nmads)
        for metric in metric_specs
    }

    groups: dict[str, pd.Index]

    if not group_keys:
        groups = {"global": result.index}
        print("\nMAD filtering: global")
    else:
        grouping = obs_metadata.reindex(result.index)[group_keys].copy()

        for key in group_keys:
            grouping[key] = grouping[key].astype("string").fillna("__MISSING__")

        groups = {}
        grouped = grouping.groupby(
            group_keys,
            observed=True,
            dropna=False,
            sort=False,
        )

        for name, positions in grouped.indices.items():
            group_values = name if isinstance(name, tuple) else (name,)
            group_label = " / ".join(
                f"{key}={value}" for key, value in zip(group_keys, group_values)
            )
            groups[group_label] = result.index[positions]

        print("\nMAD filtering by:")
        for key in group_keys:
            print(f"  {key}")
        print(f"MAD strata: {len(groups):,}")

    stats_records: list[dict[str, object]] = []

    for group_name, index in groups.items():
        n_cells = len(index)

        for metric, spec in metric_specs.items():
            metric_values = result.loc[index, metric]

            if group_name == "global":
                limits = global_limits[metric]
                threshold_source = "global"
            elif n_cells < min_group_size:
                limits = global_limits[metric]
                threshold_source = "global_fallback"
            else:
                limits = calculate_mad_limits(metric_values, nmads=nmads)

                if not limits.valid:
                    limits = global_limits[metric]
                    threshold_source = "global_fallback"
                else:
                    threshold_source = "group"

            if spec.lower_flag is not None:
                result.loc[index, spec.lower_flag] = metric_values < limits.lower

            if spec.upper_flag is not None:
                result.loc[index, spec.upper_flag] = metric_values > limits.upper

            stats_records.append(
                {
                    "group": group_name,
                    "n_cells": n_cells,
                    "metric": metric,
                    "median": limits.median,
                    "mad": limits.mad,
                    "lower": limits.lower,
                    "upper": limits.upper,
                    "threshold_source": threshold_source,
                    "threshold_valid": limits.valid,
                }
            )

    result["n_mad_outliers"] = result[flag_names].sum(axis=1)
    result["pass_qc"] = ~result[flag_names].any(axis=1)

    mad_stats = pd.DataFrame(stats_records)
    return result, mad_stats
