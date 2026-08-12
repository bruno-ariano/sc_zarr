import numpy as np
import pandas as pd

# ============================================================
# HELPERS
# ============================================================


def calculate_mad_limits(
    values,
    nmads=3.0,
):
    """
    Calculate robust median/MAD limits.

    Returns
    -------
    dict
        median
        mad
        lower
        upper
        valid
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "median": np.nan,
            "mad": np.nan,
            "lower": -np.inf,
            "upper": np.inf,
            "valid": False,
        }

    median = np.median(values)

    mad = np.median(np.abs(values - median))

    # A zero MAD means the values are essentially identical.
    # Such a group cannot define a useful MAD threshold.
    if not np.isfinite(mad) or mad == 0:
        return {
            "median": median,
            "mad": mad,
            "lower": -np.inf,
            "upper": np.inf,
            "valid": False,
        }

    return {
        "median": median,
        "mad": mad,
        "lower": median - nmads * mad,
        "upper": median + nmads * mad,
        "valid": True,
    }


def add_mad_cell_qc(
    obs_qc,
    obs_metadata,
    *,
    batch_key=None,
    covariate_keys=None,
    nmads=3.0,
    min_group_size=100,
):
    """
    Add MAD-based cell QC flags.

    Parameters
    ----------
    obs_qc
        Output from scanpy.pp.calculate_qc_metrics().

    obs_metadata
        Original AnnData obs metadata.

    batch_key
        Optional technical batch column.

    covariate_keys
        Optional additional categorical technical covariates.

        If multiple variables are supplied, MAD thresholds are
        calculated within their combined strata.

        Example:

            batch_key="batch"
            covariate_keys=["pool"]

        gives strata such as:

            batch1 / pool1
            batch1 / pool2
            batch2 / pool1
            ...

    nmads
        Number of MADs defining an outlier.

    min_group_size
        Minimum cells required to calculate group-specific MADs.

        Smaller groups fall back to global thresholds.

    Returns
    -------
    obs_qc
        QC dataframe with MAD flags and pass_qc.

    mad_stats
        Dataframe containing thresholds used for each stratum.
    """

    covariate_keys = list(covariate_keys or [])

    group_keys = []

    if batch_key is not None:
        group_keys.append(batch_key)

    group_keys.extend(covariate_keys)

    # Remove accidental duplicates while preserving order.
    group_keys = list(dict.fromkeys(group_keys))

    # --------------------------------------------------------
    # Validate metadata columns
    # --------------------------------------------------------

    missing = [key for key in group_keys if key not in obs_metadata.columns]

    if missing:
        raise KeyError(f"MAD grouping columns not found in obs: {missing}")

    result = obs_qc.copy()

    # --------------------------------------------------------
    # Metrics used for cell QC
    #
    # Counts and detected genes use the log1p summaries
    # produced by calculate_qc_metrics(log1p=True).
    #
    # Mitochondrial percentage stays on its original scale.
    # --------------------------------------------------------

    metric_specs = {
        "log1p_total_counts": {
            "lower_flag": "low_counts",
            "upper_flag": "high_counts",
        },
        "log1p_n_genes_by_counts": {
            "lower_flag": "low_genes",
            "upper_flag": "high_genes",
        },
        "pct_counts_mt": {
            "lower_flag": None,
            "upper_flag": "high_mt",
        },
    }

    # --------------------------------------------------------
    # Validate QC columns
    # --------------------------------------------------------

    missing_metrics = [
        metric for metric in metric_specs if metric not in result.columns
    ]

    if missing_metrics:
        raise KeyError(f"Expected QC columns are missing: {missing_metrics}")

    # --------------------------------------------------------
    # Initialize flags
    # --------------------------------------------------------

    flag_names = [
        "low_counts",
        "high_counts",
        "low_genes",
        "high_genes",
        "high_mt",
    ]

    for flag in flag_names:
        result[flag] = False

    # --------------------------------------------------------
    # Global thresholds
    #
    # These are also used as fallback thresholds for groups
    # that are too small or have MAD == 0.
    # --------------------------------------------------------

    global_limits = {}

    for metric in metric_specs:
        global_limits[metric] = calculate_mad_limits(
            result[metric],
            nmads=nmads,
        )

    # --------------------------------------------------------
    # Build strata
    # --------------------------------------------------------

    if not group_keys:
        groups = {"global": result.index}

        print("\nMAD filtering: global")

    else:
        grouping = obs_metadata.reindex(result.index)[group_keys].copy()

        # Make missing categories explicit.
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
            if not isinstance(name, tuple):
                name = (name,)

            group_name = tuple(zip(group_keys, name))

            groups[group_name] = result.index[positions]

        print("\nMAD filtering by:")

        for key in group_keys:
            print(f"  {key}")

        print(f"MAD strata: {len(groups):,}")

    # --------------------------------------------------------
    # Apply MAD filtering
    # --------------------------------------------------------

    stats_records = []

    for group_name, index in groups.items():
        n_cells = len(index)

        use_global = n_cells < min_group_size

        for metric, spec in metric_specs.items():
            group_values = result.loc[
                index,
                metric,
            ]

            if use_global:
                limits = global_limits[metric]

                threshold_source = "global_fallback"

            else:
                limits = calculate_mad_limits(
                    group_values,
                    nmads=nmads,
                )

                # If the group has MAD == 0,
                # fall back to the global distribution.
                if not limits["valid"]:
                    limits = global_limits[metric]

                    threshold_source = "global_fallback"

                else:
                    threshold_source = "group"

            # ----------------------------------------------
            # Lower-tail outlier
            # ----------------------------------------------

            lower_flag = spec["lower_flag"]

            if lower_flag is not None:
                result.loc[
                    index,
                    lower_flag,
                ] = group_values < limits["lower"]

            # ----------------------------------------------
            # Upper-tail outlier
            # ----------------------------------------------

            upper_flag = spec["upper_flag"]

            if upper_flag is not None:
                result.loc[
                    index,
                    upper_flag,
                ] = group_values > limits["upper"]

            stats_records.append(
                {
                    "group": str(group_name),
                    "n_cells": n_cells,
                    "metric": metric,
                    "median": limits["median"],
                    "mad": limits["mad"],
                    "lower": limits["lower"],
                    "upper": limits["upper"],
                    "threshold_source": threshold_source,
                }
            )

    # --------------------------------------------------------
    # Combined QC decision
    # --------------------------------------------------------

    result["n_mad_outliers"] = result[flag_names].sum(axis=1)

    result["pass_qc"] = ~(result[flag_names].any(axis=1))

    mad_stats = pd.DataFrame(stats_records)

    return result, mad_stats
