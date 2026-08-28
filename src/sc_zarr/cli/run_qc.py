import json
from pathlib import Path
from typing import Any

import click
import cloup
from dask.distributed import Client

from sc_zarr.utils.zarr_qc import ScZarrRunQc
from sc_zarr.utils.zarr_qc import zarr_qc_plot as run_qc_plot_logic

CONFIG_KEYS = {
    "zarr_path",
    "batch_key",
    "covariate_keys",
    "mad_min_group_size",
    "mad_n",
    "qc_source",
    "gene_name",
    "output_dir",
}


def load_config(
    ctx: click.Context,
    param: click.Parameter,
    value: Path | None,
) -> Path | None:
    """Load JSON values as Click defaults.

    Explicit command-line arguments take precedence over these defaults.
    """
    if value is None:
        return None

    try:
        with value.open(encoding="utf-8") as config_file:
            config: Any = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            ctx=ctx,
            param=param,
        ) from exc

    if not isinstance(config, dict):
        raise click.BadParameter(
            "The configuration must contain a JSON object.",
            ctx=ctx,
            param=param,
        )

    unknown_keys = set(config) - CONFIG_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise click.BadParameter(
            f"Unknown configuration keys: {unknown}",
            ctx=ctx,
            param=param,
        )

    # JSON values become defaults. Values explicitly supplied on the
    # command line still take precedence.
    ctx.default_map = {
        **(ctx.default_map or {}),
        **config,
    }

    return value


@cloup.command(
    name="zarr_qc_plot",
    help="Calculate QC metrics on Zarr single cell data and save them as a PNG",
)
@cloup.option(
    "--config",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="JSON configuration file. CLI arguments override its values.",
)
@cloup.option(
    "--zarr_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to input Zarr file",
)
@cloup.option(
    "--batch_key",
    required=False,
    default=None,
    type=str,
    help="Batch key",
)
@cloup.option(
    "--covariate_keys",
    required=False,
    default=[],
    multiple=True,
    type=str,
    help="Covariate keys (can be specified multiple times)",
)
@cloup.option(
    "--qc_source",
    type=click.Choice(["auto", "counts", "raw", "X"]),
    default="auto",
    help="Variable to use for QC",
)
@cloup.option(
    "--gene_name",
    required=False,
    type=str,
    default=None,
    help="Gene name column",
)
@cloup.option(
    "--output_dir",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for the QC PNG and metric tables "
    "(default: <zarr name>_qc next to the Zarr store)",
)
def zarr_qc_plot(
    zarr_path: Path,
    qc_source: str = "auto",
    batch_key: str | None = None,
    covariate_keys: tuple[str, ...] = (),
    gene_name: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Build QC parameters from CLI options and plot the QC metrics.

    Nothing is filtered here: the command only reports the metrics so the
    thresholds can be inspected before running `zarr_qc_filter`.

    Examples
    --------
    >>> zarr_qc_plot(Path("data.zarr"), qc_source="auto", batch_key=None)
    # writes data_qc/qc_plots.png, data_qc/obs_qc.csv, data_qc/var_qc_results.csv
    """
    # Convert tuple to list for `ScZarrRunQc` (dataclass expects a list)
    covariate_key_list = list(covariate_keys) if covariate_keys is not None else []
    sc_zarr_run_qc = ScZarrRunQc(
        zarr_path=zarr_path,
        batch_key=batch_key,
        covariate_keys=covariate_key_list,
        qc_source=qc_source,
        gene_name_column=gene_name,
        output_dir=output_dir,
    )
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
        png_path = run_qc_plot_logic(sc_zarr_run_qc)
        print(f"\nQC plot: {png_path}")
        print("\nDone. Closing Dask cluster.")


@cloup.command(
    name="zarr_qc_filter",
    help="Filter cells with MAD thresholds using the metrics from zarr_qc_plot",
)
@cloup.option(
    "--config",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="JSON configuration file. CLI arguments override its values.",
)
@cloup.option(
    "--zarr_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to input Zarr file",
)
@cloup.option(
    "--batch_key",
    required=False,
    default=None,
    type=str,
    help="Batch key",
)
@cloup.option(
    "--covariate_keys",
    required=False,
    default=[],
    multiple=True,
    type=str,
    help="Covariate keys (can be specified multiple times)",
)
@cloup.option(
    "--mad_n",
    required=False,
    type=float,
    default=3.0,
    help="Number of MADs",
)
@cloup.option(
    "--mad_min_group_size",
    required=False,
    type=int,
    default=100,
    help="Minimum group size for MAD calculation",
)
@cloup.option(
    "--output_dir",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory holding the tables written by zarr_qc_plot "
    "(default: <zarr name>_qc next to the Zarr store)",
)
def zarr_qc_filter(
    zarr_path: Path,
    batch_key: str | None = None,
    covariate_keys: tuple[str, ...] = (),
    mad_n: float = 3.0,
    mad_min_group_size: int = 100,
    output_dir: Path | None = None,
) -> None:
    """Apply the MAD-based cell filtering after the QC plots were reviewed.

    Examples
    --------
    >>> zarr_qc_filter(Path("data.zarr"), mad_n=3.0)
    Error: zarr_qc_filter is not implemented yet.
    """
    raise click.ClickException(
        "zarr_qc_filter is not implemented yet. "
        "Run 'zarr_qc_plot' to inspect the QC metrics."
    )


if __name__ == "__main__":
    zarr_qc_plot()
