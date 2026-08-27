from pathlib import Path

import click
import cloup

from sc_zarr.utils.zarr_qc import zarr_qc as run_qc_logic


@cloup.command(help="Run QC on Zarr single cell data")
@cloup.option(
    "--zarr-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input Zarr file",
)
def run_qc(zarr_path: Path) -> None:
    """CLI entry point for running QC."""
    run_qc_logic(zarr_path)
