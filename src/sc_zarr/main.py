import cloup

from sc_zarr.cli.ingestion import h5ad_zarr
from sc_zarr.cli.run_qc import zarr_qc_filter, zarr_qc_plot


@cloup.group(
    name="sc_zarr", help="Basic scRNA-seq zarr processing", no_args_is_help=True
)
def cli() -> None:
    """CLI entry point."""


def main() -> None:
    cli.add_command(h5ad_zarr)
    cli.add_command(zarr_qc_plot)
    cli.add_command(zarr_qc_filter)
    cli()


if __name__ == "__main__":
    main()
