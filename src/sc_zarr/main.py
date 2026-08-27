import cloup

from sc_zarr.cli.ingestion import ingest_h5ad
from sc_zarr.cli.run_qc import run_qc


@cloup.group(
    name="sc_zarr", help="Basic scRNA-seq zarr processing", no_args_is_help=True
)
def cli() -> None:
    """CLI entry point."""


def main() -> None:
    cli.add_command(ingest_h5ad)
    cli.add_command(run_qc)
    cli()


if __name__ == "__main__":
    main()
