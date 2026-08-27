from dataclasses import dataclass
from pathlib import Path

import cloup
from loguru import logger

from sc_zarr.utils.ingestion_zarr import ingest_h5ad


@dataclass
class ScZarrIngestion:
    h5ad: Path | None = None
    out_zarr: Path | None = None

    def run(self) -> None:
        if self.h5ad is None:
            raise ValueError("h5ad must be provided")

        if self.out_zarr is None:
            raise ValueError("out_zarr must be provided")

        ingest_h5ad(
            h5ad=self.h5ad,
            output_zarr=self.out_zarr,
        )

        logger.info("Ingestion completed")


@cloup.command(name="ingest-h5ad")
@cloup.option("--h5ad", type=str, help="Input h5ad file")
@cloup.option("--output_zarr", type=str, help="Output zarr file")
def ingest_h5ad_cli(
    h5ad: Path | None,
    out_zarr: Path | None,
) -> None:
    processor = ScZarrIngestion(
        h5ad=h5ad,
        out_zarr=out_zarr,
    )

    processor.run()
