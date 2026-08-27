from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from sc_zarr.utils.ingestion_zarr import ingest_h5ad


@dataclass
class ScZarrIngestion:
    h5ad: Path | None = None
    out_zarr: Path | None = None
    mad: float | None = None
    mad_group_size: int | None = None
    min_gene_cells: int | None = None
    batch: str | None = None
    covariates: tuple[str, ...] = ()

    def run(self) -> None:
        if self.h5ad is None:
            raise ValueError("h5ad must be provided")

        if self.out_zarr is None:
            raise ValueError("out_zarr must be provided")

        ingest_h5ad(
            input_h5ad=self.h5ad,
            output_zarr=self.out_zarr,
        )

        logger.info("Ingestion completed")


def main(
    h5ad: Path | None,
    out_zarr: Path | None,
    mad: float | None,
    mad_group_size: int | None,
    min_gene_cells: int | None,
    batch: str | None,
    covariates: tuple[str, ...],
) -> None:
    processor = ScZarrIngestion(
        h5ad=h5ad,
        out_zarr=out_zarr,
        mad=mad,
        mad_group_size=mad_group_size,
        min_gene_cells=min_gene_cells,
        batch=batch,
        covariates=covariates,
    )

    processor.run()
