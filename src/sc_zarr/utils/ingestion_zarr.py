import shutil
from pathlib import Path

import anndata as ad
import cloup
from loguru import logger

ad.settings.allow_write_nullable_strings = True


def ingest_h5ad(
    input_h5ad: str | Path,
    output_zarr: str | Path,
) -> None:
    input_h5ad = Path(input_h5ad)
    output_zarr = Path(output_zarr)

    ad.settings.zarr_write_format = 3
    ad.settings.auto_shard_zarr_v3 = True

    if output_zarr.exists():
        logger.info(f"Removing existing zarr folder {output_zarr}")
        shutil.rmtree(output_zarr)

    adata = ad.experimental.read_lazy(input_h5ad)

    logger.info(f"Loaded the corresponding anndata with dimension {adata.shape}")

    try:
        # -----------------------------------------------------
        # Main obs / var metadata
        # -----------------------------------------------------

        if hasattr(adata.obs, "to_memory"):
            logger.info("\nLoading obs metadata into memory...")
            adata.obs = adata.obs.to_memory()

        if hasattr(adata.var, "to_memory"):
            logger.info("Loading var metadata into memory...")
            adata.var = adata.var.to_memory()

        # -----------------------------------------------------
        # raw.var metadata
        #
        # raw.X stays lazy.
        # Rebuild Raw through the public AnnData/raw API rather
        # than modifying private attributes such as raw._var.
        # -----------------------------------------------------

        if adata.raw is not None and hasattr(adata.raw.var, "to_memory"):
            logger.info("Loading raw.var metadata into memory...")

            raw_adata = ad.AnnData(
                X=adata.raw.X,
                var=adata.raw.var.to_memory(),
                varm=dict(adata.raw.varm),
            )

            adata.raw = raw_adata

        if adata.raw is not None:
            logger.debug("raw.X:   ", type(adata.raw.X))
            logger.debug("raw.var: ", type(adata.raw.var))

        # -----------------------------------------------------
        # Write
        # -----------------------------------------------------

        logger.info(f"Starting writing zarr in {output_zarr}...")
        adata.write_zarr(output_zarr)
    except Exception as e:
        logger.error(f"Failed to ingest the anndata caused by {e}")
        raise

    finally:
        if adata.file is not None:
            adata.file.close()
