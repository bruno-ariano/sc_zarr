import shutil
from pathlib import Path

import anndata as ad

ad.settings.allow_write_nullable_strings = True


def ingest_h5ad(
    input_h5ad: str | Path,
    output_zarr: str | Path,
):
    input_h5ad = Path(input_h5ad)
    output_zarr = Path(output_zarr)

    print("\n# INGEST\n")
    print(f"Input:  {input_h5ad}")
    print(f"Output: {output_zarr}")

    ad.settings.zarr_write_format = 3
    ad.settings.auto_shard_zarr_v3 = True

    if output_zarr.exists():
        print("Removing existing output...")
        shutil.rmtree(output_zarr)

    adata = ad.experimental.read_lazy(input_h5ad)

    print("\nLoaded:")
    print(adata)

    try:
        # -----------------------------------------------------
        # Main obs / var metadata
        # -----------------------------------------------------

        if hasattr(adata.obs, "to_memory"):
            print("\nLoading obs metadata into memory...")
            adata.obs = adata.obs.to_memory()

        if hasattr(adata.var, "to_memory"):
            print("Loading var metadata into memory...")
            adata.var = adata.var.to_memory()

        # -----------------------------------------------------
        # raw.var metadata
        #
        # raw.X stays lazy.
        # Rebuild Raw through the public AnnData/raw API rather
        # than modifying private attributes such as raw._var.
        # -----------------------------------------------------

        if adata.raw is not None and hasattr(adata.raw.var, "to_memory"):
            print("Loading raw.var metadata into memory...")

            raw_adata = ad.AnnData(
                X=adata.raw.X,
                var=adata.raw.var.to_memory(),
                varm=dict(adata.raw.varm),
            )

            adata.raw = raw_adata

        # -----------------------------------------------------
        # Debug types
        # -----------------------------------------------------

        print("\nBefore writing:")
        print("X:       ", type(adata.X))
        print("obs:     ", type(adata.obs))
        print("var:     ", type(adata.var))

        if adata.raw is not None:
            print("raw.X:   ", type(adata.raw.X))
            print("raw.var: ", type(adata.raw.var))

        # -----------------------------------------------------
        # Write
        # -----------------------------------------------------

        print("\nWriting Zarr...")
        adata.write_zarr(output_zarr)

    finally:
        if adata.file is not None:
            adata.file.close()

    print("\nIngest complete.")
    print(f"Zarr written to: {output_zarr}")


if __name__ == "__main__":
    INPUT_H5AD = (
        "/Users/bruno.ariano/projects/hello_world/sc_practice/"
        "770a441b-b903-4d22-8873-09b1abfd797a.h5ad"
    )

    OUTPUT_ZARR = "/Users/bruno.ariano/projects/hello_world/sc_practice/pbmc_adata.zarr"

    ingest_h5ad(
        INPUT_H5AD,
        OUTPUT_ZARR,
    )


import zarr

root = zarr.open_group(OUTPUT_ZARR, mode="r")

print("X format:", root["X"].attrs["encoding-type"])

if "raw" in root and "X" in root["raw"]:
    print("raw.X format:", root["raw"]["X"].attrs["encoding-type"])
