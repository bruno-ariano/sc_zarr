from pathlib import Path

import anndata as ad
import numpy as np
from pandas.testing import assert_frame_equal

# test_ingestion_zarr.py
from scipy import sparse as sp


def test_ingest_h5ad_metadata_parity(h5ad_file: str, zarr_data: Path) -> None:
    ad_zarr = ad.read_zarr(zarr_data)
    ad_h5ad = ad.read_h5ad(h5ad_file)

    assert_frame_equal(
        ad_zarr.obs,
        ad_h5ad.obs,
        check_dtype=False,
        check_categorical=False,
        check_index_type=False,
    )
    assert_frame_equal(
        ad_zarr.var,
        ad_h5ad.var,
        check_dtype=False,
        check_categorical=False,
        check_index_type=False,
    )


def test_ingest_h5ad_matrix_parity(h5ad_file: str, zarr_data: Path) -> None:
    ad_h5ad = ad.read_h5ad(h5ad_file)
    ad_zarr = ad.read_zarr(zarr_data)

    X_zarr = ad_zarr.X.toarray() if sp.issparse(ad_zarr.X) else ad_zarr.X
    X_h5ad = ad_h5ad.X.toarray() if sp.issparse(ad_h5ad.X) else ad_h5ad.X

    np.testing.assert_allclose(X_zarr, X_h5ad, rtol=1e-5)
