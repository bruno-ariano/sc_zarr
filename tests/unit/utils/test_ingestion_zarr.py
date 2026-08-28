from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from scipy import sparse as sp

from sc_zarr.utils.ingestion_zarr import ingest_h5ad


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


def test_ingest_h5ad_overwrites_existing_zarr(zarr_data: Path, h5ad_file: str) -> None:
    # 1. zarr_data fixture has already created the directory
    assert zarr_data.exists()

    # 2. Re-run ingestion on the existing directory (triggers lines 20-21)
    ingest_h5ad(h5ad=h5ad_file, output_zarr=zarr_data)

    # 3. Confirm output directory still exists after being wiped and re-created
    assert zarr_data.exists()


def test_ingest_h5ad_raises_on_invalid_file(tmp_path: Path):
    invalid_file = tmp_path / "non_existent.h5ad"
    zarr_path = tmp_path / "output.zarr"

    with pytest.raises(FileNotFoundError):
        ingest_h5ad(h5ad=invalid_file, output_zarr=zarr_path)
