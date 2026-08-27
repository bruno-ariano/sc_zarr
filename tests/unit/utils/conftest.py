from pathlib import Path

import pytest

from sc_zarr.utils.ingestion_zarr import ingest_h5ad


@pytest.fixture
def h5ad_file() -> str:
    return "tests/data/df1e3368-67f5-49ce-92cd-fe4f13298b4f.h5ad"


@pytest.fixture
def zarr_data(tmp_path: Path, h5ad_file: str) -> Path:
    zarr_path = tmp_path / "test.zarr"
    ingest_h5ad(h5ad_file, str(zarr_path))
    return zarr_path
