from pathlib import Path

import pytest

from sc_zarr.utils.ingestion_zarr import ingest_h5ad


@pytest.fixture
def h5ad_file() -> str:
    return "tests/data/1d29777b-80c4-44ed-a6b2-e5431c074c34.h5ad"


@pytest.fixture
def zarr_data(tmp_path: Path, h5ad_file: str) -> Path:
    zarr_path = tmp_path / "test.zarr"
    # Use Click's parameter syntax: pass options as --input-h5ad ... --output-zarr ...
    ingest_h5ad(h5ad_file, str(zarr_path))
    # Ensure the command completed successfully
    return zarr_path
