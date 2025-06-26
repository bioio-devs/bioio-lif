import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from bioio_base import dimensions

from bioio_lif.reader import Reader

RESOURCE_DIR = Path(__file__).parent / "resources"


def make_mock_reader(
    metadata_file: Path,
    dims: str,
    shape: tuple,
) -> Reader:
    metadata = ET.parse(metadata_file).getroot()

    with patch("bioio_base.io.pathlike_to_fs") as mock_fs_patch, patch(
        "bioio_lif.reader.LifFile"
    ) as mock_lif_patch:
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_fs_patch.return_value = (mock_fs, "dummy_path.lif")

        mock_lif = MagicMock()
        mock_lif.image_list = [{"name": "Scene_1"}]
        mock_lif.xml_root = None
        mock_lif_patch.return_value = mock_lif

        reader = Reader(image="dummy_path.lif")
        reader._metadata = metadata
        reader._dims = dimensions.Dimensions(dims, shape)

        return reader


@pytest.mark.parametrize(
    "reader_source, expected_values",
    [
        # Mock Reader A (This is from a real 800GB file)
        (
            {
                "type": "mock",
                "metadata_file": RESOURCE_DIR / "sample_lif_metadata.xml",
                "dims": "MTCZYX",
                "shape": (9, 49, 2, 1, 2048, 2048),
            },
            {
                "row": "B",
                "column": "2",
                "binning": "1x1",
                "objective": "HC PL FLUOTAR    10x/0.30 DRY",
                "total_time_duration": datetime.timedelta(seconds=172801),
                "timelapse_interval": datetime.timedelta(seconds=3600.0),
                "imaging_datetime": datetime.datetime(2024, 11, 22, 22, 25, 18, 897001),
            },
        ),
        # Real Reader 1 #TODO: add test file with more embedded metadata
        (
            {
                "type": "real",
                "path": RESOURCE_DIR / "s_1_t_1_c_2_z_1.lif",
            },
            {
                "imaging_datetime": datetime.datetime(2020, 3, 11, 19, 48, 24, 472000),
            },
        ),
    ],
)
def test_metadata_properties(
    reader_source: Dict[str, Any], expected_values: Dict[str, Any]
) -> None:
    # Setup reader
    if reader_source["type"] == "mock":
        reader = make_mock_reader(
            metadata_file=reader_source["metadata_file"],
            dims=reader_source["dims"],
            shape=reader_source["shape"],
        )
    elif reader_source["type"] == "real":
        reader = Reader(reader_source["path"])
    else:
        raise ValueError(f"Unknown reader type: {reader_source['type']}")

    # Test each property
    for property_name, expected in expected_values.items():
        result = getattr(reader, property_name)
        if isinstance(expected, datetime.timedelta):
            assert isinstance(
                result, datetime.timedelta
            ), f"{property_name} should be a timedelta"
            assert (
                abs(result.total_seconds() - expected.total_seconds()) < 1e-1
            ), f"{property_name} expected {expected}, got {result}"
        else:
            assert (
                result == expected
            ), f"{property_name} expected {expected}, got {result}"
