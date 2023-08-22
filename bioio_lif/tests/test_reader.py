#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import numpy as np
import pytest
from bioio_base import dimensions, exceptions, test_utilities
from readlif.reader import LifFile

from bioio_lif import Reader

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "s_1_t_1_c_2_z_1.lif",
            "PEI_laminin_35k",
            ("PEI_laminin_35k",),
            (1, 2, 1, 2048, 2048),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"],
            (None, 0.32499999999999996, 0.32499999999999996),
        ),
        (
            "s_1_t_4_c_2_z_1.lif",
            "b2_001_Crop001_Resize001",
            ("b2_001_Crop001_Resize001",),
            (4, 2, 1, 614, 614),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Gray--TL-PH--EMP_BF", "Green--FLUO--GFP"],
            (None, 0.33914910277324634, 0.33914910277324634),
        ),
        (
            "tiled.lif",
            "TileScan_002",
            ("TileScan_002",),
            (165, 1, 4, 1, 512, 512),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
            ["Gray", "Red", "Green", "Cyan"],
            (None, 0.20061311154598827, 0.20061311154598827),
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "3d-cell-viewer.ome.tiff",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_lif_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=ET.Element,
    )


@pytest.mark.parametrize("filename", ["s_1_t_1_c_2_z_1.lif", "s_1_t_4_c_2_z_1.lif"])
@pytest.mark.parametrize("chunk_dims", ["ZYX", "TYX", "CYX"])
@pytest.mark.parametrize("get_dims", ["ZYX", "TYX"])
def test_sanity_check_correct_indexing(
    filename: str,
    chunk_dims: str,
    get_dims: str,
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Construct reader
    reader = Reader(uri, chunk_dims=chunk_dims, is_x_and_y_swapped=True)
    lif_img = LifFile(uri).get_image(0)

    # Pull a chunk from LifReader
    chunk_from_lif_reader = reader.get_image_dask_data(get_dims).compute()

    # Pull what should be the same chunk from LifImage
    planes = []
    reshape_values = []
    for dim in get_dims:
        dim_size = getattr(lif_img.info["dims"], dim.lower())
        reshape_values.append(dim_size)

        if dim not in ["Y", "X"]:
            for i in range(dim_size):
                planes.append(np.asarray(lif_img.get_frame(**{dim.lower(): i})))

    # Stack and reshape
    chunk_from_read_lif = np.stack(planes).reshape(tuple(reshape_values))

    # Compare
    np.testing.assert_array_equal(chunk_from_lif_reader, chunk_from_read_lif)


@pytest.mark.parametrize(
    "tiles_filename, " "stitched_filename, " "tiles_set_scene, " "stitched_set_scene, ",
    [
        (
            "tiled.lif",
            "merged-tiles.lif",
            "TileScan_002",
            "TileScan_002_Merging",
        ),
        # s_1_t_4_c_2_z_1.lif has no mosaic tiles
        pytest.param(
            "s_1_t_4_c_2_z_1.lif",
            "merged-tiles.lif",
            "b2_001_Crop001_Resize001",
            "TileScan_002_Merging",
            marks=pytest.mark.xfail(raises=exceptions.InvalidDimensionOrderingError),
        ),
    ],
)
def test_lif_reader_mosaic_stitching(
    tiles_filename: str,
    stitched_filename: str,
    tiles_set_scene: str,
    stitched_set_scene: str,
) -> None:
    # Construct full filepath
    tiles_uri = LOCAL_RESOURCES_DIR / tiles_filename
    stitched_uri = LOCAL_RESOURCES_DIR / stitched_filename

    # Construct reader
    tiles_reader = Reader(tiles_uri, is_x_and_y_swapped=True)
    stitched_reader = Reader(stitched_uri)

    # Run checks
    test_utilities.run_reader_mosaic_checks(
        tiles_reader=tiles_reader,
        stitched_reader=stitched_reader,
        tiles_set_scene=tiles_set_scene,
        stitched_set_scene=stitched_set_scene,
    )


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_tile_dims, "
    "select_tile_index, "
    "expected_tile_top_left",
    [
        (
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            0,
            (5110, 7154),
        ),
        (
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            50,
            (3577, 4599),
        ),
        (
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            3,
            (5110, 5621),
        ),
        (
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            164,
            (0, 0),
        ),
        pytest.param(
            "tiled.lif",
            "TileScan_002",
            (512, 512),
            999,
            None,
            marks=pytest.mark.xfail(raises=IndexError),
        ),
        pytest.param(
            "merged-tiles.lif",
            "TileScan_002_Merging",
            None,
            None,
            None,
            # Doesn't have mosaic tiles
            marks=pytest.mark.xfail(raises=AssertionError),
        ),
    ],
)
def test_lif_reader_mosaic_tile_inspection(
    filename: str,
    set_scene: str,
    expected_tile_dims: Tuple[int, int],
    select_tile_index: int,
    expected_tile_top_left: Tuple[int, int],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Construct reader
    reader = Reader(uri, is_x_and_y_swapped=True)
    reader.set_scene(set_scene)

    # Check basics
    assert reader.mosaic_tile_dims is not None
    assert reader.mosaic_tile_dims.Y == expected_tile_dims[0]
    assert reader.mosaic_tile_dims.X == expected_tile_dims[1]

    # Pull tile info for compare
    tile_y_pos, tile_x_pos = reader.get_mosaic_tile_position(select_tile_index)
    assert tile_y_pos == expected_tile_top_left[0]
    assert tile_x_pos == expected_tile_top_left[1]

    # Pull actual pixel data to compare
    tile_from_m_index = reader.get_image_dask_data(
        reader.dims.order.replace(dimensions.DimensionNames.MosaicTile, ""),
        M=select_tile_index,
    ).compute()
    tile_from_position = reader.mosaic_xarray_dask_data[
        :,
        :,
        :,
        tile_y_pos : (tile_y_pos + reader.mosaic_tile_dims.Y),
        tile_x_pos : (tile_x_pos + reader.mosaic_tile_dims.X),
    ].compute()

    # (sanity-check) Make sure they are the same shape before shaving pixels
    assert tile_from_position.shape == tile_from_m_index.shape

    # Remove the first Y and X pixels
    # The stitched tiles overlap each other by 1px each so this is just
    # ignoring what would be overlap / cleaned up
    tile_from_m_index = tile_from_m_index[:, :, :, 1:, 1:]
    tile_from_position = tile_from_position[:, :, :, 1:, 1:]

    # Assert equal
    np.testing.assert_array_equal(tile_from_m_index, tile_from_position)
