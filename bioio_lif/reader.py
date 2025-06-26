#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
import xml.etree.ElementTree as ET
from copy import copy
from dataclasses import dataclass
from math import floor
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import constants, dimensions, exceptions, io, reader, transforms, types
from bioio_base.standard_metadata import StandardMetadata
from dask import delayed
from fsspec.spec import AbstractFileSystem
from readlif.reader import LifFile

from .io import search_for_node

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass
class WellPosition:
    row: str
    col: str


class Reader(reader.Reader):
    """
    Wraps the readlif API to provide a bioio reader plugin for
    volumetric LIF images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    chunk_dims: Union[str, List[str]]
        Which dimensions to create chunks for.
        Default: dimensions.DEFAULT_CHUNK_DIMS
        Note: Dimensions.SpatialY, Dimensions.SpatialX, and
        dimensions.DimensionNames.Samples, will always be added
        to the list if not present during dask array construction.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    is_x_flipped: bool, default = False
        If True, the respective dimension will be
        flipped along the other axis.
    is_y_flipped: bool, default = False
        If True, the respective dimension will be
        flipped along the other axis.
    is_x_and_y_swapped:bool, default = True
        If `is_x_and_y_swapped` is True, the field_x and field_y given
        from the mosaic_position will be swapped such that field_x represents y
        and field_y represents x.
    Notes
    -----
    To use this reader, install with: `pip install readlif>=0.6.4`.

    readlif is licensed under GPLv3 and is not included in this package.
    """

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_dask_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[dimensions.Dimensions] = None
    _metadata: Optional[Any] = None
    _physical_pixel_sizes: Optional[types.PhysicalPixelSizes] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    # Do not provide default value because
    # they may not need to be used by your reader (i.e. input param is an array)
    _fs: "AbstractFileSystem"
    _path: str

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with fs.open(path) as open_resource:
                LifFile(open_resource)
                return True

        except Exception as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-lif",
                path,
                "file is not supported: " + str(e),
            )

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = dimensions.DEFAULT_CHUNK_DIMS,
        fs_kwargs: Dict[str, Any] = {},
        is_x_flipped: bool = False,
        is_y_flipped: bool = False,
        is_x_and_y_swapped: bool = True,
    ):
        # Expand details of provided image
        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        self.chunk_dims = chunk_dims

        # If either of these are True, the respective dimension will be
        # flipped along the other axis.
        # Ex. if `is_x_flipped = True` in a 4x4 tiled space, coordinate (2, 3)
        # will be (1, 3)
        # Ex. where both are true in a 4x4 tiled space, coordinate (1, 0)
        # will be (2, 3)
        # from the mosaic_position will be swapped such that field_x represents y
        # and field_y represents x.
        self.is_x_flipped = is_x_flipped
        self.is_y_flipped = is_y_flipped

        # If `is_x_and_y_swapped` is True, the field_x and field_y given
        # from the mosaic_position will be swapped such that field_x represents y
        # and field_y represents x.
        self.is_x_and_y_swapped = is_x_and_y_swapped

        # Delayed storage
        self._scene_short_info: Dict[str, Any] = {}

        # Enforce valid image
        self._is_supported_image(self._fs, self._path)

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            with self._fs.open(self._path) as open_resource:
                lif = LifFile(open_resource)
                self._scenes = tuple(image["name"] for image in lif.image_list)
        return self._scenes

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        retrieve_dims: List[str],
        retrieve_indices: List[Optional[int]],
    ) -> np.ndarray:
        """
        Open a file for reading, construct a Zarr store, select data, and compute to
        numpy.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to file to read.
        scene: int
            The scene index to pull the chunk from.
        retrieve_dims: List[str]
            The order of the retrieve indicies operations
        retrieve_indices: List[Optional[int]],
            The image index operations to retrieve.
            If None, retrieve the whole dimension.

        Returns
        -------
        chunk: np.ndarray
            The image chunk as a numpy array.
        """
        MISSING_DIM_SENTINAL_VALUE = -1

        # Open and select the target image
        with fs.open(path) as open_resource:
            selected_scene = LifFile(open_resource).get_image(scene)

            # Create the fill array shape
            # Drop the YX as we will be pulling the individual YX planes
            retrieve_shape: List[int] = []
            use_selected_or_np_map: Dict[str, int] = {}
            for dim, index_op in zip(retrieve_dims, retrieve_indices):
                if dim not in [
                    dimensions.DimensionNames.SpatialY,
                    dimensions.DimensionNames.SpatialX,
                ]:
                    # Handle slices
                    if index_op is None:
                        # Store the dim for later to inform to use the np index
                        use_selected_or_np_map[dim] = MISSING_DIM_SENTINAL_VALUE
                        if dim == dimensions.DimensionNames.MosaicTile:
                            retrieve_shape.append(selected_scene.n_mosaic)
                        elif dim == dimensions.DimensionNames.Time:
                            retrieve_shape.append(selected_scene.nt)
                        elif dim == dimensions.DimensionNames.Channel:
                            retrieve_shape.append(selected_scene.channels)
                        elif dim == dimensions.DimensionNames.SpatialZ:
                            retrieve_shape.append(selected_scene.nz)

                    # Handle non-chunk dimensions (specific indices / ints)
                    else:
                        # Store the dim for later to inform to use the provided index
                        use_selected_or_np_map[dim] = index_op
                        retrieve_shape.append(1)

            # Create list of planes that we will add each plane to, later we reshape
            # Create empty arr with the desired shape to enumerate over the np index
            planes: List[np.ndarray] = []
            np_array_for_indices = np.empty(tuple(retrieve_shape), dtype=object)
            for np_index, _ in np.ndenumerate(np_array_for_indices):
                # Get each plane's index selection operations
                # If the dimension is None, use the enumerated np index
                # If the dimension is not None, use the provided value
                plane_indices: Dict[str, int] = {}

                # Handle MosaicTile
                if dimensions.DimensionNames.MosaicTile in use_selected_or_np_map:
                    if (
                        use_selected_or_np_map[dimensions.DimensionNames.MosaicTile]
                        == MISSING_DIM_SENTINAL_VALUE
                    ):
                        plane_indices["m"] = np_index[
                            retrieve_dims.index(dimensions.DimensionNames.MosaicTile)
                        ]
                    else:
                        plane_indices["m"] = use_selected_or_np_map[
                            dimensions.DimensionNames.MosaicTile
                        ]

                # Handle Time
                if (
                    use_selected_or_np_map[dimensions.DimensionNames.Time]
                    == MISSING_DIM_SENTINAL_VALUE
                ):
                    plane_indices["t"] = np_index[
                        retrieve_dims.index(dimensions.DimensionNames.Time)
                    ]
                else:
                    plane_indices["t"] = use_selected_or_np_map[
                        dimensions.DimensionNames.Time
                    ]

                # Handle Channels
                if (
                    use_selected_or_np_map[dimensions.DimensionNames.Channel]
                    == MISSING_DIM_SENTINAL_VALUE
                ):
                    plane_indices["c"] = np_index[
                        retrieve_dims.index(dimensions.DimensionNames.Channel)
                    ]
                else:
                    plane_indices["c"] = use_selected_or_np_map[
                        dimensions.DimensionNames.Channel
                    ]

                # Handle SpatialZ
                if (
                    use_selected_or_np_map[dimensions.DimensionNames.SpatialZ]
                    == MISSING_DIM_SENTINAL_VALUE
                ):
                    plane_indices["z"] = np_index[
                        retrieve_dims.index(dimensions.DimensionNames.SpatialZ)
                    ]
                else:
                    plane_indices["z"] = use_selected_or_np_map[
                        dimensions.DimensionNames.SpatialZ
                    ]

                # Append the retrieved plane as a numpy array
                planes.append(np.asarray(selected_scene.get_frame(**plane_indices)))

            # Stack and reshape to get rid of the array of arrays
            scene_dims = selected_scene.info["dims"]
            retrieved_chunk = np.stack(planes).reshape(
                np_array_for_indices.shape + (scene_dims.y, scene_dims.x)
            )

            # Remove extra dimensions if they were not requested
            remove_dim_ops_list: List[Union[int, slice]] = []
            for index in retrieve_indices:
                if isinstance(index, int):
                    remove_dim_ops_list.append(0)
                else:
                    remove_dim_ops_list.append(slice(None, None, None))

            # Remove extra dimensions by using dim ops
            retrieved_chunk = retrieved_chunk[tuple(remove_dim_ops_list)]

            return retrieved_chunk

    def _create_dask_array(
        self, lif: LifFile, selected_scene_dims: List[str]
    ) -> xr.DataArray:
        """
        Creates a delayed dask array for the file.

        Parameters
        ----------
        lif: LifFile
            An open LifFile for processing.
        selected_scene_dims: List[str]
            The dimensions for the scene to create the dask array for

        Returns
        -------
        image_data: da.Array
            The fully constructed and fully delayed image as a Dask Array object.
        """
        # Always add the plane dimensions if not present already
        for dim in dimensions.REQUIRED_CHUNK_DIMS:
            if dim not in self.chunk_dims:
                self.chunk_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_dims = [d.upper() for d in self.chunk_dims]

        # Construct the delayed dask array
        selected_scene = lif.get_image(self.current_scene_index)
        selected_scene_shape: List[int] = []
        for dim in selected_scene_dims:
            if dim == dimensions.DimensionNames.MosaicTile:
                selected_scene_shape.append(selected_scene.n_mosaic)
            elif dim == dimensions.DimensionNames.Time:
                selected_scene_shape.append(selected_scene.nt)
            elif dim == dimensions.DimensionNames.Channel:
                selected_scene_shape.append(selected_scene.channels)
            elif dim == dimensions.DimensionNames.SpatialZ:
                selected_scene_shape.append(selected_scene.nz)
            elif dim == dimensions.DimensionNames.SpatialY:
                selected_scene_shape.append(selected_scene.info["dims"].y)
            elif dim == dimensions.DimensionNames.SpatialX:
                selected_scene_shape.append(selected_scene.info["dims"].x)

        # Get sample for dtype
        sample_plane = np.asarray(selected_scene.get_frame())

        # Constuct the chunk and non-chunk shapes one dim at a time
        # We also collect the chunk and non-chunk dimension order so that
        # we can swap the dimensions after we block out the array
        non_chunk_dim_order = []
        non_chunk_shape = []
        chunk_dim_order = []
        chunk_shape = []
        for dim, size in zip(selected_scene_dims, selected_scene_shape):
            if dim in self.chunk_dims:
                chunk_dim_order.append(dim)
                chunk_shape.append(size)

            else:
                non_chunk_dim_order.append(dim)
                non_chunk_shape.append(size)

        # Fill out the rest of the blocked shape with dimension sizes of 1 to
        # match the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to
        # outer-most with the chunks as long as the dimension is size 1
        blocked_dim_order = non_chunk_dim_order + chunk_dim_order
        blocked_shape = tuple(non_chunk_shape) + ((1,) * len(chunk_shape))

        # Make ndarray for lazy arrays to fill
        lazy_arrays: np.ndarray = np.ndarray(blocked_shape, dtype=object)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            # All dimensions get their normal index except for chunk dims
            # which get None, which tell the get data func to pull the whole dim
            retrieve_indices = np_index[: len(non_chunk_shape)] + (
                (None,) * len(chunk_shape)
            )

            # Fill the numpy array with the delayed arrays
            lazy_arrays[np_index] = da.from_delayed(
                delayed(Reader._get_image_data)(
                    fs=self._fs,
                    path=self._path,
                    scene=self.current_scene_index,
                    retrieve_dims=blocked_dim_order,
                    retrieve_indices=retrieve_indices,
                ),
                shape=chunk_shape,
                dtype=sample_plane.dtype,
            )

        # Convert the numpy array of lazy readers into a dask array
        image_data = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example, if the original dimension ordering was "TZYX" and we
        # chunked by "T", "Y", and "X"
        # we created an array with dimensions ordering "ZTYX"
        transpose_indices = []
        for i, d in enumerate(selected_scene_dims):
            new_index = blocked_dim_order.index(d)
            if new_index != i:
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Only run if the transpose is actually required
        image_data = da.transpose(image_data, tuple(transpose_indices))

        return image_data

    @staticmethod
    def _get_coords_and_physical_px_sizes(
        xml: ET.Element, image_short_info: Dict[str, Any], scene_index: int
    ) -> Tuple[Dict[str, Any], types.PhysicalPixelSizes]:
        # Create coord dict
        coords: Dict[str, Any] = {}

        # Get all images
        img_sets = xml.findall(".//Image")

        # Select the current scene
        img = img_sets[scene_index]

        # Construct channel list
        scene_channel_list = []
        channels = img.findall(".//ChannelDescription")
        channel_details = img.findall(".//WideFieldChannelInfo")
        for i, channel in enumerate(channels):
            if len(channels) <= len(channel_details):
                channel_detail = channel_details[i]
                scene_channel_list.append(
                    (
                        f"{channel_detail.attrib['LUT']}"
                        f"--{channel_detail.attrib['ContrastingMethodName']}"
                        f"--{channel_detail.attrib['FluoCubeName']}"
                    )
                )
            else:
                scene_channel_list.append(f"{channel.attrib['LUTName']}")

        # Attach channel names to coords
        coords[dimensions.DimensionNames.Channel] = scene_channel_list

        # Unpack short info scales
        scale_x, scale_y, scale_z, scale_t = image_short_info["scale"]

        # Scales from readlif are returned as px/µm
        # We want to return as µm/px
        scale_x = 1 / scale_x if scale_x is not None else None
        scale_y = 1 / scale_y if scale_y is not None else None
        scale_z = 1 / scale_z if scale_z is not None else None

        # Handle Spatial Dimensions
        if scale_z is not None:
            coords[dimensions.DimensionNames.SpatialZ] = Reader._generate_coord_array(
                0, image_short_info["dims"].z, scale_z
            )
        if scale_y is not None:
            coords[dimensions.DimensionNames.SpatialY] = Reader._generate_coord_array(
                0, image_short_info["dims"].y, scale_y
            )
        if scale_x is not None:
            coords[dimensions.DimensionNames.SpatialX] = Reader._generate_coord_array(
                0, image_short_info["dims"].x, scale_x
            )

        # Time
        if scale_t is not None:
            coords[dimensions.DimensionNames.Time] = Reader._generate_coord_array(
                0, image_short_info["dims"].t, scale_t
            )

        # Create physical pixal sizes
        px_sizes = types.PhysicalPixelSizes(scale_z, scale_y, scale_x)

        return coords, px_sizes

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray object.
            Metadata is attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            lif = LifFile(open_resource)
            selected_scene = lif.get_image(self.current_scene_index)
            self._scene_short_info = selected_scene.info

            # Check for mosaic tiles
            tile_positions = self._scene_short_info["mosaic_position"]

            # If there are tiles in the image use mosaic dims
            if len(tile_positions) > 0:
                dims = dimensions.DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES

            # Otherwise use standard dims
            else:
                dims = dimensions.DEFAULT_DIMENSION_ORDER_LIST

            # Get image data
            image_data = self._create_dask_array(lif, dims)

            # Get metadata
            meta = lif.xml_root

            # Create coordinate planes
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                image_short_info=self._scene_short_info,
                scene_index=self.current_scene_index,
            )

            # Store pixel sizes
            self._physical_pixel_sizes = px_sizes

            return xr.DataArray(
                image_data,
                dims=dims,
                coords=coords,
                attrs={constants.METADATA_UNPROCESSED: meta},
            )

    def _read_immediate(self) -> xr.DataArray:
        """
        Construct the in-memory xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully read into memory image as a DataArray
            object. Metadata is attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            lif = LifFile(open_resource)
            selected_scene = lif.get_image(self.current_scene_index)
            self._scene_short_info = selected_scene.info

            # Check for mosaic tiles
            tile_positions = self._scene_short_info["mosaic_position"]

            # If there are tiles in the image use mosaic dims
            if len(tile_positions) > 0:
                dims = dimensions.DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES

            # Otherwise use standard dims
            else:
                dims = dimensions.DEFAULT_DIMENSION_ORDER_LIST

            # Get image data
            image_data = self._get_image_data(
                fs=self._fs,
                path=self._path,
                scene=self.current_scene_index,
                retrieve_dims=dims,
                retrieve_indices=[None] * len(dims),  # Get all planes
            )

            # Get metadata
            meta = lif.xml_root

            # Create coordinate planes
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                image_short_info=self._scene_short_info,
                scene_index=self.current_scene_index,
            )

            # Store pixel sizes
            self._physical_pixel_sizes = px_sizes

            return xr.DataArray(
                image_data,
                dims=dims,
                coords=coords,
                attrs={constants.METADATA_UNPROCESSED: meta},
            )

    def _stitch_tiles(
        self,
        data: types.ArrayLike,
        dims: str,
        mosaic_position: List[Tuple[int, int, float, float]],
    ) -> types.ArrayLike:
        """
        This uses the mosaic_position of the LIF file to index into the data array,
        retrieve the tile, transform it, and then recreate the XY plane of the tiles
        before eventually combining them back together into one array (representing
        the stitched mosaic image).

        This stitching expects LIF files to have an extra pixel of overlap between tiles
        and will shave off those pixels.

        The X and Y coordinates may need to be flipped or swapped, this information is
        stored in the LIF file.

        Returns
        -------
        mosaic image: types.ArrayLike
            The previously seperate tiles as one stitched image
        """
        # Prefill a 2D list representing the XY plane
        number_of_rows, number_of_columns = self._get_yx_tile_count()
        xy_plane = np.zeros((number_of_rows, number_of_columns), dtype=object)

        # Iterate over each mosaic_position coordinate using the relative
        # field position (XY coordinate) given to retrieve each tile from
        # the data array, transform it, and put back into a 2D (XY) array
        for tile_index, tile_position, *_ in enumerate(mosaic_position):
            # Get tile by getting all data for specific M
            tile = transforms.reshape_data(
                data,
                given_dims=dims,
                return_dims=dims.replace(dimensions.DimensionNames.MosaicTile, ""),
                M=tile_index,
            )

            column_index, row_index, *_ = tile_position
            if self.is_x_and_y_swapped:
                column_index, row_index = row_index, column_index

            # LIF image stitching has a 1 pixel overlap;
            # Drop the first pixel unless this is the last tile for that dimension
            is_last_row = row_index + 1 >= number_of_rows
            is_last_column = column_index + 1 >= number_of_columns
            if not is_last_row:
                tile = tile[:, :, :, 1:, :]

            if not is_last_column:
                tile = tile[:, :, :, :, 1:]

            # LIF tiles are packed starting from bottom right so
            # the origin (0, 0) needs to be the bottom right of the grid
            # i.e. the end of the array hence the negative indexing
            xy_plane[-(row_index + 1), -(column_index + 1)] = tile

        # LIF files can have their X or Y coordinates flipped or even
        # swapped, this information is stored in their metadata.
        if self.is_x_flipped:
            xy_plane = np.fliplr(xy_plane)
        if self.is_y_flipped:
            xy_plane = np.flipud(xy_plane)

        # Concatenate plane into singular mosaic image
        rows = [np.concatenate(row_as_tiles, axis=-1) for row_as_tiles in xy_plane]
        return np.concatenate(rows, axis=-2)

    def _construct_mosaic_xarray(self, data: types.ArrayLike) -> xr.DataArray:
        # Get max of mosaic positions from lif
        with self._fs.open(self._path) as open_resource:
            lif = LifFile(open_resource)
            selected_scene = lif.get_image(self.current_scene_index)

        # Stitch
        stitched = self._stitch_tiles(
            data=data,
            dims=self.dims.order,
            mosaic_position=selected_scene.mosaic_position,
        )

        # Copy metadata
        dims = [
            d
            for d in self.xarray_dask_data.dims
            if d is not dimensions.DimensionNames.MosaicTile
        ]
        coords: Dict[Hashable, Any] = {
            d: v
            for d, v in self.xarray_dask_data.coords.items()
            if d
            not in [
                dimensions.DimensionNames.MosaicTile,
                dimensions.DimensionNames.SpatialY,
                dimensions.DimensionNames.SpatialX,
            ]
        }

        # Add expanded Y and X coords
        scale_x, scale_y, _, _ = selected_scene.info["scale"]
        scale_x = 1 / scale_x if scale_x is not None else None
        scale_y = 1 / scale_y if scale_y is not None else None

        if scale_y is not None:
            coords[dimensions.DimensionNames.SpatialY] = Reader._generate_coord_array(
                0, stitched.shape[-2], scale_y
            )
        if scale_x is not None:
            coords[dimensions.DimensionNames.SpatialX] = Reader._generate_coord_array(
                0, stitched.shape[-1], scale_x
            )

        attrs = copy(self.xarray_dask_data.attrs)

        return xr.DataArray(
            data=stitched,
            dims=dims,
            coords=coords,
            attrs=attrs,
        )

    def _get_yx_tile_count(self) -> Tuple[int, int]:
        """
        Get the number of tiles along the Y and X axis respectively.

        Ex. Y = 3, X = 4 would mean the YX plane looks something like:
        - - - -
        - - - -
        - - - -
        while Y = 3, X = 2 would be:
        _ _
        _ _
        _ _

        Returns
        -------
        Y dimension length: int
            The number of tiles along the Y axis.
        X dimension length: int
            The number of tiles along the X axis.
        """
        # Determine the length of the x dimension (i.e. number of columns in XY plane)
        x_dim_length = 1
        for x, *_ in self._scene_short_info["mosaic_position"]:
            if x + 1 > x_dim_length:
                x_dim_length = x + 1

        # The length of the mosaic_position array == X * Y so
        # Y = (X * Y) / X
        y_dim_length = floor(
            len(self._scene_short_info["mosaic_position"]) / x_dim_length
        )

        if self.is_x_and_y_swapped:
            y_dim_length, x_dim_length = x_dim_length, y_dim_length

        return y_dim_length, x_dim_length

    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        return self._construct_mosaic_xarray(self.dask_data)

    def _get_stitched_mosaic(self) -> xr.DataArray:
        return self._construct_mosaic_xarray(self.data)

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.

        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        if self._physical_pixel_sizes is None:
            # We get pixel sizes as a part of array construct
            # so simply run array construct
            self.dask_data

        if self._physical_pixel_sizes is None:
            raise ValueError("Pixel sizes weren't created as a part of image reading")

        return self._physical_pixel_sizes

    def get_mosaic_tile_position(
        self, mosaic_tile_index: int, **kwargs: int
    ) -> Tuple[int, int]:
        """
        Get the absolute position of the top left point for a single mosaic tile.
        Not equivalent to readlif's notion of mosaic_position.

        Parameters
        ----------
        mosaic_tile_index: int
            The index for the mosaic tile to retrieve position information for.
        kwargs: int
            The keywords below allow you to specify the dimensions that you wish
            to match. If you under-specify the constraints you can easily
            end up with a massive image stack.
                       Z = 1   # The Z-dimension.
                       C = 2   # The C-dimension ("channel").
                       T = 3   # The T-dimension ("time").

        Returns
        -------
        top: int
            The Y coordinate for the tile position.
        left: int
            The X coordinate for the tile position.

        Raises
        ------
        UnexpectedShapeError
            The image has no mosaic dimension available.
        IndexError
            No matching mosaic tile index found.
        """
        if dimensions.DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError("No mosaic dimension in image.")

        if kwargs:
            raise NotImplementedError(
                "Selecting mosaic positions by dimensions is not supporting "
                + "by LifReader. Retrieve a specific mosaic position via the "
                + "mosaic tile index (M) by using .get_mosaic_tile_position() instead."
            )

        # LIFs are packed from bottom right to top left
        # To counter: subtract 1 + M from list index to get from back of list
        index_x, index_y, _, _ = self._scene_short_info["mosaic_position"][
            -(mosaic_tile_index + 1)
        ]
        y_dim_length, x_dim_length = self._get_yx_tile_count()

        if self.is_x_and_y_swapped:
            index_x, index_y = index_y, index_x
        if self.is_x_flipped:
            index_x = x_dim_length - index_x
        if self.is_y_flipped:
            index_y = y_dim_length - index_y

        # Formula: (Dim position * Tile dim length) - Dim position
        # where the "- Dim position" is to account for shaving a pixel off
        # of each tile to account for overlap
        return (
            (index_y * self.dims.Y) - index_y,
            (index_x * self.dims.X) - index_x,
        )

    def get_mosaic_tile_positions(self, **kwargs: int) -> List[Tuple[int, int]]:
        """
        Get the absolute positions of the top left points for each mosaic tile
        matching the specified dimensions and current scene.

        Parameters
        ----------
        kwargs: int
            The keywords below allow you to specify the dimensions that you wish
            to match. If you under-specify the constraints you can easily
            end up with a massive image stack.
                       Z = 1   # The Z-dimension.
                       C = 2   # The C-dimension ("channel").
                       T = 3   # The T-dimension ("time").

        Returns
        -------
        mosaic_tile_positions: List[Tuple[int, int]]
            List of the Y and X coordinate for the tile positions.

        Raises
        ------
        UnexpectedShapeError
            The image has no mosaic dimension available.
        NotImplementedError
            This reader does not support indexing tiles by dimensions other than M
        """
        if dimensions.DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError("No mosaic dimension in image.")

        if kwargs:
            raise NotImplementedError(
                "Selecting mosaic positions by dimensions is not supporting "
                + "by LifReader. Retrieve a specific mosaic position via the "
                + "mosaic tile index (M) by using .get_mosaic_tile_position() instead."
            )

        mosaic_positions: List[Tuple[int, int, float, float]] = self._scene_short_info[
            "mosaic_position"
        ]

        # LIFs are packed from bottom right to top left
        # To counter: read the positions in reverse
        adjusted_mosaic_positions: List[Tuple[int, int]] = []
        y_dim_length, x_dim_length = self._get_yx_tile_count()
        for x, y, *_ in reversed(mosaic_positions):
            if self.is_x_and_y_swapped:
                x, y = y, x
            if self.is_x_flipped:
                x = x_dim_length - x
            if self.is_y_flipped:
                y = y_dim_length - y

            # Formula: (Dim position * Tile dim length) - Dim position
            # where the "- Dim position" is to account for shaving a pixel off
            # of each tile to account for overlap
            adjusted_mosaic_positions.append(
                (
                    (y * self.dims.Y) - y,
                    (x * self.dims.X) - x,
                )
            )

        return adjusted_mosaic_positions

    @staticmethod
    def _convert_lif_timestamp(hex_timestamp: str) -> datetime.datetime:
        """
        Converts a LIF hexadecimal timestamp to a UTC datetime.

        Parameters
        ----------
        hex_timestamp : str
            A Windows FILETIME-format timestamp in hex (e.g., '1db3d2d696a7a10').

        Returns
        -------
        datetime.datetime
            The corresponding UTC datetime object.
        """
        timestamp_as_complex_decimal = int(hex_timestamp, 16)
        nanoseconds_since_filetime_epoch = timestamp_as_complex_decimal * 100
        seconds_since_filetime_epoch = nanoseconds_since_filetime_epoch / 1e9
        seconds_since_unix_epoch = seconds_since_filetime_epoch - 11644473600
        return datetime.datetime.utcfromtimestamp(seconds_since_unix_epoch)

    @property
    def _scene_to_well_map(self) -> Dict[int, WellPosition]:
        if not hasattr(self, "__scene_to_well_map"):
            scene_to_well_map = {}
            index = 0
            for row_elem in self.metadata.findall(".//Element"):
                row_name = row_elem.attrib.get("Name")
                if not row_name or not row_name.isalpha():
                    continue
                for col_elem in row_elem.findall("./Children/Element"):
                    col_name = col_elem.attrib.get("Name")
                    if col_name:
                        scene_to_well_map[index] = WellPosition(
                            row=row_name.upper(), col=col_name
                        )
                        index += 1
            self.__scene_to_well_map = scene_to_well_map
        return self.__scene_to_well_map

    @property
    def scene_root(self) -> ET.Element:
        """
        Returns the XML node corresponding to the current scene.

        Returns
        -------
        ET.Element
            The XML element representing the scene node based on row and column.

        Raises
        ------
        ValueError
            If either the row or column value is missing.
        """
        if self.row is None or self.column is None:
            raise ValueError(
                "Row or column value is missing; cannot locate the scene node."
            )

        row_node = search_for_node(self.metadata, "Element", {"Name": self.row})
        return search_for_node(row_node, "Element", {"Name": self.column})

    @property
    def column(self) -> Optional[str]:
        """
        Extracts the well column index from the current scene name.

        Returns
        -------
        Optional[str]
            The column index as a string. Returns None if parsing fails.
        """
        try:
            pos = self._scene_to_well_map.get(self.current_scene_index)
            return pos.col if pos else None
        except Exception as exc:
            log.warning("Failed to extract well column: %s", exc, exc_info=True)
            return None

    @property
    def row(self) -> Optional[str]:
        """
        Extracts the well row index from the current scene name.

        Returns
        -------
        Optional[str]
            The row index as a string. Returns None if parsing fails.
        """
        try:
            pos = self._scene_to_well_map.get(self.current_scene_index)
            return pos.row if pos else None
        except Exception as exc:
            log.warning("Failed to extract well row: %s", exc, exc_info=True)
            return None

    @property
    def binning(self) -> Optional[str]:
        """
        Extracts the binning setting from the scene metadata.

        Returns
        -------
        Optional[str]
            The binning setting (e.g., "1x1" or "2x2").
            Returns None if not found or unknown.
        """
        try:
            camera_format_node = search_for_node(self.scene_root, "CameraFormat")
            binning = camera_format_node.get("Binning")
            live_binning = camera_format_node.get("LiveBinning")
            if binning == "1" and live_binning == "1":
                return "1x1"
            if binning == "2" and live_binning == "2":
                return "2x2"
            raise NotImplementedError(
                f"Unexpected binning, found: {binning}x{live_binning}"
            )
        except Exception as exc:
            log.warning("Failed to extract binning: %s", exc, exc_info=True)
        return None

    @property
    def objective(self) -> Optional[str]:
        """
        Extracts the microscope objective magnification and NA.

        Returns
        -------
        Optional[str]
            The formatted objective (e.g., "10x/0.30"). Returns None if not found.
        """
        try:
            camera_settings_node = search_for_node(
                self.scene_root, "ATLCameraSettingDefinition"
            )
            objective_name = camera_settings_node.get("ObjectiveName")
            if objective_name is None:
                raise ValueError("ObjectiveName attribute not found")
            return objective_name.strip()
        except Exception as exc:
            log.warning("Failed to extract objective: %s", exc, exc_info=True)
        return None

    @property
    def total_time_duration(self) -> Optional[datetime.timedelta]:
        """
        Extracts the total time duration of the acquisition as a timedelta object.
        This is the interval from the start of the first timepoint to the
        start of the last timepoint.

        Returns
        -------
        Optional[timedelta]
            Total duration as a timedelta object.
            Returns None if timestamps are missing or invalid.
        """
        try:
            # Parse timestamps
            timestamp_list_node = search_for_node(self.scene_root, "TimeStampList")
            if timestamp_list_node is None or timestamp_list_node.text is None:
                raise ValueError("TimeStampList node not found or contains no text.")

            timestamps_str = timestamp_list_node.text.strip()
            if not timestamps_str:
                raise ValueError("TimeStampList node text is empty.")

            timestamps = timestamps_str.split(" ")
            if len(timestamps) < 2:
                raise ValueError("Not enough timestamps to compute duration.")

            # Extract dimensions with defaults
            m = getattr(self.dims, "M", 1)
            c = getattr(self.dims, "C", 1)
            z = getattr(self.dims, "Z", 1)
            t = getattr(self.dims, "T", 1)

            images_per_timepoint = m * c * z
            expected_total_timestamps = images_per_timepoint * t

            if images_per_timepoint == 0:
                raise ValueError("Invalid dimension sizes (zero images per timepoint).")

            if len(timestamps) != expected_total_timestamps:
                raise ValueError(
                    f"Timestamp count {len(timestamps)} does not match "
                    f"expected {expected_total_timestamps} based on "
                    f"dims M={m}, C={c}, Z={z}, T={t}."
                )

            # Compute start of first and last timepoints
            start_of_first = self._convert_lif_timestamp(timestamps[0])
            start_of_last = self._convert_lif_timestamp(
                timestamps[-images_per_timepoint]
            )

            total_duration = start_of_last - start_of_first

            return total_duration

        except Exception as exc:
            log.warning("Failed to extract total time duration: %s", exc, exc_info=True)
            return None

    @property
    def timelapse_interval(self) -> Optional[datetime.timedelta]:
        """
        Average time between consecutive timepoints as a timedelta object.

        Returns
        -------
        Optional[timedelta]
            Average interval between timepoints.
            Returns None if total_time_duration is None or less than two timepoints.
        """
        if self.total_time_duration is None or self.dims.T < 2:
            return None
        return self.total_time_duration / (self.dims.T - 1)

    @property
    def imaging_datetime(self) -> Optional[datetime.datetime]:
        """
        Extracts the acquisition start date as a UTC datetime.

        Returns
        -------
        Optional[datetime.datetime]
            Acquisition start time. Returns None if unavailable or malformed.
        """
        try:
            timestamp_list_node = search_for_node(self.metadata, "TimeStampList")
            if timestamp_list_node is None:
                raise ValueError("TimeStampList node not found in metadata.")
            if timestamp_list_node.text is None:
                raise ValueError("TimeStampList node text is missing in metadata.")

            timestamps_str = timestamp_list_node.text.strip()
            if not timestamps_str:
                raise ValueError("TimeStampList node text is empty in metadata.")
            timestamps = timestamps_str.split(" ")

            return self._convert_lif_timestamp(timestamps[0])
        except Exception as exc:
            log.warning("Failed to extract imaging date: %s", exc, exc_info=True)
            return None

    @property
    def standard_metadata(self) -> StandardMetadata:
        """
        Return the standard metadata for this reader, updating specific fields.

        This implementation calls the base reader’s standard_metadata property
        via super() and then assigns the new values.
        """
        metadata = super().standard_metadata
        metadata.binning = self.binning
        metadata.column = self.column
        metadata.row = self.row
        metadata.objective = self.objective
        metadata.imaging_datetime = self.imaging_datetime
        metadata.total_time_duration = self.total_time_duration
        metadata.timelapse_interval = self.timelapse_interval
        metadata.position_index = self.current_scene_index

        return metadata
