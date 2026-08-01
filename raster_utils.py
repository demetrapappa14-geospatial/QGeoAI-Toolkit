# -*- coding: utf-8 -*-

"""
QGeoAI Toolkit

Raster utility functions used by the plugin.

This module contains reusable GDAL-based tools for:

- Opening raster datasets
- Reading raster metadata
- Validating raster inputs
- Reading raster data in blocks
- Managing NoData values
- Creating GeoTIFF outputs
- Copying projection and georeferencing
- Calculating raster statistics
- Loading QGIS raster-layer sources
- Removing provider-specific source parameters

Author:
Dimitra Pappa
"""

import os
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from osgeo import gdal
except ImportError as error:
    raise ImportError(
        "GDAL could not be imported. "
        "QGeoAI Toolkit requires the GDAL Python bindings "
        "included with QGIS."
    ) from error


# Enable GDAL exceptions so errors are raised instead of
# being printed silently in the QGIS Python console.
gdal.UseExceptions()


Number = Union[int, float]

WindowTuple = Tuple[int, int, int, int]

BlockResult = Tuple[
    int,
    int,
    int,
    int,
    np.ndarray
]


# ==========================================================
# GENERAL PATH UTILITIES
# ==========================================================

def normalize_raster_source(source: str) -> str:
    """
    Normalize a raster source path.

    QGIS raster providers may append parameters to a source
    path using the pipe character.

    Example
    -------
    /data/sentinel.tif|layerid=0

    The GDAL dataset usually requires only:

    /data/sentinel.tif

    Parameters
    ----------
    source : str
        Raster source path.

    Returns
    -------
    str
        Clean raster source path.
    """

    if source is None:
        return ""

    source = str(source).strip()

    if "|" in source:
        source = source.split("|", 1)[0]

    return os.path.normpath(source)


def raster_layer_source(raster_layer) -> str:
    """
    Extract and normalize the source path of a QGIS raster layer.

    Parameters
    ----------
    raster_layer
        QgsRasterLayer object.

    Returns
    -------
    str
        Normalized raster source path.

    Raises
    ------
    ValueError
        If the layer is missing or has no valid source.
    """

    if raster_layer is None:
        raise ValueError(
            "No raster layer was provided."
        )

    if not hasattr(raster_layer, "source"):
        raise TypeError(
            "The provided object is not a valid QGIS raster layer."
        )

    source = normalize_raster_source(
        raster_layer.source()
    )

    if not source:
        raise ValueError(
            "The selected raster layer does not contain "
            "a readable source path."
        )

    return source


def ensure_parent_directory(path: str) -> None:
    """
    Create the parent directory of an output path when needed.

    Parameters
    ----------
    path : str
        Output file path.
    """

    if not path:
        raise ValueError(
            "The output path is empty."
        )

    absolute_path = os.path.abspath(path)

    parent_directory = os.path.dirname(
        absolute_path
    )

    if parent_directory and not os.path.exists(
        parent_directory
    ):
        os.makedirs(
            parent_directory,
            exist_ok=True
        )


def ensure_tif_extension(path: str) -> str:
    """
    Add a GeoTIFF extension if the path has none.

    Parameters
    ----------
    path : str
        Output raster path.

    Returns
    -------
    str
        Path ending in .tif or .tiff.
    """

    if not path:
        raise ValueError(
            "The output raster path is empty."
        )

    if path.lower().endswith(
        (".tif", ".tiff")
    ):
        return path

    return f"{path}.tif"


# ==========================================================
# GDAL DATASET OPENING
# ==========================================================

def open_raster(
    source: str,
    update: bool = False
):
    """
    Open a raster dataset with GDAL.

    Parameters
    ----------
    source : str
        Raster file path.

    update : bool, optional
        Open the dataset in update mode when True.

    Returns
    -------
    gdal.Dataset
        Open GDAL raster dataset.

    Raises
    ------
    FileNotFoundError
        If the raster does not exist.

    RuntimeError
        If GDAL cannot open the raster.
    """

    source = normalize_raster_source(source)

    if not source:
        raise ValueError(
            "The raster source path is empty."
        )

    if not os.path.exists(source):
        raise FileNotFoundError(
            f"Raster file does not exist:\n{source}"
        )

    access_mode = (
        gdal.GA_Update
        if update
        else gdal.GA_ReadOnly
    )

    dataset = gdal.Open(
        source,
        access_mode
    )

    if dataset is None:
        raise RuntimeError(
            f"GDAL could not open the raster:\n{source}"
        )

    if dataset.RasterCount < 1:
        dataset = None

        raise RuntimeError(
            "The raster does not contain any readable bands."
        )

    if dataset.RasterXSize < 1 or dataset.RasterYSize < 1:
        dataset = None

        raise RuntimeError(
            "The raster has invalid dimensions."
        )

    return dataset


def open_raster_layer(
    raster_layer,
    update: bool = False
):
    """
    Open a QGIS raster layer as a GDAL dataset.

    Parameters
    ----------
    raster_layer
        QgsRasterLayer object.

    update : bool, optional
        Open in update mode when True.

    Returns
    -------
    gdal.Dataset
        GDAL raster dataset.
    """

    source = raster_layer_source(
        raster_layer
    )

    return open_raster(
        source,
        update=update
    )


# ==========================================================
# RASTER INFORMATION
# ==========================================================

def get_raster_info(
    dataset
) -> Dict[str, object]:
    """
    Read the main metadata of a GDAL raster dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        Open GDAL dataset.

    Returns
    -------
    dict
        Raster dimensions, bands, projection,
        geotransform, NoData values and data types.
    """

    if dataset is None:
        raise ValueError(
            "The GDAL dataset is None."
        )

    band_count = dataset.RasterCount

    nodata_values = []

    data_types = []

    for band_index in range(
        1,
        band_count + 1
    ):
        band = dataset.GetRasterBand(
            band_index
        )

        nodata_values.append(
            band.GetNoDataValue()
        )

        data_types.append(
            band.DataType
        )

    geotransform = dataset.GetGeoTransform(
        can_return_null=True
    )

    projection = dataset.GetProjection()

    return {
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "band_count": band_count,
        "geotransform": geotransform,
        "projection": projection,
        "nodata_values": nodata_values,
        "data_types": data_types,
        "driver": (
            dataset.GetDriver().ShortName
            if dataset.GetDriver()
            else None
        )
    }


def validate_band_number(
    dataset,
    band_number: int
) -> None:
    """
    Validate a 1-based raster band number.

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    band_number : int
        1-based band number.

    Raises
    ------
    ValueError
        If the number is outside the raster band range.
    """

    if dataset is None:
        raise ValueError(
            "The GDAL dataset is None."
        )

    if not isinstance(
        band_number,
        int
    ):
        raise TypeError(
            "Raster band numbers must be integers."
        )

    if band_number < 1:
        raise ValueError(
            "Raster band numbers start from 1."
        )

    if band_number > dataset.RasterCount:
        raise ValueError(
            f"Band {band_number} does not exist. "
            f"The raster contains "
            f"{dataset.RasterCount} band(s)."
        )


def validate_band_numbers(
    dataset,
    band_numbers: Sequence[int]
) -> List[int]:
    """
    Validate several raster band numbers.

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    band_numbers : sequence of int
        1-based raster band numbers.

    Returns
    -------
    list of int
        Validated band numbers.
    """

    if not band_numbers:
        raise ValueError(
            "No raster bands were selected."
        )

    validated = []

    for band_number in band_numbers:
        value = int(band_number)

        validate_band_number(
            dataset,
            value
        )

        validated.append(value)

    return validated


def get_band_nodata(
    dataset,
    band_number: int
) -> Optional[Number]:
    """
    Return the NoData value of a raster band.

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    band_number : int
        1-based band number.

    Returns
    -------
    int, float or None
        Band NoData value.
    """

    validate_band_number(
        dataset,
        band_number
    )

    band = dataset.GetRasterBand(
        band_number
    )

    return band.GetNoDataValue()


def get_band_description(
    dataset,
    band_number: int
) -> str:
    """
    Return the text description of a raster band.

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    band_number : int
        1-based band number.

    Returns
    -------
    str
        Band description or a generated default name.
    """

    validate_band_number(
        dataset,
        band_number
    )

    band = dataset.GetRasterBand(
        band_number
    )

    description = (
        band.GetDescription()
        or ""
    ).strip()

    if not description:
        description = f"Band {band_number}"

    return description


# ==========================================================
# RASTER READING
# ==========================================================

def read_band(
    dataset,
    band_number: int,
    window: Optional[WindowTuple] = None,
    output_dtype=np.float32
) -> np.ndarray:
    """
    Read a raster band into a NumPy array.

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    band_number : int
        1-based band number.

    window : tuple, optional
        Window formatted as:
        x_offset, y_offset, width, height.

    output_dtype : numpy dtype, optional
        Desired NumPy output type.

    Returns
    -------
    numpy.ndarray
        Two-dimensional raster array.
    """

    validate_band_number(
        dataset,
        band_number
    )

    band = dataset.GetRasterBand(
        band_number
    )

    if window is None:
        array = band.ReadAsArray()

    else:
        x_offset, y_offset, width, height = (
            validate_window(
                dataset,
                window
            )
        )

        array = band.ReadAsArray(
            x_offset,
            y_offset,
            width,
            height
        )

    if array is None:
        raise RuntimeError(
            f"Could not read raster band {band_number}."
        )

    array = np.asarray(array)

    if output_dtype is not None:
        array = array.astype(
            output_dtype,
            copy=False
        )

    return array


def read_bands(
    dataset,
    band_numbers: Optional[Sequence[int]] = None,
    window: Optional[WindowTuple] = None,
    output_dtype=np.float32
) -> np.ndarray:
    """
    Read multiple bands into a 3D NumPy array.

    Output shape
    ------------
    rows, columns, bands

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    band_numbers : sequence of int, optional
        Bands to read. When omitted, all bands are read.

    window : tuple, optional
        x_offset, y_offset, width, height.

    output_dtype : numpy dtype, optional
        Desired NumPy output type.

    Returns
    -------
    numpy.ndarray
        Three-dimensional raster array.
    """

    if band_numbers is None:
        band_numbers = list(
            range(
                1,
                dataset.RasterCount + 1
            )
        )

    band_numbers = validate_band_numbers(
        dataset,
        band_numbers
    )

    arrays = []

    for band_number in band_numbers:
        array = read_band(
            dataset=dataset,
            band_number=band_number,
            window=window,
            output_dtype=output_dtype
        )

        arrays.append(array)

    return np.stack(
        arrays,
        axis=-1
    )


def validate_window(
    dataset,
    window: WindowTuple
) -> WindowTuple:
    """
    Validate and normalize a raster read window.

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    window : tuple
        x_offset, y_offset, width, height.

    Returns
    -------
    tuple
        Validated integer window.
    """

    if len(window) != 4:
        raise ValueError(
            "A raster window must contain exactly "
            "four values: x_offset, y_offset, width, height."
        )

    x_offset, y_offset, width, height = (
        int(value)
        for value in window
    )

    if x_offset < 0 or y_offset < 0:
        raise ValueError(
            "Raster window offsets cannot be negative."
        )

    if width < 1 or height < 1:
        raise ValueError(
            "Raster window width and height must be positive."
        )

    if x_offset + width > dataset.RasterXSize:
        raise ValueError(
            "Raster window exceeds the dataset width."
        )

    if y_offset + height > dataset.RasterYSize:
        raise ValueError(
            "Raster window exceeds the dataset height."
        )

    return (
        x_offset,
        y_offset,
        width,
        height
    )


# ==========================================================
# BLOCK PROCESSING
# ==========================================================

def iter_windows(
    width: int,
    height: int,
    block_width: int = 512,
    block_height: int = 512
) -> Generator[
    WindowTuple,
    None,
    None
]:
    """
    Generate raster windows for block processing.

    Parameters
    ----------
    width : int
        Raster width in pixels.

    height : int
        Raster height in pixels.

    block_width : int, optional
        Block width.

    block_height : int, optional
        Block height.

    Yields
    ------
    tuple
        x_offset, y_offset, width, height.
    """

    width = int(width)

    height = int(height)

    block_width = int(block_width)

    block_height = int(block_height)

    if width < 1 or height < 1:
        raise ValueError(
            "Raster dimensions must be positive."
        )

    if block_width < 1 or block_height < 1:
        raise ValueError(
            "Block dimensions must be positive."
        )

    for y_offset in range(
        0,
        height,
        block_height
    ):
        current_height = min(
            block_height,
            height - y_offset
        )

        for x_offset in range(
            0,
            width,
            block_width
        ):
            current_width = min(
                block_width,
                width - x_offset
            )

            yield (
                x_offset,
                y_offset,
                current_width,
                current_height
            )


def block_count(
    width: int,
    height: int,
    block_width: int = 512,
    block_height: int = 512
) -> int:
    """
    Calculate the total number of processing blocks.

    Parameters
    ----------
    width : int
        Raster width.

    height : int
        Raster height.

    block_width : int
        Processing block width.

    block_height : int
        Processing block height.

    Returns
    -------
    int
        Number of raster blocks.
    """

    columns = (
        int(width) + int(block_width) - 1
    ) // int(block_width)

    rows = (
        int(height) + int(block_height) - 1
    ) // int(block_height)

    return rows * columns


def iter_band_blocks(
    dataset,
    band_number: int,
    block_width: int = 512,
    block_height: int = 512,
    output_dtype=np.float32
) -> Generator[
    BlockResult,
    None,
    None
]:
    """
    Iterate through one raster band in blocks.

    Yields
    ------
    tuple
        x_offset, y_offset, width, height, array.
    """

    validate_band_number(
        dataset,
        band_number
    )

    for window in iter_windows(
        width=dataset.RasterXSize,
        height=dataset.RasterYSize,
        block_width=block_width,
        block_height=block_height
    ):
        x_offset, y_offset, width, height = (
            window
        )

        array = read_band(
            dataset=dataset,
            band_number=band_number,
            window=window,
            output_dtype=output_dtype
        )

        yield (
            x_offset,
            y_offset,
            width,
            height,
            array
        )


def iter_multiband_blocks(
    dataset,
    band_numbers: Optional[Sequence[int]] = None,
    block_width: int = 512,
    block_height: int = 512,
    output_dtype=np.float32
) -> Generator[
    BlockResult,
    None,
    None
]:
    """
    Iterate through several raster bands in blocks.

    Output arrays have shape:

    rows, columns, bands

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    band_numbers : sequence of int, optional
        Bands to read.

    block_width : int, optional
        Processing block width.

    block_height : int, optional
        Processing block height.

    output_dtype : numpy dtype, optional
        Desired NumPy output type.

    Yields
    ------
    tuple
        x_offset, y_offset, width, height, array.
    """

    if band_numbers is None:
        band_numbers = list(
            range(
                1,
                dataset.RasterCount + 1
            )
        )

    band_numbers = validate_band_numbers(
        dataset,
        band_numbers
    )

    for window in iter_windows(
        width=dataset.RasterXSize,
        height=dataset.RasterYSize,
        block_width=block_width,
        block_height=block_height
    ):
        x_offset, y_offset, width, height = (
            window
        )

        array = read_bands(
            dataset=dataset,
            band_numbers=band_numbers,
            window=window,
            output_dtype=output_dtype
        )

        yield (
            x_offset,
            y_offset,
            width,
            height,
            array
        )


# ==========================================================
# NODATA AND VALID-PIXEL MASKS
# ==========================================================

def nodata_mask(
    array: np.ndarray,
    nodata_value: Optional[Number] = None,
    include_nan: bool = True,
    include_inf: bool = True
) -> np.ndarray:
    """
    Create a Boolean mask of invalid pixels.

    True means that a pixel is invalid.

    Parameters
    ----------
    array : numpy.ndarray
        Raster data array.

    nodata_value : int, float or None
        NoData value.

    include_nan : bool
        Mask NaN values.

    include_inf : bool
        Mask positive and negative infinity.

    Returns
    -------
    numpy.ndarray
        Boolean invalid-pixel mask.
    """

    array = np.asarray(array)

    mask = np.zeros(
        array.shape,
        dtype=bool
    )

    if include_nan and np.issubdtype(
        array.dtype,
        np.floating
    ):
        mask |= np.isnan(array)

    if include_inf and np.issubdtype(
        array.dtype,
        np.floating
    ):
        mask |= np.isinf(array)

    if nodata_value is not None:

        if (
            isinstance(nodata_value, float)
            and np.isnan(nodata_value)
        ):
            if np.issubdtype(
                array.dtype,
                np.floating
            ):
                mask |= np.isnan(array)

        elif np.issubdtype(
            array.dtype,
            np.floating
        ):
            mask |= np.isclose(
                array,
                nodata_value,
                equal_nan=True
            )

        else:
            mask |= array == nodata_value

    return mask


def valid_data_mask(
    array: np.ndarray,
    nodata_value: Optional[Number] = None
) -> np.ndarray:
    """
    Create a mask in which True means a valid pixel.

    Parameters
    ----------
    array : numpy.ndarray
        Raster data array.

    nodata_value : int, float or None
        Raster NoData value.

    Returns
    -------
    numpy.ndarray
        Boolean valid-data mask.
    """

    return ~nodata_mask(
        array=array,
        nodata_value=nodata_value
    )


def multiband_valid_mask(
    array: np.ndarray,
    nodata_values: Optional[
        Sequence[Optional[Number]]
    ] = None
) -> np.ndarray:
    """
    Create a valid-pixel mask for a multiband array.

    A pixel is valid only when every selected band is valid.

    Parameters
    ----------
    array : numpy.ndarray
        Array shaped rows, columns, bands.

    nodata_values : sequence, optional
        One NoData value per band.

    Returns
    -------
    numpy.ndarray
        Two-dimensional Boolean valid-data mask.
    """

    array = np.asarray(array)

    if array.ndim != 3:
        raise ValueError(
            "The multiband array must have three dimensions: "
            "rows, columns, bands."
        )

    number_of_bands = array.shape[-1]

    if nodata_values is None:
        nodata_values = [
            None
        ] * number_of_bands

    if len(nodata_values) != number_of_bands:
        raise ValueError(
            "The number of NoData values must match "
            "the number of raster bands."
        )

    valid_mask = np.ones(
        array.shape[:2],
        dtype=bool
    )

    for band_index in range(
        number_of_bands
    ):
        valid_mask &= valid_data_mask(
            array=array[..., band_index],
            nodata_value=nodata_values[band_index]
        )

    return valid_mask


def replace_invalid_values(
    array: np.ndarray,
    replacement_value: Number,
    nodata_value: Optional[Number] = None
) -> np.ndarray:
    """
    Replace invalid raster values in a copy of an array.

    Parameters
    ----------
    array : numpy.ndarray
        Input raster array.

    replacement_value : int or float
        Replacement value.

    nodata_value : int, float or None
        Original NoData value.

    Returns
    -------
    numpy.ndarray
        Modified array.
    """

    output = np.array(
        array,
        copy=True
    )

    invalid = nodata_mask(
        output,
        nodata_value=nodata_value
    )

    output[invalid] = replacement_value

    return output


# ==========================================================
# OUTPUT DATASET CREATION
# ==========================================================

def create_geotiff(
    output_path: str,
    width: int,
    height: int,
    band_count: int = 1,
    data_type: int = gdal.GDT_Float32,
    geotransform=None,
    projection: Optional[str] = None,
    nodata_value: Optional[Number] = None,
    creation_options: Optional[
        Sequence[str]
    ] = None,
    overwrite: bool = True
):
    """
    Create an empty GeoTIFF dataset.

    Parameters
    ----------
    output_path : str
        Output GeoTIFF path.

    width : int
        Raster width.

    height : int
        Raster height.

    band_count : int, optional
        Number of raster bands.

    data_type : int, optional
        GDAL raster data type.

    geotransform : tuple, optional
        Six-value GDAL geotransform.

    projection : str, optional
        Projection WKT.

    nodata_value : int, float or None
        NoData value applied to all bands.

    creation_options : sequence of str, optional
        GDAL GeoTIFF creation options.

    overwrite : bool, optional
        Delete an existing output when True.

    Returns
    -------
    gdal.Dataset
        Writable output dataset.
    """

    output_path = ensure_tif_extension(
        output_path
    )

    ensure_parent_directory(
        output_path
    )

    width = int(width)

    height = int(height)

    band_count = int(band_count)

    if width < 1 or height < 1:
        raise ValueError(
            "Output raster dimensions must be positive."
        )

    if band_count < 1:
        raise ValueError(
            "The output raster must contain at least one band."
        )

    if overwrite and os.path.exists(
        output_path
    ):
        driver = gdal.GetDriverByName(
            "GTiff"
        )

        try:
            driver.Delete(
                output_path
            )
        except RuntimeError:
            os.remove(
                output_path
            )

    if os.path.exists(output_path):
        raise FileExistsError(
            f"The output raster already exists:\n{output_path}"
        )

    if creation_options is None:
        creation_options = [
            "TILED=YES",
            "COMPRESS=LZW",
            "BIGTIFF=IF_SAFER"
        ]

    driver = gdal.GetDriverByName(
        "GTiff"
    )

    if driver is None:
        raise RuntimeError(
            "The GDAL GeoTIFF driver is not available."
        )

    dataset = driver.Create(
        output_path,
        width,
        height,
        band_count,
        data_type,
        options=list(creation_options)
    )

    if dataset is None:
        raise RuntimeError(
            f"Could not create output GeoTIFF:\n{output_path}"
        )

    if geotransform is not None:
        dataset.SetGeoTransform(
            geotransform
        )

    if projection:
        dataset.SetProjection(
            projection
        )

    if nodata_value is not None:
        for band_index in range(
            1,
            band_count + 1
        ):
            band = dataset.GetRasterBand(
                band_index
            )

            band.SetNoDataValue(
                float(nodata_value)
            )

            band.Fill(
                float(nodata_value)
            )

    return dataset


def create_geotiff_like(
    reference_dataset,
    output_path: str,
    band_count: int = 1,
    data_type: int = gdal.GDT_Float32,
    nodata_value: Optional[Number] = None,
    creation_options: Optional[
        Sequence[str]
    ] = None,
    overwrite: bool = True
):
    """
    Create a GeoTIFF using the size and spatial reference
    of another raster dataset.

    Parameters
    ----------
    reference_dataset : gdal.Dataset
        Reference raster.

    output_path : str
        Output path.

    band_count : int
        Number of output bands.

    data_type : int
        GDAL data type.

    nodata_value : int, float or None
        Output NoData value.

    creation_options : sequence of str, optional
        GeoTIFF creation options.

    overwrite : bool
        Overwrite existing output.

    Returns
    -------
    gdal.Dataset
        Writable output dataset.
    """

    if reference_dataset is None:
        raise ValueError(
            "The reference raster dataset is None."
        )

    return create_geotiff(
        output_path=output_path,
        width=reference_dataset.RasterXSize,
        height=reference_dataset.RasterYSize,
        band_count=band_count,
        data_type=data_type,
        geotransform=reference_dataset.GetGeoTransform(
            can_return_null=True
        ),
        projection=reference_dataset.GetProjection(),
        nodata_value=nodata_value,
        creation_options=creation_options,
        overwrite=overwrite
    )


# ==========================================================
# RASTER WRITING
# ==========================================================

def write_array(
    dataset,
    array: np.ndarray,
    band_number: int = 1,
    x_offset: int = 0,
    y_offset: int = 0
) -> None:
    """
    Write a NumPy array to a GDAL raster band.

    Parameters
    ----------
    dataset : gdal.Dataset
        Writable raster dataset.

    array : numpy.ndarray
        Two-dimensional array.

    band_number : int, optional
        Output band number.

    x_offset : int, optional
        Pixel-column offset.

    y_offset : int, optional
        Pixel-row offset.
    """

    validate_band_number(
        dataset,
        band_number
    )

    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            "Only two-dimensional arrays can be written "
            "to a single raster band."
        )

    x_offset = int(x_offset)

    y_offset = int(y_offset)

    if x_offset < 0 or y_offset < 0:
        raise ValueError(
            "Raster write offsets cannot be negative."
        )

    rows, columns = array.shape

    if x_offset + columns > dataset.RasterXSize:
        raise ValueError(
            "The array exceeds the output raster width."
        )

    if y_offset + rows > dataset.RasterYSize:
        raise ValueError(
            "The array exceeds the output raster height."
        )

    band = dataset.GetRasterBand(
        band_number
    )

    result = band.WriteArray(
        array,
        xoff=x_offset,
        yoff=y_offset
    )

    if result != 0:
        raise RuntimeError(
            f"Could not write data to raster band "
            f"{band_number}."
        )


def write_multiband_array(
    dataset,
    array: np.ndarray,
    x_offset: int = 0,
    y_offset: int = 0
) -> None:
    """
    Write an array shaped rows, columns, bands.

    Parameters
    ----------
    dataset : gdal.Dataset
        Writable raster dataset.

    array : numpy.ndarray
        Three-dimensional raster array.

    x_offset : int
        Output x offset.

    y_offset : int
        Output y offset.
    """

    array = np.asarray(array)

    if array.ndim != 3:
        raise ValueError(
            "The multiband array must have shape "
            "rows, columns, bands."
        )

    number_of_bands = array.shape[-1]

    if number_of_bands != dataset.RasterCount:
        raise ValueError(
            "The number of array bands does not match "
            "the number of output raster bands."
        )

    for band_index in range(
        number_of_bands
    ):
        write_array(
            dataset=dataset,
            array=array[..., band_index],
            band_number=band_index + 1,
            x_offset=x_offset,
            y_offset=y_offset
        )


def set_band_metadata(
    dataset,
    band_number: int,
    description: Optional[str] = None,
    nodata_value: Optional[Number] = None,
    unit_type: Optional[str] = None
) -> None:
    """
    Set common metadata on an output raster band.

    Parameters
    ----------
    dataset : gdal.Dataset
        Output raster dataset.

    band_number : int
        1-based output band.

    description : str, optional
        Band description.

    nodata_value : int, float or None
        Band NoData value.

    unit_type : str, optional
        Band unit.
    """

    validate_band_number(
        dataset,
        band_number
    )

    band = dataset.GetRasterBand(
        band_number
    )

    if description:
        band.SetDescription(
            str(description)
        )

    if nodata_value is not None:
        band.SetNoDataValue(
            float(nodata_value)
        )

    if unit_type:
        band.SetUnitType(
            str(unit_type)
        )


def flush_and_close(dataset) -> None:
    """
    Flush a GDAL dataset and release its file handle.

    Parameters
    ----------
    dataset : gdal.Dataset
        GDAL dataset.
    """

    if dataset is not None:
        dataset.FlushCache()

    dataset = None


# ==========================================================
# STATISTICS
# ==========================================================

def compute_band_statistics(
    dataset,
    band_number: int = 1,
    approximate: bool = False,
    force: bool = True
) -> Dict[str, Optional[float]]:
    """
    Compute descriptive statistics for one raster band.

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    band_number : int
        1-based band number.

    approximate : bool
        Permit approximate statistics.

    force : bool
        Force calculation when statistics are absent.

    Returns
    -------
    dict
        Minimum, maximum, mean and standard deviation.
    """

    validate_band_number(
        dataset,
        band_number
    )

    band = dataset.GetRasterBand(
        band_number
    )

    statistics = band.GetStatistics(
        int(bool(approximate)),
        int(bool(force))
    )

    if not statistics:
        return {
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None
        }

    minimum, maximum, mean, standard_deviation = (
        statistics
    )

    return {
        "minimum": float(minimum),
        "maximum": float(maximum),
        "mean": float(mean),
        "standard_deviation": float(
            standard_deviation
        )
    }


def compute_all_statistics(
    dataset,
    approximate: bool = False,
    force: bool = True
) -> List[Dict[str, Optional[float]]]:
    """
    Compute statistics for every raster band.

    Parameters
    ----------
    dataset : gdal.Dataset
        Raster dataset.

    approximate : bool
        Permit approximate statistics.

    force : bool
        Force calculation.

    Returns
    -------
    list of dict
        One statistics dictionary for every band.
    """

    results = []

    for band_number in range(
        1,
        dataset.RasterCount + 1
    ):
        statistics = compute_band_statistics(
            dataset=dataset,
            band_number=band_number,
            approximate=approximate,
            force=force
        )

        statistics["band"] = band_number

        statistics["description"] = (
            get_band_description(
                dataset,
                band_number
            )
        )

        results.append(statistics)

    return results


def calculate_array_statistics(
    array: np.ndarray,
    nodata_value: Optional[Number] = None
) -> Dict[str, Optional[float]]:
    """
    Calculate statistics directly from a NumPy array.

    Parameters
    ----------
    array : numpy.ndarray
        Raster values.

    nodata_value : int, float or None
        NoData value.

    Returns
    -------
    dict
        Count, minimum, maximum, mean and standard deviation.
    """

    array = np.asarray(array)

    valid = valid_data_mask(
        array=array,
        nodata_value=nodata_value
    )

    values = array[valid]

    if values.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None
        }

    values = values.astype(
        np.float64,
        copy=False
    )

    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(
            np.std(values)
        )
    }


# ==========================================================
# DATA-TYPE UTILITIES
# ==========================================================

def numpy_dtype_to_gdal(
    dtype
) -> int:
    """
    Convert a NumPy dtype to a GDAL data type.

    Parameters
    ----------
    dtype
        NumPy dtype or compatible value.

    Returns
    -------
    int
        GDAL data type constant.
    """

    dtype = np.dtype(dtype)

    mapping = {
        np.dtype(np.uint8): gdal.GDT_Byte,
        np.dtype(np.int16): gdal.GDT_Int16,
        np.dtype(np.uint16): gdal.GDT_UInt16,
        np.dtype(np.int32): gdal.GDT_Int32,
        np.dtype(np.uint32): gdal.GDT_UInt32,
        np.dtype(np.float32): gdal.GDT_Float32,
        np.dtype(np.float64): gdal.GDT_Float64
    }

    if dtype not in mapping:
        raise ValueError(
            f"Unsupported NumPy raster dtype: {dtype}"
        )

    return mapping[dtype]


def gdal_dtype_to_numpy(
    data_type: int
):
    """
    Convert a GDAL data type to a NumPy dtype.

    Parameters
    ----------
    data_type : int
        GDAL data type constant.

    Returns
    -------
    numpy.dtype
        NumPy data type.
    """

    mapping = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64
    }

    if data_type not in mapping:
        raise ValueError(
            "Unsupported GDAL raster data type: "
            f"{gdal.GetDataTypeName(data_type)}"
        )

    return np.dtype(
        mapping[data_type]
    )


def suitable_integer_gdal_type(
    minimum_value: int,
    maximum_value: int
) -> int:
    """
    Select an appropriate integer GDAL type.

    Useful for classified rasters.

    Parameters
    ----------
    minimum_value : int
        Minimum class value.

    maximum_value : int
        Maximum class value.

    Returns
    -------
    int
        GDAL data type.
    """

    minimum_value = int(minimum_value)

    maximum_value = int(maximum_value)

    if minimum_value >= 0:

        if maximum_value <= 255:
            return gdal.GDT_Byte

        if maximum_value <= 65535:
            return gdal.GDT_UInt16

        return gdal.GDT_UInt32

    if (
        minimum_value >= -32768
        and maximum_value <= 32767
    ):
        return gdal.GDT_Int16

    return gdal.GDT_Int32


# ==========================================================
# SPATIAL UTILITIES
# ==========================================================

def raster_extent(
    dataset
) -> Tuple[
    float,
    float,
    float,
    float
]:
    """
    Calculate raster extent from its geotransform.

    Returns
    -------
    tuple
        minimum_x, minimum_y, maximum_x, maximum_y.
    """

    if dataset is None:
        raise ValueError(
            "The raster dataset is None."
        )

    geotransform = dataset.GetGeoTransform(
        can_return_null=True
    )

    if geotransform is None:
        raise RuntimeError(
            "The raster does not have a geotransform."
        )

    width = dataset.RasterXSize

    height = dataset.RasterYSize

    corners = [
        pixel_to_map(
            geotransform,
            0,
            0
        ),
        pixel_to_map(
            geotransform,
            width,
            0
        ),
        pixel_to_map(
            geotransform,
            0,
            height
        ),
        pixel_to_map(
            geotransform,
            width,
            height
        )
    ]

    x_values = [
        point[0]
        for point in corners
    ]

    y_values = [
        point[1]
        for point in corners
    ]

    return (
        min(x_values),
        min(y_values),
        max(x_values),
        max(y_values)
    )


def pixel_to_map(
    geotransform,
    column: Number,
    row: Number
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to map coordinates.

    Parameters
    ----------
    geotransform : sequence
        GDAL geotransform.

    column : int or float
        Pixel column.

    row : int or float
        Pixel row.

    Returns
    -------
    tuple
        Map x and y coordinates.
    """

    if geotransform is None:
        raise ValueError(
            "A geotransform is required."
        )

    x = (
        geotransform[0]
        + column * geotransform[1]
        + row * geotransform[2]
    )

    y = (
        geotransform[3]
        + column * geotransform[4]
        + row * geotransform[5]
    )

    return (
        float(x),
        float(y)
    )


def map_to_pixel(
    geotransform,
    x_coordinate: Number,
    y_coordinate: Number
) -> Tuple[float, float]:
    """
    Convert map coordinates to pixel coordinates.

    Parameters
    ----------
    geotransform : sequence
        GDAL geotransform.

    x_coordinate : int or float
        Map x coordinate.

    y_coordinate : int or float
        Map y coordinate.

    Returns
    -------
    tuple
        Pixel column and row.
    """

    if geotransform is None:
        raise ValueError(
            "A geotransform is required."
        )

    inverse_result = gdal.InvGeoTransform(
        geotransform
    )

    if inverse_result is None:
        raise RuntimeError(
            "The raster geotransform could not be inverted."
        )

    # Depending on the GDAL version, InvGeoTransform may
    # return the transform directly or a success flag plus
    # the transform.
    if (
        isinstance(inverse_result, tuple)
        and len(inverse_result) == 2
        and isinstance(inverse_result[0], (int, bool))
    ):
        success, inverse_transform = inverse_result

        if not success:
            raise RuntimeError(
                "The raster geotransform could not be inverted."
            )

    else:
        inverse_transform = inverse_result

    column = (
        inverse_transform[0]
        + x_coordinate * inverse_transform[1]
        + y_coordinate * inverse_transform[2]
    )

    row = (
        inverse_transform[3]
        + x_coordinate * inverse_transform[4]
        + y_coordinate * inverse_transform[5]
    )

    return (
        float(column),
        float(row)
    )


def same_grid(
    first_dataset,
    second_dataset,
    tolerance: float = 1e-9
) -> bool:
    """
    Check whether two rasters use the same pixel grid.

    The comparison includes:

    - Width
    - Height
    - Geotransform
    - Projection

    Parameters
    ----------
    first_dataset : gdal.Dataset
        First raster.

    second_dataset : gdal.Dataset
        Second raster.

    tolerance : float
        Geotransform comparison tolerance.

    Returns
    -------
    bool
        True when the grids are compatible.
    """

    if first_dataset is None or second_dataset is None:
        return False

    if (
        first_dataset.RasterXSize
        != second_dataset.RasterXSize
    ):
        return False

    if (
        first_dataset.RasterYSize
        != second_dataset.RasterYSize
    ):
        return False

    first_transform = first_dataset.GetGeoTransform(
        can_return_null=True
    )

    second_transform = second_dataset.GetGeoTransform(
        can_return_null=True
    )

    if first_transform is None or second_transform is None:
        if first_transform != second_transform:
            return False

    else:
        if not np.allclose(
            first_transform,
            second_transform,
            atol=tolerance,
            rtol=0.0
        ):
            return False

    first_projection = (
        first_dataset.GetProjection()
        or ""
    ).strip()

    second_projection = (
        second_dataset.GetProjection()
        or ""
    ).strip()

    return first_projection == second_projection


# ==========================================================
# SAFE NUMERICAL OPERATIONS
# ==========================================================

def safe_normalized_difference(
    first_array: np.ndarray,
    second_array: np.ndarray,
    invalid_mask: Optional[np.ndarray] = None,
    nodata_value: float = -9999.0,
    epsilon: float = 1e-12
) -> np.ndarray:
    """
    Calculate a normalized difference safely.

    Formula
    -------
    (first - second) / (first + second)

    Parameters
    ----------
    first_array : numpy.ndarray
        First raster band.

    second_array : numpy.ndarray
        Second raster band.

    invalid_mask : numpy.ndarray, optional
        Boolean mask where True means invalid.

    nodata_value : float
        Output NoData value.

    epsilon : float
        Minimum absolute denominator.

    Returns
    -------
    numpy.ndarray
        Float32 normalized-difference array.
    """

    first_array = np.asarray(
        first_array,
        dtype=np.float32
    )

    second_array = np.asarray(
        second_array,
        dtype=np.float32
    )

    if first_array.shape != second_array.shape:
        raise ValueError(
            "The raster-band arrays must have the same shape."
        )

    numerator = (
        first_array - second_array
    )

    denominator = (
        first_array + second_array
    )

    local_invalid_mask = (
        ~np.isfinite(first_array)
        | ~np.isfinite(second_array)
        | ~np.isfinite(denominator)
        | (
            np.abs(denominator)
            <= float(epsilon)
        )
    )

    if invalid_mask is not None:

        invalid_mask = np.asarray(
            invalid_mask,
            dtype=bool
        )

        if invalid_mask.shape != first_array.shape:
            raise ValueError(
                "The invalid mask must have the same shape "
                "as the input arrays."
            )

        local_invalid_mask |= invalid_mask

    result = np.full(
        first_array.shape,
        float(nodata_value),
        dtype=np.float32
    )

    valid_mask = ~local_invalid_mask

    result[valid_mask] = (
        numerator[valid_mask]
        / denominator[valid_mask]
    )

    result[~np.isfinite(result)] = (
        float(nodata_value)
    )

    return result


# ==========================================================
# OUTPUT FINALIZATION
# ==========================================================

def finalize_output_raster(
    dataset,
    calculate_statistics: bool = True,
    build_overviews: bool = False,
    overview_levels: Optional[
        Sequence[int]
    ] = None
) -> None:
    """
    Finalize an output raster before closing it.

    Parameters
    ----------
    dataset : gdal.Dataset
        Writable output dataset.

    calculate_statistics : bool
        Calculate band statistics.

    build_overviews : bool
        Build internal overviews.

    overview_levels : sequence of int, optional
        Overview levels.
    """

    if dataset is None:
        return

    dataset.FlushCache()

    if calculate_statistics:

        for band_number in range(
            1,
            dataset.RasterCount + 1
        ):
            band = dataset.GetRasterBand(
                band_number
            )

            try:
                band.ComputeStatistics(
                    False
                )
            except RuntimeError:
                pass

    if build_overviews:

        if overview_levels is None:
            overview_levels = [
                2,
                4,
                8,
                16
            ]

        try:
            dataset.BuildOverviews(
                "AVERAGE",
                list(overview_levels)
            )

        except RuntimeError:
            pass

    dataset.FlushCache()


# ==========================================================
# CONVENIENCE CLASS
# ==========================================================

class RasterReader:
    """
    Convenience wrapper around a read-only GDAL dataset.

    Examples
    --------
    reader = RasterReader("/data/image.tif")

    print(reader.width)
    print(reader.height)
    print(reader.band_count)

    block = reader.read_bands(
        band_numbers=[1, 2, 3],
        window=(0, 0, 512, 512)
    )

    reader.close()
    """

    def __init__(self, source: str):
        """
        Open the raster source.

        Parameters
        ----------
        source : str
            Raster file path.
        """

        self.source = normalize_raster_source(
            source
        )

        self.dataset = open_raster(
            self.source,
            update=False
        )

        self.info = get_raster_info(
            self.dataset
        )


    @classmethod
    def from_qgis_layer(
        cls,
        raster_layer
    ):
        """
        Construct a reader from a QGIS raster layer.
        """

        return cls(
            raster_layer_source(
                raster_layer
            )
        )


    @property
    def width(self) -> int:
        """
        Raster width.
        """

        return int(
            self.dataset.RasterXSize
        )


    @property
    def height(self) -> int:
        """
        Raster height.
        """

        return int(
            self.dataset.RasterYSize
        )


    @property
    def band_count(self) -> int:
        """
        Number of raster bands.
        """

        return int(
            self.dataset.RasterCount
        )


    @property
    def geotransform(self):
        """
        Raster geotransform.
        """

        return self.dataset.GetGeoTransform(
            can_return_null=True
        )


    @property
    def projection(self) -> str:
        """
        Raster projection WKT.
        """

        return (
            self.dataset.GetProjection()
            or ""
        )


    @property
    def nodata_values(
        self
    ) -> List[Optional[Number]]:
        """
        NoData values for all raster bands.
        """

        return [
            self.dataset
            .GetRasterBand(band_number)
            .GetNoDataValue()

            for band_number in range(
                1,
                self.band_count + 1
            )
        ]


    def read_band(
        self,
        band_number: int,
        window: Optional[
            WindowTuple
        ] = None,
        output_dtype=np.float32
    ) -> np.ndarray:
        """
        Read one raster band.
        """

        return read_band(
            dataset=self.dataset,
            band_number=band_number,
            window=window,
            output_dtype=output_dtype
        )


    def read_bands(
        self,
        band_numbers: Optional[
            Sequence[int]
        ] = None,
        window: Optional[
            WindowTuple
        ] = None,
        output_dtype=np.float32
    ) -> np.ndarray:
        """
        Read several raster bands.
        """

        return read_bands(
            dataset=self.dataset,
            band_numbers=band_numbers,
            window=window,
            output_dtype=output_dtype
        )


    def iter_blocks(
        self,
        band_numbers: Optional[
            Sequence[int]
        ] = None,
        block_width: int = 512,
        block_height: int = 512,
        output_dtype=np.float32
    ):
        """
        Iterate through multiband raster blocks.
        """

        return iter_multiband_blocks(
            dataset=self.dataset,
            band_numbers=band_numbers,
            block_width=block_width,
            block_height=block_height,
            output_dtype=output_dtype
        )


    def close(self) -> None:
        """
        Close the raster dataset.
        """

        if self.dataset is not None:
            self.dataset = None


    def __enter__(self):
        """
        Context-manager entry.
        """

        return self


    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback
    ):
        """
        Context-manager exit.
        """

        self.close()

        return False


class RasterWriter:
    """
    Convenience wrapper for writing GeoTIFF outputs.
    """

    def __init__(
        self,
        output_path: str,
        reference_dataset=None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        band_count: int = 1,
        data_type: int = gdal.GDT_Float32,
        geotransform=None,
        projection: Optional[str] = None,
        nodata_value: Optional[Number] = None,
        creation_options: Optional[
            Sequence[str]
        ] = None,
        overwrite: bool = True
    ):
        """
        Create a writable output raster.

        Either provide a reference dataset or explicitly
        provide width, height, geotransform and projection.
        """

        self.output_path = ensure_tif_extension(
            output_path
        )

        self.nodata_value = nodata_value

        if reference_dataset is not None:

            self.dataset = create_geotiff_like(
                reference_dataset=reference_dataset,
                output_path=self.output_path,
                band_count=band_count,
                data_type=data_type,
                nodata_value=nodata_value,
                creation_options=creation_options,
                overwrite=overwrite
            )

        else:

            if width is None or height is None:
                raise ValueError(
                    "Width and height are required when no "
                    "reference raster is provided."
                )

            self.dataset = create_geotiff(
                output_path=self.output_path,
                width=width,
                height=height,
                band_count=band_count,
                data_type=data_type,
                geotransform=geotransform,
                projection=projection,
                nodata_value=nodata_value,
                creation_options=creation_options,
                overwrite=overwrite
            )


    @property
    def band_count(self) -> int:
        """
        Number of output bands.
        """

        return int(
            self.dataset.RasterCount
        )


    def write(
        self,
        array: np.ndarray,
        band_number: int = 1,
        x_offset: int = 0,
        y_offset: int = 0
    ) -> None:
        """
        Write one array block.
        """

        write_array(
            dataset=self.dataset,
            array=array,
            band_number=band_number,
            x_offset=x_offset,
            y_offset=y_offset
        )


    def write_multiband(
        self,
        array: np.ndarray,
        x_offset: int = 0,
        y_offset: int = 0
    ) -> None:
        """
        Write a multiband array block.
        """

        write_multiband_array(
            dataset=self.dataset,
            array=array,
            x_offset=x_offset,
            y_offset=y_offset
        )


    def set_band_metadata(
        self,
        band_number: int,
        description: Optional[str] = None,
        nodata_value: Optional[Number] = None,
        unit_type: Optional[str] = None
    ) -> None:
        """
        Set output-band metadata.
        """

        set_band_metadata(
            dataset=self.dataset,
            band_number=band_number,
            description=description,
            nodata_value=(
                self.nodata_value
                if nodata_value is None
                else nodata_value
            ),
            unit_type=unit_type
        )


    def close(
        self,
        calculate_statistics: bool = True,
        build_overviews: bool = False
    ) -> None:
        """
        Finalize and close the output raster.
        """

        if self.dataset is not None:

            finalize_output_raster(
                dataset=self.dataset,
                calculate_statistics=calculate_statistics,
                build_overviews=build_overviews
            )

            self.dataset = None


    def __enter__(self):
        """
        Context-manager entry.
        """

        return self


    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback
    ):
        """
        Context-manager exit.
        """

        self.close(
            calculate_statistics=(
                exception_type is None
            )
        )

        return False