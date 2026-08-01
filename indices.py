# -*- coding: utf-8 -*-

"""
QGeoAI Toolkit
Spectral-index calculation module.

Supported indices:

- NDVI
- NDWI
- NBR
- SAVI
- EVI
- GNDVI
- NDMI

Author:
Dimitra Pappa
"""

import os

import numpy as np

from osgeo import gdal


class SpectralIndexCalculator:
    """
    Calculate remote-sensing spectral indices from
    multiband raster imagery.

    The output is saved as a single-band Float32 GeoTIFF.
    """

    def __init__(
        self,
        progress_callback=None,
        log_callback=None
    ):
        """
        Initialize the spectral-index calculator.

        Parameters
        ----------
        progress_callback : callable, optional
            Function receiving progress values from 0 to 100.

        log_callback : callable, optional
            Function receiving log messages.
        """

        self.progress_callback = progress_callback

        self.log_callback = log_callback

        self._progress(0)


    # ==================================================
    # PUBLIC METHOD
    # ==================================================

    def calculate(
        self,
        raster_layer,
        index_name,
        band_a,
        band_b,
        output_path
    ):
        """
        Calculate a spectral index.

        Parameters
        ----------
        raster_layer : QgsRasterLayer
            Input multiband raster.

        index_name : str
            Name of the spectral index.

        band_a : int
            First raster-band number.

        band_b : int
            Second raster-band number.

        output_path : str
            Output GeoTIFF path.

        Returns
        -------
        str
            Output raster path.
        """

        self._log(
            f"Starting {index_name} calculation."
        )

        self._progress(2)

        raster_path = self._clean_raster_source(
            raster_layer.source()
        )

        if not os.path.exists(raster_path):

            raise FileNotFoundError(
                "The raster source does not exist:\n"
                f"{raster_path}"
            )

        dataset = gdal.Open(
            raster_path,
            gdal.GA_ReadOnly
        )

        if dataset is None:

            raise RuntimeError(
                "GDAL could not open the input raster."
            )

        if band_a < 1 or band_a > dataset.RasterCount:

            raise ValueError(
                f"Band A ({band_a}) is outside the valid "
                f"range 1–{dataset.RasterCount}."
            )

        if band_b < 1 or band_b > dataset.RasterCount:

            raise ValueError(
                f"Band B ({band_b}) is outside the valid "
                f"range 1–{dataset.RasterCount}."
            )

        if band_a == band_b:

            raise ValueError(
                "Band A and Band B must be different."
            )

        self._log(
            f"Input raster: {raster_layer.name()}"
        )

        self._log(
            f"Raster size: "
            f"{dataset.RasterXSize} × "
            f"{dataset.RasterYSize}"
        )

        self._log(
            f"Using bands {band_a} and {band_b}."
        )

        self._progress(5)

        output_dataset = self._create_output_dataset(
            dataset,
            output_path
        )

        self._calculate_in_blocks(
            input_dataset=dataset,
            output_dataset=output_dataset,
            index_name=index_name,
            band_a=band_a,
            band_b=band_b
        )

        output_dataset.FlushCache()

        output_dataset = None

        dataset = None

        self._progress(100)

        self._log(
            f"{index_name} saved successfully:\n"
            f"{output_path}"
        )

        return output_path


    def calculate_change(
        self,
        raster_layer_before,
        raster_layer_after,
        index_name,
        band_a,
        band_b,
        delta_output_path
    ):
        """
        Calculate an index for two dates and create a change raster.

        The change raster is computed as:

            delta = index_after - index_before

        Three GeoTIFF files are created: before, after and delta.
        """

        base, extension = os.path.splitext(delta_output_path)
        if extension.lower() not in (".tif", ".tiff"):
            extension = ".tif"
            delta_output_path = base + extension

        before_path = f"{base}_before{extension}"
        after_path = f"{base}_after{extension}"

        self._log("Calculating index for the first raster...")
        self.calculate(
            raster_layer_before,
            index_name,
            band_a,
            band_b,
            before_path
        )

        self._log("Calculating index for the second raster...")
        self.calculate(
            raster_layer_after,
            index_name,
            band_a,
            band_b,
            after_path
        )

        before_ds = gdal.Open(before_path, gdal.GA_ReadOnly)
        after_ds = gdal.Open(after_path, gdal.GA_ReadOnly)

        if before_ds is None or after_ds is None:
            raise RuntimeError("Could not reopen the index rasters.")

        self._validate_matching_grids(before_ds, after_ds)

        delta_ds = self._create_output_dataset(
            before_ds,
            delta_output_path
        )

        before_band = before_ds.GetRasterBand(1)
        after_band = after_ds.GetRasterBand(1)
        delta_band = delta_ds.GetRasterBand(1)

        width = before_ds.RasterXSize
        height = before_ds.RasterYSize
        block_size = 512
        total_blocks = int(np.ceil(width / block_size)) * int(
            np.ceil(height / block_size)
        )
        completed = 0

        for y_offset in range(0, height, block_size):
            rows = min(block_size, height - y_offset)
            for x_offset in range(0, width, block_size):
                columns = min(block_size, width - x_offset)

                before_array = before_band.ReadAsArray(
                    x_offset, y_offset, columns, rows
                ).astype(np.float32, copy=False)
                after_array = after_band.ReadAsArray(
                    x_offset, y_offset, columns, rows
                ).astype(np.float32, copy=False)

                valid = (
                    np.isfinite(before_array)
                    & np.isfinite(after_array)
                    & (before_array != -9999.0)
                    & (after_array != -9999.0)
                )

                output = np.full(
                    (rows, columns),
                    -9999.0,
                    dtype=np.float32
                )
                output[valid] = after_array[valid] - before_array[valid]
                delta_band.WriteArray(output, x_offset, y_offset)

                completed += 1
                self._progress(
                    70 + (completed / total_blocks) * 30
                )

        delta_band.FlushCache()
        delta_ds.FlushCache()

        delta_ds = None
        before_ds = None
        after_ds = None

        self._progress(100)
        self._log(
            f"Change raster saved successfully:\n{delta_output_path}"
        )

        return {
            "before": before_path,
            "after": after_path,
            "delta": delta_output_path
        }


    @staticmethod
    def _validate_matching_grids(dataset_a, dataset_b):
        """Ensure that the two rasters use the same pixel grid."""

        if (
            dataset_a.RasterXSize != dataset_b.RasterXSize
            or dataset_a.RasterYSize != dataset_b.RasterYSize
        ):
            raise ValueError(
                "The two rasters do not have the same dimensions. "
                "Clip and align them to the same grid first."
            )

        transform_a = dataset_a.GetGeoTransform()
        transform_b = dataset_b.GetGeoTransform()

        if not np.allclose(transform_a, transform_b, atol=1e-9):
            raise ValueError(
                "The two rasters do not have the same extent or pixel grid. "
                "Align them before running change detection."
            )

        projection_a = dataset_a.GetProjection() or ""
        projection_b = dataset_b.GetProjection() or ""

        if projection_a != projection_b:
            raise ValueError(
                "The two rasters do not use the same CRS."
            )


    # ==================================================
    # OUTPUT DATASET
    # ==================================================

    def _create_output_dataset(
        self,
        input_dataset,
        output_path
    ):
        """
        Create the output GeoTIFF using the same extent,
        CRS, dimensions and pixel size as the input raster.
        """

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

        driver = gdal.GetDriverByName(
            "GTiff"
        )

        output_dataset = driver.Create(
            output_path,
            input_dataset.RasterXSize,
            input_dataset.RasterYSize,
            1,
            gdal.GDT_Float32,
            options=[
                "COMPRESS=LZW",
                "TILED=YES",
                "BIGTIFF=IF_SAFER"
            ]
        )

        if output_dataset is None:

            raise RuntimeError(
                "Could not create the output GeoTIFF."
            )

        output_dataset.SetGeoTransform(
            input_dataset.GetGeoTransform()
        )

        output_dataset.SetProjection(
            input_dataset.GetProjection()
        )

        output_band = output_dataset.GetRasterBand(1)

        output_band.SetNoDataValue(
            -9999.0
        )

        output_band.Fill(
            -9999.0
        )

        return output_dataset


    # ==================================================
    # BLOCK PROCESSING
    # ==================================================

    def _calculate_in_blocks(
        self,
        input_dataset,
        output_dataset,
        index_name,
        band_a,
        band_b
    ):
        """
        Calculate the index block-by-block to avoid loading
        the complete raster into memory.
        """

        raster_width = input_dataset.RasterXSize

        raster_height = input_dataset.RasterYSize

        input_band_a = input_dataset.GetRasterBand(
            band_a
        )

        input_band_b = input_dataset.GetRasterBand(
            band_b
        )

        output_band = output_dataset.GetRasterBand(
            1
        )

        no_data_a = input_band_a.GetNoDataValue()

        no_data_b = input_band_b.GetNoDataValue()

        block_size = 512

        total_blocks_x = int(
            np.ceil(
                raster_width / block_size
            )
        )

        total_blocks_y = int(
            np.ceil(
                raster_height / block_size
            )
        )

        total_blocks = (
            total_blocks_x
            * total_blocks_y
        )

        completed_blocks = 0

        for y_offset in range(
            0,
            raster_height,
            block_size
        ):

            rows = min(
                block_size,
                raster_height - y_offset
            )

            for x_offset in range(
                0,
                raster_width,
                block_size
            ):

                columns = min(
                    block_size,
                    raster_width - x_offset
                )

                array_a = input_band_a.ReadAsArray(
                    x_offset,
                    y_offset,
                    columns,
                    rows
                )

                array_b = input_band_b.ReadAsArray(
                    x_offset,
                    y_offset,
                    columns,
                    rows
                )

                if array_a is None or array_b is None:

                    raise RuntimeError(
                        "A raster block could not be read."
                    )

                array_a = array_a.astype(
                    np.float32,
                    copy=False
                )

                array_b = array_b.astype(
                    np.float32,
                    copy=False
                )

                valid_mask = self._valid_mask(
                    array_a=array_a,
                    array_b=array_b,
                    no_data_a=no_data_a,
                    no_data_b=no_data_b
                )

                output_array = np.full(
                    (
                        rows,
                        columns
                    ),
                    -9999.0,
                    dtype=np.float32
                )

                if np.any(valid_mask):

                    calculated_values = self._calculate_formula(
                        index_name=index_name,
                        array_a=array_a,
                        array_b=array_b
                    )

                    calculated_values = np.asarray(
                        calculated_values,
                        dtype=np.float32
                    )

                    valid_result = (
                        valid_mask
                        & np.isfinite(
                            calculated_values
                        )
                    )

                    output_array[
                        valid_result
                    ] = calculated_values[
                        valid_result
                    ]

                output_band.WriteArray(
                    output_array,
                    x_offset,
                    y_offset
                )

                completed_blocks += 1

                progress_fraction = (
                    completed_blocks
                    / total_blocks
                )

                self._progress(
                    5
                    + progress_fraction
                    * 93
                )

        output_band.FlushCache()


    # ==================================================
    # INDEX FORMULAS
    # ==================================================

    def _calculate_formula(
        self,
        index_name,
        array_a,
        array_b
    ):
        """
        Apply the selected spectral-index formula.

        The meaning of Band A and Band B depends on the
        index selected in the interface.
        """

        index_name = index_name.upper()

        epsilon = np.float32(
            1e-10
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore"
        ):

            if index_name == "NDVI":

                result = (
                    array_a - array_b
                ) / (
                    array_a
                    + array_b
                    + epsilon
                )

            elif index_name == "NDWI":

                result = (
                    array_a - array_b
                ) / (
                    array_a
                    + array_b
                    + epsilon
                )

            elif index_name == "NBR":

                result = (
                    array_a - array_b
                ) / (
                    array_a
                    + array_b
                    + epsilon
                )

            elif index_name == "GNDVI":

                result = (
                    array_a - array_b
                ) / (
                    array_a
                    + array_b
                    + epsilon
                )

            elif index_name == "NDMI":

                result = (
                    array_a - array_b
                ) / (
                    array_a
                    + array_b
                    + epsilon
                )

            elif index_name == "SAVI":

                soil_adjustment = np.float32(
                    0.5
                )

                result = (
                    (
                        array_a
                        - array_b
                    )
                    / (
                        array_a
                        + array_b
                        + soil_adjustment
                        + epsilon
                    )
                ) * (
                    1.0
                    + soil_adjustment
                )

            elif index_name == "EVI":

                raise ValueError(
                    "EVI requires three bands: NIR, RED "
                    "and BLUE. The current interface accepts "
                    "only two bands. EVI will be added with "
                    "a third band selector in a later step."
                )

            else:

                raise ValueError(
                    f"Unsupported spectral index: "
                    f"{index_name}"
                )

        return result


    # ==================================================
    # VALID PIXELS
    # ==================================================

    @staticmethod
    def _valid_mask(
        array_a,
        array_b,
        no_data_a,
        no_data_b
    ):
        """
        Create a valid-data mask for both input bands.
        """

        valid_mask = (
            np.isfinite(array_a)
            & np.isfinite(array_b)
        )

        if no_data_a is not None:

            if np.isnan(no_data_a):

                valid_mask &= ~np.isnan(
                    array_a
                )

            elif np.issubdtype(
                array_a.dtype,
                np.floating
            ):

                valid_mask &= ~np.isclose(
                    array_a,
                    no_data_a,
                    rtol=0.0,
                    atol=1e-12
                )

            else:

                valid_mask &= (
                    array_a != no_data_a
                )

        if no_data_b is not None:

            if np.isnan(no_data_b):

                valid_mask &= ~np.isnan(
                    array_b
                )

            elif np.issubdtype(
                array_b.dtype,
                np.floating
            ):

                valid_mask &= ~np.isclose(
                    array_b,
                    no_data_b,
                    rtol=0.0,
                    atol=1e-12
                )

            else:

                valid_mask &= (
                    array_b != no_data_b
                )

        return valid_mask


    # ==================================================
    # RASTER SOURCE
    # ==================================================

    @staticmethod
    def _clean_raster_source(source):
        """
        Remove QGIS provider parameters from a raster path.

        Example
        -------
        image.tif|layerid=0

        becomes

        image.tif
        """

        if "|" in source:

            source = source.split("|")[0]

        return os.path.normpath(
            source
        )


    # ==================================================
    # CALLBACK HELPERS
    # ==================================================

    def _progress(self, value):
        """
        Send progress to the plugin dialog.
        """

        if self.progress_callback is not None:

            self.progress_callback(
                max(
                    0,
                    min(
                        100,
                        int(value)
                    )
                )
            )


    def _log(self, message):
        """
        Send a message to the plugin log.
        """

        if self.log_callback is not None:

            self.log_callback(
                str(message)
            )