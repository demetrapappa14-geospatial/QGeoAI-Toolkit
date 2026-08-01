# -*- coding: utf-8 -*-

"""
QGeoAI Toolkit
Machine-learning classification engine.

This module performs:

- Extraction of training pixels from polygon ROIs
- Random Forest classification
- Support Vector Machine classification
- Stratified train/test split
- Optional feature standardization
- Accuracy assessment
- Confusion matrix
- Cohen's Kappa
- Classified GeoTIFF export
- Trained model export

Author:
Dimitra Pappa
"""

import os
from datetime import datetime

import numpy as np

from osgeo import gdal, ogr, osr

from qgis.core import (
    QgsCoordinateTransform,
    QgsProject
)


class ClassificationEngine:
    """
    Supervised raster-classification engine.

    The engine receives:

    - A multiband raster
    - A polygon training layer
    - A class-label field
    - Random Forest or SVM parameters

    It produces:

    - A classified GeoTIFF
    - An accuracy report
    - An optional saved machine-learning model
    """

    def __init__(
        self,
        progress_callback=None,
        log_callback=None
    ):
        """
        Initialize the classification engine.

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

    def run(
        self,
        raster_layer,
        training_layer,
        class_field,
        output_path,
        report_path,
        parameters,
        save_model=True
    ):
        """
        Train a classifier and classify the complete raster.

        Parameters
        ----------
        raster_layer : QgsRasterLayer
            Input multiband raster.

        training_layer : QgsVectorLayer
            Polygon layer containing training regions.

        class_field : str
            Field containing class labels.

        output_path : str
            Output classified GeoTIFF path.

        report_path : str
            Output accuracy-report path.

        parameters : dict
            Machine-learning parameters.

        save_model : bool
            Save the trained model as a joblib file.

        Returns
        -------
        dict
            Classification results and accuracy values.
        """

        self._log(
            "Initializing supervised classification."
        )

        self._progress(2)

        self._check_python_dependencies()

        raster_path = self._clean_raster_source(
            raster_layer.source()
        )

        if not os.path.exists(raster_path):

            raise FileNotFoundError(
                "The raster source does not exist:\n"
                f"{raster_path}"
            )

        raster_dataset = gdal.Open(
            raster_path,
            gdal.GA_ReadOnly
        )

        if raster_dataset is None:

            raise RuntimeError(
                "GDAL could not open the input raster."
            )

        if raster_dataset.RasterCount < 1:

            raise RuntimeError(
                "The selected raster contains no bands."
            )

        self._log(
            f"Raster size: "
            f"{raster_dataset.RasterXSize} × "
            f"{raster_dataset.RasterYSize}"
        )

        self._log(
            f"Raster bands: "
            f"{raster_dataset.RasterCount}"
        )

        self._progress(5)

        (
            training_memory_layer,
            class_mapping
        ) = self._create_integer_training_layer(
            training_layer=training_layer,
            class_field=class_field,
            raster_layer=raster_layer
        )

        if len(class_mapping) < 2:

            raise ValueError(
                "At least two different training classes "
                "are required."
            )

        self._log(
            f"Training classes detected: "
            f"{len(class_mapping)}"
        )

        for class_id, class_name in class_mapping.items():

            self._log(
                f"Class {class_id}: {class_name}"
            )

        self._progress(10)

        label_dataset = self._rasterize_training_layer(
            training_layer=training_memory_layer,
            reference_dataset=raster_dataset
        )

        self._progress(15)

        feature_matrix, target_values = (
            self._extract_training_samples(
                raster_dataset=raster_dataset,
                label_dataset=label_dataset,
                class_mapping=class_mapping,
                max_samples_per_class=parameters[
                    "max_samples_per_class"
                ],
                random_state=parameters[
                    "random_state"
                ]
            )
        )

        if feature_matrix.size == 0:

            raise RuntimeError(
                "No valid training pixels were extracted.\n\n"
                "Check that the training polygons overlap "
                "the raster and that both layers use "
                "compatible coordinate reference systems."
            )

        self._log(
            f"Total training pixels retained: "
            f"{feature_matrix.shape[0]}"
        )

        self._log(
            f"Predictor variables: "
            f"{feature_matrix.shape[1]} raster bands"
        )

        self._progress(40)

        (
            model,
            evaluation_results
        ) = self._train_and_evaluate(
            feature_matrix=feature_matrix,
            target_values=target_values,
            class_mapping=class_mapping,
            parameters=parameters
        )

        self._progress(62)

        self._classify_complete_raster(
            raster_dataset=raster_dataset,
            model=model,
            output_path=output_path
        )

        self._progress(94)

        model_path = None

        if save_model:

            model_path = self._save_model(
                model=model,
                class_mapping=class_mapping,
                parameters=parameters,
                output_path=output_path,
                raster_band_count=raster_dataset.RasterCount
            )

        self._write_accuracy_report(
            report_path=report_path,
            evaluation_results=evaluation_results,
            class_mapping=class_mapping,
            parameters=parameters,
            feature_matrix=feature_matrix,
            output_path=output_path,
            model_path=model_path
        )

        raster_dataset = None

        label_dataset = None

        self._progress(100)

        self._log(
            "Supervised classification finished."
        )

        return {
            "output_path": output_path,
            "report_path": report_path,
            "model_path": model_path,
            "overall_accuracy":
                evaluation_results[
                    "overall_accuracy"
                ],
            "kappa":
                evaluation_results[
                    "kappa"
                ],
            "class_mapping":
                class_mapping
        }


    # ==================================================
    # DEPENDENCY CHECK
    # ==================================================

    @staticmethod
    def _check_python_dependencies():
        """
        Verify that scikit-learn and joblib are available.
        """

        try:

            import sklearn  # noqa: F401

            import joblib  # noqa: F401

        except ImportError as error:

            raise ImportError(
                "QGeoAI Toolkit requires scikit-learn "
                "and joblib in the Python environment "
                "used by QGIS.\n\n"
                f"Original error: {error}"
            ) from error


    # ==================================================
    # RASTER SOURCE
    # ==================================================

    @staticmethod
    def _clean_raster_source(source):
        """
        Remove QGIS provider parameters from a raster path.

        Example
        -------
        input.tif|layerid=0

        becomes

        input.tif
        """

        if "|" in source:

            source = source.split("|")[0]

        return os.path.normpath(source)


    # ==================================================
    # TRAINING VECTOR PREPARATION
    # ==================================================

    def _create_integer_training_layer(
        self,
        training_layer,
        class_field,
        raster_layer
    ):
        """
        Create an in-memory OGR layer with integer class IDs.

        Text labels such as:

        VEGETATION
        WATER
        URBAN

        are converted internally to:

        1
        2
        3
        """

        self._log(
            "Preparing training polygons."
        )

        field_index = (
            training_layer
            .fields()
            .indexFromName(class_field)
        )

        if field_index < 0:

            raise ValueError(
                f"Class field '{class_field}' was not found."
            )

        class_values = []

        valid_features = []

        for feature in training_layer.getFeatures():

            geometry = feature.geometry()

            if geometry is None or geometry.isEmpty():

                continue

            class_value = feature[class_field]

            if class_value is None:

                continue

            class_text = str(class_value).strip()

            if not class_text:

                continue

            valid_features.append(
                (
                    feature,
                    class_text
                )
            )

            if class_text not in class_values:

                class_values.append(class_text)

        if not valid_features:

            raise ValueError(
                "The training layer does not contain valid "
                "polygon features with class labels."
            )

        class_mapping = {
            class_id: class_name
            for class_id, class_name in enumerate(
                class_values,
                start=1
            )
        }

        reverse_mapping = {
            class_name: class_id
            for class_id, class_name
            in class_mapping.items()
        }

        memory_driver = ogr.GetDriverByName(
            "Memory"
        )

        memory_source = (
            memory_driver.CreateDataSource(
                "qgeoai_training"
            )
        )

        raster_crs = raster_layer.crs()

        spatial_reference = osr.SpatialReference()

        if raster_crs.isValid():

            spatial_reference.ImportFromWkt(
                raster_crs.toWkt()
            )

        ogr_layer = memory_source.CreateLayer(
            "training",
            spatial_reference,
            ogr.wkbUnknown
        )

        class_field_definition = ogr.FieldDefn(
            "class_id",
            ogr.OFTInteger
        )

        ogr_layer.CreateField(
            class_field_definition
        )

        coordinate_transform = None

        if (
            training_layer.crs().isValid()
            and raster_crs.isValid()
            and training_layer.crs() != raster_crs
        ):

            coordinate_transform = QgsCoordinateTransform(
                training_layer.crs(),
                raster_crs,
                QgsProject.instance()
            )

            self._log(
                "Training polygons will be transformed "
                "to the raster CRS."
            )

        for feature, class_text in valid_features:

            geometry = feature.geometry()

            geometry = type(geometry)(geometry)

            if coordinate_transform is not None:

                transform_result = geometry.transform(
                    coordinate_transform
                )

                if transform_result != 0:

                    self._log(
                        "Warning: one training geometry "
                        "could not be transformed."
                    )

                    continue

            ogr_geometry = ogr.CreateGeometryFromWkb(
                bytes(geometry.asWkb())
            )

            if ogr_geometry is None:

                continue

            output_feature = ogr.Feature(
                ogr_layer.GetLayerDefn()
            )

            output_feature.SetField(
                "class_id",
                reverse_mapping[class_text]
            )

            output_feature.SetGeometry(
                ogr_geometry
            )

            ogr_layer.CreateFeature(
                output_feature
            )

            output_feature = None

        ogr_layer.ResetReading()

        return memory_source, class_mapping


    # ==================================================
    # TRAINING POLYGON RASTERIZATION
    # ==================================================

    def _rasterize_training_layer(
        self,
        training_layer,
        reference_dataset
    ):
        """
        Rasterize integer class IDs using the input raster
        grid, extent, projection and resolution.
        """

        self._log(
            "Rasterizing training polygons."
        )

        memory_driver = gdal.GetDriverByName(
            "MEM"
        )

        label_dataset = memory_driver.Create(
            "",
            reference_dataset.RasterXSize,
            reference_dataset.RasterYSize,
            1,
            gdal.GDT_UInt16
        )

        if label_dataset is None:

            raise RuntimeError(
                "Could not create the training-label raster."
            )

        label_dataset.SetGeoTransform(
            reference_dataset.GetGeoTransform()
        )

        label_dataset.SetProjection(
            reference_dataset.GetProjection()
        )

        label_band = label_dataset.GetRasterBand(1)

        label_band.Fill(0)

        label_band.SetNoDataValue(0)

        ogr_layer = training_layer.GetLayer(0)

        rasterize_result = gdal.RasterizeLayer(
            label_dataset,
            [1],
            ogr_layer,
            options=[
                "ATTRIBUTE=class_id",
                "ALL_TOUCHED=FALSE"
            ]
        )

        if rasterize_result != 0:

            raise RuntimeError(
                "GDAL failed to rasterize the training polygons."
            )

        label_band.FlushCache()

        return label_dataset


    # ==================================================
    # TRAINING PIXEL EXTRACTION
    # ==================================================

    def _extract_training_samples(
        self,
        raster_dataset,
        label_dataset,
        class_mapping,
        max_samples_per_class,
        random_state
    ):
        """
        Extract balanced training pixels.

        Random keys are used to retain a random sample of
        at most `max_samples_per_class` pixels for each class.
        """

        self._log(
            "Extracting raster values below training polygons."
        )

        raster_width = raster_dataset.RasterXSize

        raster_height = raster_dataset.RasterYSize

        band_count = raster_dataset.RasterCount

        block_size = 512

        random_generator = np.random.default_rng(
            random_state
        )

        sample_store = {}

        for class_id in class_mapping:

            sample_store[class_id] = {
                "features": np.empty(
                    (0, band_count),
                    dtype=np.float32
                ),
                "keys": np.empty(
                    0,
                    dtype=np.float64
                )
            }

        no_data_values = self._raster_nodata_values(
            raster_dataset
        )

        total_blocks_x = int(
            np.ceil(raster_width / block_size)
        )

        total_blocks_y = int(
            np.ceil(raster_height / block_size)
        )

        total_blocks = (
            total_blocks_x
            * total_blocks_y
        )

        processed_blocks = 0

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

                label_array = (
                    label_dataset
                    .GetRasterBand(1)
                    .ReadAsArray(
                        x_offset,
                        y_offset,
                        columns,
                        rows
                    )
                )

                if label_array is None:

                    continue

                if not np.any(label_array > 0):

                    processed_blocks += 1

                    self._training_progress(
                        processed_blocks,
                        total_blocks
                    )

                    continue

                raster_array = raster_dataset.ReadAsArray(
                    x_offset,
                    y_offset,
                    columns,
                    rows
                )

                raster_array = self._normalize_raster_array(
                    raster_array,
                    band_count
                )

                valid_mask = self._valid_pixel_mask(
                    raster_array,
                    no_data_values
                )

                label_flat = label_array.reshape(-1)

                features_flat = (
                    raster_array
                    .reshape(
                        band_count,
                        -1
                    )
                    .T
                )

                valid_flat = valid_mask.reshape(-1)

                for class_id in class_mapping:

                    class_mask = (
                        (label_flat == class_id)
                        & valid_flat
                    )

                    class_features = features_flat[
                        class_mask
                    ]

                    if class_features.shape[0] == 0:

                        continue

                    class_features = class_features.astype(
                        np.float32,
                        copy=False
                    )

                    new_keys = random_generator.random(
                        class_features.shape[0]
                    )

                    current_features = (
                        sample_store[class_id][
                            "features"
                        ]
                    )

                    current_keys = (
                        sample_store[class_id][
                            "keys"
                        ]
                    )

                    combined_features = np.vstack(
                        [
                            current_features,
                            class_features
                        ]
                    )

                    combined_keys = np.concatenate(
                        [
                            current_keys,
                            new_keys
                        ]
                    )

                    if (
                        combined_features.shape[0]
                        > max_samples_per_class
                    ):

                        retained_indices = np.argpartition(
                            combined_keys,
                            max_samples_per_class - 1
                        )[
                            :max_samples_per_class
                        ]

                        combined_features = (
                            combined_features[
                                retained_indices
                            ]
                        )

                        combined_keys = combined_keys[
                            retained_indices
                        ]

                    sample_store[class_id][
                        "features"
                    ] = combined_features

                    sample_store[class_id][
                        "keys"
                    ] = combined_keys

                processed_blocks += 1

                self._training_progress(
                    processed_blocks,
                    total_blocks
                )

        feature_parts = []

        target_parts = []

        for class_id, class_name in class_mapping.items():

            class_features = sample_store[
                class_id
            ]["features"]

            sample_count = class_features.shape[0]

            self._log(
                f"Class '{class_name}': "
                f"{sample_count} valid pixels retained."
            )

            if sample_count < 2:

                raise ValueError(
                    f"Class '{class_name}' contains fewer "
                    "than two valid training pixels."
                )

            feature_parts.append(
                class_features
            )

            target_parts.append(
                np.full(
                    sample_count,
                    class_id,
                    dtype=np.uint16
                )
            )

        feature_matrix = np.vstack(
            feature_parts
        )

        target_values = np.concatenate(
            target_parts
        )

        return feature_matrix, target_values


    def _training_progress(
        self,
        completed_blocks,
        total_blocks
    ):
        """
        Update progress during sample extraction.
        """

        if total_blocks <= 0:

            return

        fraction = (
            completed_blocks
            / total_blocks
        )

        progress_value = 15 + (
            fraction * 23
        )

        self._progress(
            progress_value
        )


    # ==================================================
    # MODEL TRAINING
    # ==================================================

    def _train_and_evaluate(
        self,
        feature_matrix,
        target_values,
        class_mapping,
        parameters
    ):
        """
        Train and evaluate Random Forest or SVM.
        """

        from sklearn.ensemble import (
            RandomForestClassifier
        )

        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            cohen_kappa_score,
            confusion_matrix
        )

        from sklearn.model_selection import (
            train_test_split
        )

        from sklearn.pipeline import Pipeline

        from sklearn.preprocessing import (
            StandardScaler
        )

        from sklearn.svm import SVC

        self._log(
            "Creating stratified train/test split."
        )

        (
            x_train,
            x_test,
            y_train,
            y_test
        ) = train_test_split(
            feature_matrix,
            target_values,
            test_size=parameters[
                "test_fraction"
            ],
            random_state=parameters[
                "random_state"
            ],
            stratify=target_values
        )

        algorithm = parameters[
            "algorithm"
        ]

        if algorithm == "random_forest":

            self._log(
                "Training Random Forest model."
            )

            classifier = RandomForestClassifier(
                n_estimators=parameters[
                    "n_estimators"
                ],
                max_depth=parameters[
                    "max_depth"
                ],
                min_samples_split=parameters[
                    "min_samples_split"
                ],
                min_samples_leaf=parameters[
                    "min_samples_leaf"
                ],
                class_weight=parameters[
                    "class_weight"
                ],
                random_state=parameters[
                    "random_state"
                ],
                n_jobs=-1
            )

        elif algorithm == "svm":

            self._log(
                "Training Support Vector Machine model."
            )

            classifier = SVC(
                kernel="rbf",
                C=parameters["c"],
                gamma=parameters["gamma"],
                probability=parameters[
                    "probability"
                ],
                class_weight="balanced",
                random_state=parameters[
                    "random_state"
                ]
            )

        else:

            raise ValueError(
                f"Unsupported classifier: {algorithm}"
            )

        steps = []

        if parameters.get(
            "standardize",
            False
        ):

            self._log(
                "Standardizing predictor variables."
            )

            steps.append(
                (
                    "scaler",
                    StandardScaler()
                )
            )

        steps.append(
            (
                "model",
                classifier
            )
        )

        model = Pipeline(
            steps
        )

        model.fit(
            x_train,
            y_train
        )

        self._progress(55)

        self._log(
            "Evaluating model on test pixels."
        )

        predicted_test = model.predict(
            x_test
        )

        labels = list(
            class_mapping.keys()
        )

        target_names = [
            class_mapping[class_id]
            for class_id in labels
        ]

        overall_accuracy = accuracy_score(
            y_test,
            predicted_test
        )

        kappa = cohen_kappa_score(
            y_test,
            predicted_test
        )

        matrix = confusion_matrix(
            y_test,
            predicted_test,
            labels=labels
        )

        report_text = classification_report(
            y_test,
            predicted_test,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0
        )

        report_dictionary = classification_report(
            y_test,
            predicted_test,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )

        self._log(
            f"Overall Accuracy: "
            f"{overall_accuracy:.4f}"
        )

        self._log(
            f"Cohen's Kappa: "
            f"{kappa:.4f}"
        )

        feature_importances = None

        trained_classifier = (
            model.named_steps["model"]
        )

        if hasattr(
            trained_classifier,
            "feature_importances_"
        ):

            feature_importances = (
                trained_classifier
                .feature_importances_
            )

        results = {
            "overall_accuracy":
                float(overall_accuracy),
            "kappa":
                float(kappa),
            "confusion_matrix":
                matrix,
            "classification_report_text":
                report_text,
            "classification_report_dict":
                report_dictionary,
            "feature_importances":
                feature_importances,
            "training_sample_count":
                int(x_train.shape[0]),
            "test_sample_count":
                int(x_test.shape[0])
        }

        return model, results


    # ==================================================
    # COMPLETE RASTER CLASSIFICATION
    # ==================================================

    def _classify_complete_raster(
        self,
        raster_dataset,
        model,
        output_path
    ):
        """
        Predict classes for the complete raster in blocks.
        """

        self._log(
            "Classifying the complete raster."
        )

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

        raster_width = raster_dataset.RasterXSize

        raster_height = raster_dataset.RasterYSize

        band_count = raster_dataset.RasterCount

        driver = gdal.GetDriverByName(
            "GTiff"
        )

        output_dataset = driver.Create(
            output_path,
            raster_width,
            raster_height,
            1,
            gdal.GDT_UInt16,
            options=[
                "COMPRESS=LZW",
                "TILED=YES",
                "BIGTIFF=IF_SAFER"
            ]
        )

        if output_dataset is None:

            raise RuntimeError(
                "Could not create the classified GeoTIFF."
            )

        output_dataset.SetGeoTransform(
            raster_dataset.GetGeoTransform()
        )

        output_dataset.SetProjection(
            raster_dataset.GetProjection()
        )

        output_band = output_dataset.GetRasterBand(1)

        output_band.SetNoDataValue(0)

        output_band.Fill(0)

        no_data_values = self._raster_nodata_values(
            raster_dataset
        )

        block_size = 512

        total_blocks_x = int(
            np.ceil(raster_width / block_size)
        )

        total_blocks_y = int(
            np.ceil(raster_height / block_size)
        )

        total_blocks = (
            total_blocks_x
            * total_blocks_y
        )

        processed_blocks = 0

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

                raster_array = raster_dataset.ReadAsArray(
                    x_offset,
                    y_offset,
                    columns,
                    rows
                )

                raster_array = self._normalize_raster_array(
                    raster_array,
                    band_count
                )

                valid_mask = self._valid_pixel_mask(
                    raster_array,
                    no_data_values
                )

                output_array = np.zeros(
                    (
                        rows,
                        columns
                    ),
                    dtype=np.uint16
                )

                if np.any(valid_mask):

                    flat_features = (
                        raster_array
                        .reshape(
                            band_count,
                            -1
                        )
                        .T
                    )

                    valid_flat = (
                        valid_mask.reshape(-1)
                    )

                    valid_features = flat_features[
                        valid_flat
                    ].astype(
                        np.float32,
                        copy=False
                    )

                    predicted_classes = model.predict(
                        valid_features
                    )

                    output_flat = (
                        output_array.reshape(-1)
                    )

                    output_flat[
                        valid_flat
                    ] = predicted_classes.astype(
                        np.uint16
                    )

                output_band.WriteArray(
                    output_array,
                    x_offset,
                    y_offset
                )

                processed_blocks += 1

                fraction = (
                    processed_blocks
                    / total_blocks
                )

                self._progress(
                    62 + fraction * 31
                )

        output_band.FlushCache()

        output_dataset.FlushCache()

        output_dataset = None

        self._log(
            f"Classified raster saved:\n"
            f"{output_path}"
        )


    # ==================================================
    # RASTER ARRAY HELPERS
    # ==================================================

    @staticmethod
    def _normalize_raster_array(
        raster_array,
        band_count
    ):
        """
        Ensure raster arrays use shape:

        bands × rows × columns
        """

        if raster_array is None:

            raise RuntimeError(
                "Raster block could not be read."
            )

        raster_array = np.asarray(
            raster_array
        )

        if band_count == 1:

            if raster_array.ndim == 2:

                raster_array = raster_array[
                    np.newaxis,
                    :,
                    :
                ]

        if raster_array.ndim != 3:

            raise RuntimeError(
                "Unexpected raster-array dimensions."
            )

        return raster_array


    @staticmethod
    def _raster_nodata_values(
        raster_dataset
    ):
        """
        Return the NoData value of every raster band.
        """

        no_data_values = []

        for band_number in range(
            1,
            raster_dataset.RasterCount + 1
        ):

            raster_band = (
                raster_dataset
                .GetRasterBand(band_number)
            )

            no_data_values.append(
                raster_band.GetNoDataValue()
            )

        return no_data_values


    @staticmethod
    def _valid_pixel_mask(
        raster_array,
        no_data_values
    ):
        """
        Determine pixels valid across all raster bands.
        """

        valid_mask = np.all(
            np.isfinite(raster_array),
            axis=0
        )

        for band_index, no_data_value in enumerate(
            no_data_values
        ):

            if no_data_value is None:

                continue

            band_array = raster_array[
                band_index
            ]

            if np.isnan(no_data_value):

                valid_mask &= ~np.isnan(
                    band_array
                )

            elif np.issubdtype(
                band_array.dtype,
                np.floating
            ):

                valid_mask &= ~np.isclose(
                    band_array,
                    no_data_value,
                    rtol=0.0,
                    atol=1e-12
                )

            else:

                valid_mask &= (
                    band_array
                    != no_data_value
                )

        return valid_mask


    # ==================================================
    # MODEL EXPORT
    # ==================================================

    def _save_model(
        self,
        model,
        class_mapping,
        parameters,
        output_path,
        raster_band_count
    ):
        """
        Save the trained classifier and its metadata.
        """

        import joblib

        model_path = (
            os.path.splitext(output_path)[0]
            + "_model.joblib"
        )

        model_package = {
            "model": model,
            "class_mapping": class_mapping,
            "parameters": parameters,
            "raster_band_count":
                raster_band_count,
            "created":
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            "plugin":
                "QGeoAI Toolkit"
        }

        joblib.dump(
            model_package,
            model_path
        )

        self._log(
            f"Trained model saved:\n"
            f"{model_path}"
        )

        return model_path


    # ==================================================
    # ACCURACY REPORT
    # ==================================================

    def _write_accuracy_report(
        self,
        report_path,
        evaluation_results,
        class_mapping,
        parameters,
        feature_matrix,
        output_path,
        model_path
    ):
        """
        Write model settings and accuracy metrics.
        """

        report_directory = os.path.dirname(
            report_path
        )

        if report_directory:

            os.makedirs(
                report_directory,
                exist_ok=True
            )

        algorithm_name = {
            "random_forest":
                "Random Forest",
            "svm":
                "Support Vector Machine (RBF)"
        }.get(
            parameters["algorithm"],
            parameters["algorithm"]
        )

        matrix = evaluation_results[
            "confusion_matrix"
        ]

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as report_file:

            report_file.write(
                "QGeoAI Toolkit\n"
            )

            report_file.write(
                "Supervised Classification Report\n"
            )

            report_file.write(
                "=" * 50 + "\n\n"
            )

            report_file.write(
                f"Created: "
                f"{datetime.now().isoformat(timespec='seconds')}\n"
            )

            report_file.write(
                f"Algorithm: {algorithm_name}\n"
            )

            report_file.write(
                f"Classified raster: {output_path}\n"
            )

            report_file.write(
                f"Saved model: "
                f"{model_path or 'Not saved'}\n\n"
            )

            report_file.write(
                "DATASET\n"
            )

            report_file.write(
                "-" * 50 + "\n"
            )

            report_file.write(
                f"Total retained samples: "
                f"{feature_matrix.shape[0]}\n"
            )

            report_file.write(
                f"Predictor bands: "
                f"{feature_matrix.shape[1]}\n"
            )

            report_file.write(
                f"Training samples: "
                f"{evaluation_results['training_sample_count']}\n"
            )

            report_file.write(
                f"Test samples: "
                f"{evaluation_results['test_sample_count']}\n"
            )

            report_file.write(
                f"Test fraction: "
                f"{parameters['test_fraction']}\n"
            )

            report_file.write(
                f"Random state: "
                f"{parameters['random_state']}\n\n"
            )

            report_file.write(
                "CLASS MAPPING\n"
            )

            report_file.write(
                "-" * 50 + "\n"
            )

            for class_id, class_name in class_mapping.items():

                report_file.write(
                    f"{class_id} = {class_name}\n"
                )

            report_file.write("\n")

            report_file.write(
                "ACCURACY\n"
            )

            report_file.write(
                "-" * 50 + "\n"
            )

            report_file.write(
                f"Overall Accuracy: "
                f"{evaluation_results['overall_accuracy']:.6f}\n"
            )

            report_file.write(
                f"Cohen's Kappa: "
                f"{evaluation_results['kappa']:.6f}\n\n"
            )

            report_file.write(
                "CLASSIFICATION REPORT\n"
            )

            report_file.write(
                "-" * 50 + "\n"
            )

            report_file.write(
                evaluation_results[
                    "classification_report_text"
                ]
            )

            report_file.write("\n")

            report_file.write(
                "CONFUSION MATRIX\n"
            )

            report_file.write(
                "-" * 50 + "\n"
            )

            report_file.write(
                "Rows = reference classes\n"
            )

            report_file.write(
                "Columns = predicted classes\n\n"
            )

            class_ids = list(
                class_mapping.keys()
            )

            report_file.write(
                "Class order: "
                + ", ".join(
                    str(class_id)
                    for class_id in class_ids
                )
                + "\n\n"
            )

            report_file.write(
                np.array2string(
                    matrix,
                    separator=", "
                )
            )

            report_file.write("\n\n")

            feature_importances = (
                evaluation_results[
                    "feature_importances"
                ]
            )

            if feature_importances is not None:

                report_file.write(
                    "RANDOM FOREST FEATURE IMPORTANCE\n"
                )

                report_file.write(
                    "-" * 50 + "\n"
                )

                for band_index, importance in enumerate(
                    feature_importances,
                    start=1
                ):

                    report_file.write(
                        f"Band {band_index}: "
                        f"{importance:.8f}\n"
                    )

                report_file.write("\n")

            report_file.write(
                "MODEL PARAMETERS\n"
            )

            report_file.write(
                "-" * 50 + "\n"
            )

            for key, value in parameters.items():

                report_file.write(
                    f"{key}: {value}\n"
                )

        self._log(
            f"Accuracy report saved:\n"
            f"{report_path}"
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