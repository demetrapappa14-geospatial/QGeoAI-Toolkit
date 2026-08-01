# -*- coding: utf-8 -*-

"""
QGeoAI Toolkit
Main graphical user interface.

Author:
Dimitra Pappa
"""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget
)

from qgis.core import (
    QgsMapLayerProxyModel,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer
)

from qgis.gui import (
    QgsFieldComboBox,
    QgsMapLayerComboBox
)


class QGeoAIDialog(QDialog):
    """
    Main dialog of the QGeoAI Toolkit plugin.
    """

    def __init__(self, iface, parent=None):
        """
        Initialize the plugin dialog.

        Parameters
        ----------
        iface : QgisInterface
            QGIS application interface.

        parent : QWidget, optional
            Parent widget.
        """

        if parent is None:
            parent = iface.mainWindow()

        super().__init__(parent)

        self.iface = iface

        self.setWindowTitle("QGeoAI Toolkit")

        self.setMinimumSize(720, 650)

        self.resize(820, 760)

        self._create_interface()

        self._connect_signals()

        self._update_algorithm_options()

        self._update_index_defaults()

        self._update_index_mode()


    # ==================================================
    # MAIN INTERFACE
    # ==================================================

    def _create_interface(self):
        """
        Create the complete graphical interface.
        """

        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(
            14,
            14,
            14,
            14
        )

        main_layout.setSpacing(10)

        title_label = QLabel(
            "<h2>QGeoAI Toolkit</h2>"
            "<p>"
            "Supervised machine-learning and remote-sensing "
            "tools for QGIS."
            "</p>"
        )

        title_label.setTextFormat(Qt.RichText)

        title_label.setWordWrap(True)

        main_layout.addWidget(title_label)

        self.tabs = QTabWidget()

        self.classification_tab = (
            self._create_classification_tab()
        )

        self.indices_tab = (
            self._create_indices_tab()
        )

        self.tabs.addTab(
            self.classification_tab,
            "RF / SVM Classification"
        )

        self.tabs.addTab(
            self.indices_tab,
            "Spectral Indices"
        )

        main_layout.addWidget(self.tabs)

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(0, 100)

        self.progress_bar.setValue(0)

        self.progress_bar.setTextVisible(True)

        main_layout.addWidget(self.progress_bar)

        self.log_box = QPlainTextEdit()

        self.log_box.setReadOnly(True)

        self.log_box.setMaximumHeight(150)

        self.log_box.setPlaceholderText(
            "Processing messages will appear here."
        )

        main_layout.addWidget(self.log_box)

        bottom_layout = QHBoxLayout()

        bottom_layout.addStretch()

        self.close_button = QPushButton("Close")

        self.close_button.setMinimumWidth(100)

        bottom_layout.addWidget(self.close_button)

        main_layout.addLayout(bottom_layout)


    # ==================================================
    # CLASSIFICATION TAB
    # ==================================================

    def _create_classification_tab(self):
        """
        Create the Random Forest and SVM classification tab.
        """

        tab = QWidget()

        layout = QVBoxLayout(tab)

        layout.setSpacing(12)

        input_group = self._create_classification_inputs()

        model_group = self._create_model_options()

        output_group = self._create_classification_outputs()

        layout.addWidget(input_group)

        layout.addWidget(model_group)

        layout.addWidget(output_group)

        self.run_classification_button = QPushButton(
            "Train Model and Classify Raster"
        )

        self.run_classification_button.setMinimumHeight(42)

        layout.addWidget(self.run_classification_button)

        layout.addStretch()

        return tab


    def _create_classification_inputs(self):
        """
        Create input layer controls.
        """

        group = QGroupBox("Input Data")

        form = QFormLayout(group)

        self.raster_combo = QgsMapLayerComboBox()

        self.raster_combo.setFilters(
            QgsMapLayerProxyModel.RasterLayer
        )

        self.raster_combo.setAllowEmptyLayer(True)

        self.raster_combo.setCurrentIndex(-1)

        self.training_combo = QgsMapLayerComboBox()

        self.training_combo.setFilters(
            QgsMapLayerProxyModel.PolygonLayer
        )

        self.training_combo.setAllowEmptyLayer(True)

        self.training_combo.setCurrentIndex(-1)

        self.class_field_combo = QgsFieldComboBox()

        self.class_field_combo.setAllowEmptyFieldName(True)

        self.class_field_combo.setEnabled(False)

        raster_help = QLabel(
            "Use a multiband raster whose bands contain "
            "the predictor variables."
        )

        raster_help.setWordWrap(True)

        training_help = QLabel(
            "Training polygons must overlap the raster and "
            "contain a field with the class name or class ID."
        )

        training_help.setWordWrap(True)

        form.addRow(
            "Multiband raster:",
            self.raster_combo
        )

        form.addRow(
            "",
            raster_help
        )

        form.addRow(
            "Training polygons:",
            self.training_combo
        )

        form.addRow(
            "",
            training_help
        )

        form.addRow(
            "Class field:",
            self.class_field_combo
        )

        return group


    def _create_model_options(self):
        """
        Create machine-learning model parameters.
        """

        group = QGroupBox("Machine-Learning Settings")

        main_layout = QVBoxLayout(group)

        general_form = QFormLayout()

        self.algorithm_combo = QComboBox()

        self.algorithm_combo.addItem(
            "Random Forest",
            "random_forest"
        )

        self.algorithm_combo.addItem(
            "Support Vector Machine (RBF)",
            "svm"
        )

        self.test_fraction_spin = QDoubleSpinBox()

        self.test_fraction_spin.setRange(
            0.10,
            0.50
        )

        self.test_fraction_spin.setSingleStep(
            0.05
        )

        self.test_fraction_spin.setDecimals(2)

        self.test_fraction_spin.setValue(0.25)

        self.random_state_spin = QSpinBox()

        self.random_state_spin.setRange(
            0,
            999999
        )

        self.random_state_spin.setValue(42)

        self.max_samples_spin = QSpinBox()

        self.max_samples_spin.setRange(
            100,
            1000000
        )

        self.max_samples_spin.setSingleStep(
            500
        )

        self.max_samples_spin.setValue(5000)

        self.max_samples_spin.setToolTip(
            "Maximum number of training pixels retained "
            "for each class."
        )

        self.standardize_checkbox = QCheckBox(
            "Standardize raster values before training"
        )

        self.standardize_checkbox.setChecked(False)

        general_form.addRow(
            "Algorithm:",
            self.algorithm_combo
        )

        general_form.addRow(
            "Test fraction:",
            self.test_fraction_spin
        )

        general_form.addRow(
            "Random state:",
            self.random_state_spin
        )

        general_form.addRow(
            "Maximum samples per class:",
            self.max_samples_spin
        )

        general_form.addRow(
            "",
            self.standardize_checkbox
        )

        main_layout.addLayout(general_form)

        self.rf_group = self._create_rf_options()

        self.svm_group = self._create_svm_options()

        main_layout.addWidget(self.rf_group)

        main_layout.addWidget(self.svm_group)

        return group


    def _create_rf_options(self):
        """
        Create Random Forest parameters.
        """

        group = QGroupBox("Random Forest Parameters")

        form = QFormLayout(group)

        self.rf_trees_spin = QSpinBox()

        self.rf_trees_spin.setRange(
            10,
            2000
        )

        self.rf_trees_spin.setSingleStep(
            50
        )

        self.rf_trees_spin.setValue(300)

        self.rf_max_depth_spin = QSpinBox()

        self.rf_max_depth_spin.setRange(
            0,
            100
        )

        self.rf_max_depth_spin.setValue(0)

        self.rf_max_depth_spin.setSpecialValueText(
            "Unlimited"
        )

        self.rf_min_samples_split_spin = QSpinBox()

        self.rf_min_samples_split_spin.setRange(
            2,
            100
        )

        self.rf_min_samples_split_spin.setValue(2)

        self.rf_min_samples_leaf_spin = QSpinBox()

        self.rf_min_samples_leaf_spin.setRange(
            1,
            100
        )

        self.rf_min_samples_leaf_spin.setValue(1)

        self.rf_class_weight_combo = QComboBox()

        self.rf_class_weight_combo.addItem(
            "Balanced",
            "balanced"
        )

        self.rf_class_weight_combo.addItem(
            "No weighting",
            None
        )

        form.addRow(
            "Number of trees:",
            self.rf_trees_spin
        )

        form.addRow(
            "Maximum tree depth:",
            self.rf_max_depth_spin
        )

        form.addRow(
            "Minimum samples to split:",
            self.rf_min_samples_split_spin
        )

        form.addRow(
            "Minimum samples per leaf:",
            self.rf_min_samples_leaf_spin
        )

        form.addRow(
            "Class weighting:",
            self.rf_class_weight_combo
        )

        return group


    def _create_svm_options(self):
        """
        Create Support Vector Machine parameters.
        """

        group = QGroupBox("SVM Parameters")

        form = QFormLayout(group)

        self.svm_c_spin = QDoubleSpinBox()

        self.svm_c_spin.setRange(
            0.001,
            100000.0
        )

        self.svm_c_spin.setDecimals(3)

        self.svm_c_spin.setValue(10.0)

        self.svm_gamma_combo = QComboBox()

        self.svm_gamma_combo.addItem(
            "Scale",
            "scale"
        )

        self.svm_gamma_combo.addItem(
            "Auto",
            "auto"
        )

        self.svm_probability_checkbox = QCheckBox(
            "Enable probability estimates"
        )

        self.svm_probability_checkbox.setChecked(False)

        self.svm_probability_checkbox.setToolTip(
            "Probability estimates increase model training time."
        )

        form.addRow(
            "C parameter:",
            self.svm_c_spin
        )

        form.addRow(
            "Gamma:",
            self.svm_gamma_combo
        )

        form.addRow(
            "",
            self.svm_probability_checkbox
        )

        return group


    def _create_classification_outputs(self):
        """
        Create output controls for classification.
        """

        group = QGroupBox("Classification Outputs")

        form = QFormLayout(group)

        self.classified_output_edit = QLineEdit()

        self.classified_output_button = QPushButton(
            "Browse..."
        )

        classified_row = self._create_file_row(
            self.classified_output_edit,
            self.classified_output_button
        )

        self.report_output_edit = QLineEdit()

        self.report_output_button = QPushButton(
            "Browse..."
        )

        report_row = self._create_file_row(
            self.report_output_edit,
            self.report_output_button
        )

        self.add_result_checkbox = QCheckBox(
            "Add classified raster to QGIS"
        )

        self.add_result_checkbox.setChecked(True)

        self.save_model_checkbox = QCheckBox(
            "Save trained model"
        )

        self.save_model_checkbox.setChecked(True)

        form.addRow(
            "Classified GeoTIFF:",
            classified_row
        )

        form.addRow(
            "Accuracy report:",
            report_row
        )

        form.addRow(
            "",
            self.add_result_checkbox
        )

        form.addRow(
            "",
            self.save_model_checkbox
        )

        return group


    # ==================================================
    # SPECTRAL INDICES TAB
    # ==================================================

    def _create_indices_tab(self):
        """
        Create the spectral-index tab.
        """

        tab = QWidget()

        layout = QVBoxLayout(tab)

        input_group = QGroupBox(
            "Spectral Index Settings"
        )

        form = QFormLayout(input_group)

        self.index_raster_combo = QgsMapLayerComboBox()

        self.index_raster_combo.setFilters(
            QgsMapLayerProxyModel.RasterLayer
        )

        self.index_raster_combo.setAllowEmptyLayer(True)

        self.index_raster_combo.setCurrentIndex(-1)

        self.index_mode_combo = QComboBox()

        self.index_mode_combo.addItem(
            "Single raster",
            "single"
        )

        self.index_mode_combo.addItem(
            "Two rasters / Change detection",
            "change"
        )

        self.index_raster_2_combo = QgsMapLayerComboBox()

        self.index_raster_2_combo.setFilters(
            QgsMapLayerProxyModel.RasterLayer
        )

        self.index_raster_2_combo.setAllowEmptyLayer(True)

        self.index_raster_2_combo.setCurrentIndex(-1)

        self.index_raster_2_combo.setEnabled(False)

        self.index_combo = QComboBox()

        self.index_combo.addItem(
            "NDVI – Normalized Difference Vegetation Index",
            "NDVI"
        )

        self.index_combo.addItem(
            "NDWI – Normalized Difference Water Index",
            "NDWI"
        )

        self.index_combo.addItem(
            "NBR – Normalized Burn Ratio",
            "NBR"
        )

        self.index_combo.addItem(
            "SAVI – Soil Adjusted Vegetation Index",
            "SAVI"
        )

        self.index_combo.addItem(
            "GNDVI – Green Normalized Difference Vegetation Index",
            "GNDVI"
        )

        self.index_combo.addItem(
            "NDMI – Normalized Difference Moisture Index",
            "NDMI"
        )

        self.index_band_a_spin = QSpinBox()

        self.index_band_a_spin.setRange(
            1,
            999
        )

        self.index_band_a_spin.setValue(4)

        self.index_band_b_spin = QSpinBox()

        self.index_band_b_spin.setRange(
            1,
            999
        )

        self.index_band_b_spin.setValue(3)

        self.index_output_edit = QLineEdit()

        self.index_output_button = QPushButton(
            "Browse..."
        )

        index_output_row = self._create_file_row(
            self.index_output_edit,
            self.index_output_button
        )

        self.index_add_result_checkbox = QCheckBox(
            "Add index raster to QGIS"
        )

        self.index_add_result_checkbox.setChecked(True)

        self.index_formula_label = QLabel()

        self.index_formula_label.setWordWrap(True)

        form.addRow(
            "Processing mode:",
            self.index_mode_combo
        )

        form.addRow(
            "Raster 1 (before / single):",
            self.index_raster_combo
        )

        form.addRow(
            "Raster 2 (after):",
            self.index_raster_2_combo
        )

        form.addRow(
            "Index:",
            self.index_combo
        )

        form.addRow(
            "Band A:",
            self.index_band_a_spin
        )

        form.addRow(
            "Band B:",
            self.index_band_b_spin
        )

        form.addRow(
            "Formula:",
            self.index_formula_label
        )

        form.addRow(
            "Output GeoTIFF / ΔIndex:",
            index_output_row
        )

        form.addRow(
            "",
            self.index_add_result_checkbox
        )

        layout.addWidget(input_group)

        self.run_index_button = QPushButton(
            "Calculate Spectral Index"
        )

        self.run_index_button.setMinimumHeight(42)

        layout.addWidget(self.run_index_button)

        layout.addStretch()

        return tab


    # ==================================================
    # SIGNAL CONNECTIONS
    # ==================================================

    def _connect_signals(self):
        """
        Connect interface controls to their functions.
        """

        self.close_button.clicked.connect(
            self.close
        )

        self.training_combo.layerChanged.connect(
            self._training_layer_changed
        )

        self.algorithm_combo.currentIndexChanged.connect(
            self._update_algorithm_options
        )

        self.classified_output_button.clicked.connect(
            self._browse_classified_output
        )

        self.report_output_button.clicked.connect(
            self._browse_report_output
        )

        self.index_output_button.clicked.connect(
            self._browse_index_output
        )

        self.index_combo.currentIndexChanged.connect(
            self._update_index_defaults
        )

        self.index_mode_combo.currentIndexChanged.connect(
            self._update_index_mode
        )

        self.run_classification_button.clicked.connect(
            self.run_classification
        )

        self.run_index_button.clicked.connect(
            self.run_spectral_index
        )


    # ==================================================
    # INTERFACE HELPERS
    # ==================================================

    def _create_file_row(self, line_edit, button):
        """
        Create a line-edit and browse-button row.
        """

        widget = QWidget()

        layout = QHBoxLayout(widget)

        layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        layout.addWidget(line_edit)

        layout.addWidget(button)

        return widget


    def _training_layer_changed(self, layer):
        """
        Update the available class fields when the
        training layer changes.
        """

        if isinstance(layer, QgsVectorLayer):

            self.class_field_combo.setLayer(layer)

            self.class_field_combo.setEnabled(True)

        else:

            self.class_field_combo.setLayer(None)

            self.class_field_combo.setEnabled(False)


    def _update_algorithm_options(self):
        """
        Show the correct parameter panel for the
        selected classifier.
        """

        algorithm = self.algorithm_combo.currentData()

        is_rf = algorithm == "random_forest"

        self.rf_group.setVisible(is_rf)

        self.svm_group.setVisible(not is_rf)

        self.standardize_checkbox.setChecked(
            not is_rf
        )

        if is_rf:

            self.standardize_checkbox.setToolTip(
                "Feature scaling is generally not required "
                "for Random Forest."
            )

        else:

            self.standardize_checkbox.setToolTip(
                "Feature scaling is strongly recommended "
                "for SVM classification."
            )


    def _update_index_mode(self):
        """Enable the second raster only for change detection."""

        is_change = (
            self.index_mode_combo.currentData() == "change"
        )

        self.index_raster_2_combo.setEnabled(is_change)

        if is_change:
            self.run_index_button.setText(
                "Calculate Before, After and Change Raster"
            )
            self.index_output_edit.setPlaceholderText(
                "Output path for the delta index GeoTIFF"
            )
        else:
            self.run_index_button.setText(
                "Calculate Spectral Index"
            )
            self.index_output_edit.setPlaceholderText(
                "Output path for the index GeoTIFF"
            )


    def _update_index_defaults(self):
        """
        Set default band numbers and formula explanation
        for each spectral index.

        The numbers refer to raster stack positions,
        not necessarily original satellite band names.
        """

        index_name = self.index_combo.currentData()

        if index_name == "NDVI":

            self.index_band_a_spin.setValue(4)

            self.index_band_b_spin.setValue(3)

            self.index_formula_label.setText(
                "(NIR − RED) / (NIR + RED)"
            )

        elif index_name == "NDWI":

            self.index_band_a_spin.setValue(2)

            self.index_band_b_spin.setValue(4)

            self.index_formula_label.setText(
                "(GREEN − NIR) / (GREEN + NIR)"
            )

        elif index_name == "NBR":

            self.index_band_a_spin.setValue(4)

            self.index_band_b_spin.setValue(6)

            self.index_formula_label.setText(
                "(NIR − SWIR2) / (NIR + SWIR2)"
            )

        elif index_name == "SAVI":

            self.index_band_a_spin.setValue(4)

            self.index_band_b_spin.setValue(3)

            self.index_formula_label.setText(
                "1.5 × (NIR − RED) / "
                "(NIR + RED + 0.5)"
            )

        elif index_name == "GNDVI":

            self.index_band_a_spin.setValue(4)

            self.index_band_b_spin.setValue(2)

            self.index_formula_label.setText(
                "(NIR − GREEN) / (NIR + GREEN)"
            )

        elif index_name == "NDMI":

            self.index_band_a_spin.setValue(4)

            self.index_band_b_spin.setValue(5)

            self.index_formula_label.setText(
                "(NIR − SWIR1) / (NIR + SWIR1)"
            )


    # ==================================================
    # OUTPUT FILE DIALOGS
    # ==================================================

    def _browse_classified_output(self):
        """
        Select the classified raster output.
        """

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Classified Raster",
            "",
            "GeoTIFF (*.tif *.tiff)"
        )

        if path:

            path = self._ensure_extension(
                path,
                ".tif"
            )

            self.classified_output_edit.setText(path)

            if not self.report_output_edit.text().strip():

                report_path = os.path.splitext(path)[0]

                report_path += "_accuracy_report.txt"

                self.report_output_edit.setText(
                    report_path
                )


    def _browse_report_output(self):
        """
        Select the accuracy report output.
        """

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Accuracy Report",
            "",
            "Text file (*.txt)"
        )

        if path:

            path = self._ensure_extension(
                path,
                ".txt"
            )

            self.report_output_edit.setText(path)


    def _browse_index_output(self):
        """
        Select the spectral-index output.
        """

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spectral Index",
            "",
            "GeoTIFF (*.tif *.tiff)"
        )

        if path:

            path = self._ensure_extension(
                path,
                ".tif"
            )

            self.index_output_edit.setText(path)


    @staticmethod
    def _ensure_extension(path, extension):
        """
        Add a file extension when missing.
        """

        valid_extensions = (
            ".tif",
            ".tiff"
        )

        if extension == ".tif":

            if not path.lower().endswith(
                valid_extensions
            ):
                path += extension

        elif not path.lower().endswith(
            extension.lower()
        ):

            path += extension

        return path


    # ==================================================
    # INPUT VALIDATION
    # ==================================================

    def _validate_classification_inputs(self):
        """
        Validate all classification inputs.

        Returns
        -------
        bool
            True when inputs are valid.
        """

        raster_layer = self.raster_combo.currentLayer()

        training_layer = self.training_combo.currentLayer()

        class_field = self.class_field_combo.currentField()

        output_path = (
            self.classified_output_edit
            .text()
            .strip()
        )

        report_path = (
            self.report_output_edit
            .text()
            .strip()
        )

        if not isinstance(raster_layer, QgsRasterLayer):

            self._warning(
                "Select a valid multiband raster."
            )

            return False

        if not raster_layer.isValid():

            self._warning(
                "The selected raster is not valid."
            )

            return False

        if not isinstance(training_layer, QgsVectorLayer):

            self._warning(
                "Select a valid polygon training layer."
            )

            return False

        if not training_layer.isValid():

            self._warning(
                "The selected training layer is not valid."
            )

            return False

        if not class_field:

            self._warning(
                "Select the field containing the class labels."
            )

            return False

        if training_layer.fields().indexFromName(
            class_field
        ) < 0:

            self._warning(
                "The selected class field does not exist."
            )

            return False

        if not output_path:

            self._warning(
                "Select an output GeoTIFF."
            )

            return False

        if not report_path:

            self._warning(
                "Select an output accuracy report."
            )

            return False

        if raster_layer.bandCount() < 1:

            self._warning(
                "The raster does not contain readable bands."
            )

            return False

        return True


    def _validate_index_inputs(self):
        """Validate spectral-index and change-detection inputs."""

        raster_layer = self.index_raster_combo.currentLayer()
        raster_layer_2 = self.index_raster_2_combo.currentLayer()
        mode = self.index_mode_combo.currentData()
        output_path = self.index_output_edit.text().strip()

        if not isinstance(raster_layer, QgsRasterLayer):
            self._warning("Select a valid first raster layer.")
            return False

        if not raster_layer.isValid():
            self._warning("The first raster is not valid.")
            return False

        if mode == "change":
            if not isinstance(raster_layer_2, QgsRasterLayer):
                self._warning("Select a valid second raster layer.")
                return False

            if not raster_layer_2.isValid():
                self._warning("The second raster is not valid.")
                return False

        band_a = self.index_band_a_spin.value()
        band_b = self.index_band_b_spin.value()

        for number, layer in ((1, raster_layer), (2, raster_layer_2)):
            if number == 2 and mode != "change":
                continue
            if band_a > layer.bandCount() or band_b > layer.bandCount():
                self._warning(
                    f"Raster {number} has {layer.bandCount()} bands, "
                    f"but bands {band_a} and {band_b} were requested."
                )
                return False

        if band_a == band_b:
            self._warning("Band A and Band B must be different.")
            return False

        if not output_path:
            self._warning("Select an output GeoTIFF.")
            return False

        return True


    # ==================================================
    # MODEL PARAMETERS
    # ==================================================

    def _classification_parameters(self):
        """
        Collect all selected classification parameters.

        Returns
        -------
        dict
            Classification settings.
        """

        algorithm = self.algorithm_combo.currentData()

        parameters = {
            "algorithm": algorithm,
            "test_fraction":
                self.test_fraction_spin.value(),
            "random_state":
                self.random_state_spin.value(),
            "max_samples_per_class":
                self.max_samples_spin.value(),
            "standardize":
                self.standardize_checkbox.isChecked()
        }

        if algorithm == "random_forest":

            max_depth = (
                self.rf_max_depth_spin.value()
            )

            if max_depth == 0:

                max_depth = None

            parameters.update(
                {
                    "n_estimators":
                        self.rf_trees_spin.value(),
                    "max_depth":
                        max_depth,
                    "min_samples_split":
                        self.rf_min_samples_split_spin.value(),
                    "min_samples_leaf":
                        self.rf_min_samples_leaf_spin.value(),
                    "class_weight":
                        self.rf_class_weight_combo.currentData()
                }
            )

        elif algorithm == "svm":

            parameters.update(
                {
                    "c":
                        self.svm_c_spin.value(),
                    "gamma":
                        self.svm_gamma_combo.currentData(),
                    "probability":
                        self.svm_probability_checkbox.isChecked()
                }
            )

        return parameters


    # ==================================================
    # CLASSIFICATION EXECUTION
    # ==================================================

    def run_classification(self):
        """
        Run Random Forest or SVM classification.
        """

        if not self._validate_classification_inputs():

            return

        raster_layer = self.raster_combo.currentLayer()

        training_layer = self.training_combo.currentLayer()

        class_field = self.class_field_combo.currentField()

        output_path = (
            self.classified_output_edit
            .text()
            .strip()
        )

        report_path = (
            self.report_output_edit
            .text()
            .strip()
        )

        parameters = self._classification_parameters()

        self._set_processing_state(True)

        self._log(
            "Starting supervised classification..."
        )

        self._log(
            f"Algorithm: {parameters['algorithm']}"
        )

        self._log(
            f"Raster: {raster_layer.name()}"
        )

        self._log(
            f"Training layer: {training_layer.name()}"
        )

        self._log(
            f"Class field: {class_field}"
        )

        try:

            from .ml_engine import ClassificationEngine

            engine = ClassificationEngine(
                progress_callback=self._update_progress,
                log_callback=self._log
            )

            result = engine.run(
                raster_layer=raster_layer,
                training_layer=training_layer,
                class_field=class_field,
                output_path=output_path,
                report_path=report_path,
                parameters=parameters,
                save_model=(
                    self.save_model_checkbox.isChecked()
                )
            )

            if self.add_result_checkbox.isChecked():

                result_layer = QgsRasterLayer(
                    output_path,
                    os.path.splitext(
                        os.path.basename(output_path)
                    )[0]
                )

                if result_layer.isValid():

                    QgsProject.instance().addMapLayer(
                        result_layer
                    )

                else:

                    self._log(
                        "The output was created but could "
                        "not be loaded automatically."
                    )

            accuracy = result.get(
                "overall_accuracy"
            )

            if accuracy is not None:

                accuracy_text = f"{accuracy:.4f}"

            else:

                accuracy_text = "Not available"

            self._log(
                "Classification completed successfully."
            )

            self._information(
                "Classification completed.\n\n"
                f"Overall accuracy: {accuracy_text}\n"
                f"Raster: {output_path}\n"
                f"Report: {report_path}"
            )

        except ImportError as error:

            self._critical(
                "A required Python package or plugin module "
                "is missing.\n\n"
                f"{error}"
            )

        except Exception as error:

            self._critical(
                "Classification failed.\n\n"
                f"{error}"
            )

        finally:

            self._set_processing_state(False)


    # ==================================================
    # SPECTRAL INDEX EXECUTION
    # ==================================================

    def run_spectral_index(self):
        """Calculate a single index or a two-date change raster."""

        if not self._validate_index_inputs():
            return

        raster_layer = self.index_raster_combo.currentLayer()
        raster_layer_2 = self.index_raster_2_combo.currentLayer()
        mode = self.index_mode_combo.currentData()
        index_name = self.index_combo.currentData()
        band_a = self.index_band_a_spin.value()
        band_b = self.index_band_b_spin.value()
        output_path = self.index_output_edit.text().strip()

        self._set_processing_state(True)

        try:
            from .indices import SpectralIndexCalculator

            calculator = SpectralIndexCalculator(
                progress_callback=self._update_progress,
                log_callback=self._log
            )

            created_paths = []

            if mode == "change":
                self._log(f"Calculating {index_name} change detection...")

                result = calculator.calculate_change(
                    raster_layer_before=raster_layer,
                    raster_layer_after=raster_layer_2,
                    index_name=index_name,
                    band_a=band_a,
                    band_b=band_b,
                    delta_output_path=output_path
                )
                created_paths = [
                    result["before"],
                    result["after"],
                    result["delta"]
                ]
            else:
                self._log(f"Calculating {index_name}...")
                calculator.calculate(
                    raster_layer=raster_layer,
                    index_name=index_name,
                    band_a=band_a,
                    band_b=band_b,
                    output_path=output_path
                )
                created_paths = [output_path]

            if self.index_add_result_checkbox.isChecked():
                for path in created_paths:
                    layer_name = os.path.splitext(os.path.basename(path))[0]
                    result_layer = QgsRasterLayer(path, layer_name)
                    if result_layer.isValid():
                        QgsProject.instance().addMapLayer(result_layer)
                    else:
                        self._log(
                            f"Output created but could not be loaded: {path}"
                        )

            self._log(f"{index_name} processing completed successfully.")
            outputs_text = "\n".join(created_paths)
            self._information(
                f"{index_name} processing completed.\n\nOutputs:\n"
                f"{outputs_text}"
            )

        except ImportError as error:
            self._critical(
                "The spectral-index module is missing.\n\n"
                f"{error}"
            )
        except Exception as error:
            self._critical(
                f"{index_name} calculation failed.\n\n{error}"
            )
        finally:
            self._set_processing_state(False)


    # ==================================================
    # PROGRESS AND LOGGING
    # ==================================================

    def _set_processing_state(self, processing):
        """
        Enable or disable controls while processing.
        """

        self.run_classification_button.setEnabled(
            not processing
        )

        self.run_index_button.setEnabled(
            not processing
        )

        self.close_button.setEnabled(
            not processing
        )

        if processing:

            self.progress_bar.setValue(0)

        else:

            if self.progress_bar.value() < 100:

                self.progress_bar.setValue(0)


    def _update_progress(self, value):
        """
        Update the progress bar.

        Parameters
        ----------
        value : int or float
            Progress value from 0 to 100.
        """

        value = max(
            0,
            min(
                100,
                int(value)
            )
        )

        self.progress_bar.setValue(value)


    def _log(self, message):
        """
        Add a message to the log panel.
        """

        self.log_box.appendPlainText(
            str(message)
        )


    # ==================================================
    # MESSAGE BOXES
    # ==================================================

    def _warning(self, message):
        """
        Display a warning message.
        """

        QMessageBox.warning(
            self,
            "QGeoAI Toolkit",
            message
        )


    def _information(self, message):
        """
        Display an information message.
        """

        QMessageBox.information(
            self,
            "QGeoAI Toolkit",
            message
        )


    def _critical(self, message):
        """
        Display a critical-error message.
        """

        self._log(message)

        QMessageBox.critical(
            self,
            "QGeoAI Toolkit",
            message
        )