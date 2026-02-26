import logging
from pathlib import Path

import bites.gui.gui_utils as gui_utils

logger = logging.getLogger(__name__)
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QMessageBox,
                             QGridLayout, QFrame, QScrollArea, QWidget, QComboBox,
                             QApplication)
from PySide6.QtGui import QFont, QRegularExpressionValidator
from PySide6.QtCore import QRect, QRegularExpression, Qt

from bites.gui.mixins import DirtyTrackingMixin, GeometryPersistenceMixin
from bites.gui.widgets import PathInputWidget, PathType
from bites.constants import (
    DIVIDER_AFTER_PARAMS_TOOTHMODEL,
    DIVIDER_AFTER_PARAMS_ALPHABETICAL,
    DEFAULT_TM_SUBFOLDER_NAME,
    SAVED_BY_SHAPE_BROWSER_FOLDER,
)

class DefaultValuesWindow(GeometryPersistenceMixin, DirtyTrackingMixin, QDialog):
    GEOMETRY_KEY = "params_fixed"
    #DEFAULT_GEOMETRY = (865, 60, 200, 910)
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    def __init__(self, config, parent=None):
        super().__init__(parent)

        self.config = config

        try:
            self.default_values_dict = self.config['params_fixed_values'].copy()
            self.params_to_optimize = self.config['params_to_optimize'].copy() # used instead of optimization_ranges_dict
            self.all_hyperparameter_names = self.config['param_names_all'].copy()
            self.Abi_value_for_Pbi_too = self.config['bias_param_config']['Abi_value_for_Pbi_too']
            self.Bbi_value_for_Lbi_too = self.config['bias_param_config']['Bbi_value_for_Lbi_too']
        except KeyError:
            raise ValueError("The configuration dictionary must contain 'params_fixed_values', 'params_optimization_ranges', and 'param_names_all' keys.")

        self.initUI()

        # Initialize dirty tracking after UI is built
        self._init_dirty_tracking()
        self._connect_dirty_signals(self.value_fields)

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence()

    def initUI(self) -> None:
        # Main layout for the dialog
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Set a monospace font
        self.font = QFont("Courier New", 9)

        self.value_fields = {}

        # === SCROLLABLE CONTENT ===
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setFrameShape(QFrame.NoFrame)

        self._scroll_content = QWidget()
        scroll_layout = QVBoxLayout(self._scroll_content)
        scroll_layout.setContentsMargins(0, 0, 6, 0)  # Right margin for scrollbar

        # Parameter ordering combo box
        ordering_layout = QHBoxLayout()
        ordering_label = QLabel("Order:")
        ordering_label.setToolTip(
            'Choose how to sort the parameters in the list.'
        )
        ordering_layout.addWidget(ordering_label)
        self.ordering_combo = QComboBox()
        self.ordering_combo.addItems(["Tooth model", "Alphabetical"])
        self.ordering_combo.setToolTip(
            'Select parameter ordering:\n'
            '• Tooth model: Groups by function (growth, diffusion, etc.)\n'
            '• Alphabetical: Sorted A-Z for easy lookup'
        )
        # set default selection to "Tooth model"
        self.ordering_combo.setCurrentIndex(0)
        self.ordering_combo.currentIndexChanged.connect(self._rebuild_param_grid)
        ordering_layout.addWidget(self.ordering_combo)
        ordering_layout.addStretch()
        scroll_layout.addLayout(ordering_layout)

        # Container widget and layout for the parameter grid (will be rebuilt on ordering change)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self._build_param_grid()
        scroll_layout.addWidget(self.grid_container)

        # Add vertical stretch to push content up when window is tall
        scroll_layout.addStretch()

        self._scroll_area.setWidget(self._scroll_content)
        main_layout.addWidget(self._scroll_area, stretch=1)  # Scroll area expands

        # === FIXED BOTTOM SECTION (outside scroll) ===
        # Divider line
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(divider)

        # Option: Import from text file
        self.import_path_widget = PathInputWidget(
            parent=self,
            path_type=PathType.TXT,
            label="Import values from a text file:",
            default_value="",
        )
        self.import_path_widget.line_edit.setFixedWidth(70)
        self.import_path_widget.browse_button.setToolTip(
            'Select a text file to import parameter values from.\n\n'
            'USE THIS TO:\n'
            '• Load parameter values from a previous tooth model run\n'
            '• Use published parameter sets from research papers\n'
            '• Quickly restore a known good configuration\n\n'
            'COMPATIBLE FORMATS:\n'
            '• Text files output by ToothMaker/MorphoMaker ("Egr==0.0255"...)\n'
            '• Text files output by the tooth model used in BITES ("0.0255 Egr"... a.k.a. "Humppa format")\n'
            f'  (see subfolder "{DEFAULT_TM_SUBFOLDER_NAME}" in any output folder)\n'
            f'  (see also "{SAVED_BY_SHAPE_BROWSER_FOLDER}" folders)\n\n'
            'Parameters not found in the file keep their current values.'
        )
        self.import_path_widget.line_edit.setToolTip(
            'Path to a text file containing parameter values to import.\n'
            'Select a file using the Browse button.'
        )
        self.import_path_widget.line_edit.textChanged.connect(self._on_import_path_changed)
        main_layout.addWidget(self.import_path_widget)

        # Button to save the settings - always visible at bottom
        save_button = QPushButton("Save Settings")
        save_button.setToolTip(
            'Save fixed parameter values and return to the Config Editor.\n\n'
            'These values are used for parameters that are NOT being optimized.\n'
            'They remain constant throughout all optimization trials.'
        )
        save_button.clicked.connect(self.save_settings)
        main_layout.addWidget(save_button)

        self.setWindowTitle('Fixed parameter values')

        # Apply screen-aware sizing
        self._apply_screen_aware_sizing()

    def _apply_screen_aware_sizing(self) -> None:
        """Set window size constraints based on available screen space."""
        screen = QApplication.primaryScreen().availableGeometry()

        # Set minimum size so window remains usable
        self.setMinimumSize(200, 400)

        # Set maximum height to 85% of screen height
        max_height = int(screen.height() * 0.85)
        self.setMaximumHeight(max_height)

        # Calculate ideal height to fit scroll content without scrolling
        # Get the scroll content's size hint (actual content size)
        content_height = self._scroll_content.sizeHint().height()

        # Add height of fixed elements outside scroll area (buttons, labels, margins)
        fixed_elements_height = 120  # Approximate height of bottom section + margins

        ideal_height = content_height + fixed_elements_height

        # Cap at maximum screen height
        target_height = min(ideal_height, max_height)

        # Resize to fit content (width from sizeHint, calculated height)
        self.resize(self.sizeHint().width(), target_height)

    def _get_ordered_params(self) -> list:
        """Return parameters in the currently selected order."""
        if self.ordering_combo.currentIndex() == 0:  # Tooth model order
            return self.all_hyperparameter_names
        else:  # Alphabetical order
            return sorted(self.all_hyperparameter_names, key=str.lower)

    def _build_param_grid(self) -> None:
        """Build the parameter grid with current ordering."""
        full_param_name_dict = self.get_full_param_name_dict()
        is_tooth_model_order = self.ordering_combo.currentIndex() == 0

        row = 0
        for key in self._get_ordered_params():
            # if the parameter is set to be optimized, disable the default value field
            if key in self.params_to_optimize:
                enabled = False
            else:
                enabled = True
                # special case for the bias parameters:
                if key == 'Pbi' and self.Abi_value_for_Pbi_too:
                    enabled = False
                if key == 'Lbi' and self.Bbi_value_for_Lbi_too:
                    enabled = False

            try:
                default_value = self.default_values_dict[key]
            except KeyError:
                logger.warning(f"Key '{key}' not found in default values dictionary. Using default value of 0.0.")
                default_value = 0.0

            # Value input
            value_field = QLineEdit(str(default_value))
            value_field.setFixedWidth(60)
            value_field.setFont(self.font)
            value_field.setEnabled(enabled)
            self.value_fields[key] = value_field
            value_field.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$")))
            label = QLabel(f"{key}")
            label.setToolTip(full_param_name_dict[key])
            self.grid_layout.addWidget(label, row, 0)
            self.grid_layout.addWidget(value_field, row, 1)

            row += 1

            # Add dividers only in tooth model order
            if is_tooth_model_order and key in DIVIDER_AFTER_PARAMS_TOOTHMODEL or not is_tooth_model_order and key in DIVIDER_AFTER_PARAMS_ALPHABETICAL:
                line = QFrame()
                line.setGeometry(QRect(60, 110, 751, 20))
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                self.grid_layout.addWidget(line, row, 0, 1, 5)
                row += 1

    def _rebuild_param_grid(self) -> None:
        """Rebuild the parameter grid when ordering changes."""
        # Store current field values
        field_values = {key: field.text() for key, field in self.value_fields.items()}

        # Clear existing widgets from grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Clear dictionaries
        self.value_fields.clear()

        # Rebuild grid
        self._build_param_grid()

        # Restore field values
        for key in self.all_hyperparameter_names:
            if key in field_values:
                self.value_fields[key].setText(field_values[key])

    def save_settings(self) -> None:

        valid = True
        
        for key in self.all_hyperparameter_names:
            try:
                # Reset background color before checking validity (in case it was previously invalid)
                if self.value_fields[key].styleSheet() == "background-color: pink;":
                    self.value_fields[key].setStyleSheet("")

                value = float(self.value_fields[key].text())
                self.default_values_dict[key] = value
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Invalid input for {key}. Please enter a number.")
                
                # Highlight invalid fields with a red background color
                self.value_fields[key].setStyleSheet("background-color: pink;")
                
                valid = False
        
        if valid:
            self._reset_dirty()
            self.accept() # Close the dialog window

    def closeEvent(self, event) -> None:
        """Handle window close - prompt if unsaved changes."""
        if self._prompt_unsaved_changes(event):
            # GeometryPersistenceMixin.closeEvent saves geometry automatically
            super().closeEvent(event)

    def _on_import_path_changed(self, file_path: str) -> None:
        """Handle import path change - import values from the selected file."""
        if not file_path or not Path(file_path).exists():
            return

        try:
            imported_default_values_dict = gui_utils.dict_from_txt_file(file_path)
            self.update_values_from_imported_dict(imported_default_values_dict)
            logger.info(f"Imported parameter values from file: {file_path}")

            missing = [k for k in self.value_fields if k not in imported_default_values_dict]
            if missing:
                QMessageBox.information(
                    self, "Missing Parameters",
                    f"The imported file did not contain values for the following "
                    f"parameters (their current values were kept):\n\n"
                    + ", ".join(missing)
                )
        except ValueError as e:
            QMessageBox.warning(self, "Invalid File", f"Error importing values from file: {e}")
    
    def update_values_from_imported_dict(self, imported_default_values_dict) -> None:
        """Updates the default values dictionary with the values from the imported dictionary."""
        #self.default_values_dict.update(imported_default_values_dict) # This would add new keys to the dictionary
        for key, value in imported_default_values_dict.items():
            if key in self.value_fields.keys():
                rounded_value = gui_utils.set_decimal_places(value)
                self.value_fields[key].setText(rounded_value) # Update the value field
            # NOTE actually, we don't need to update the dictionary here - it will be updated when the settings are saved
            #if key in self.default_values_dict.keys(): # because the imported dictionary may contain additional keys (e.g. 'cells' and 'iterations')
                #self.default_values_dict[key] = value # Update the default values dictionary 

    def get_full_param_name_dict(self) -> dict:
        return {
                "Egr": "Epithelial proliferation rate.",
                "Mgr": "Mesenchymal proliferation rate.",
                "Rep": "Young's modulus (stiffness).",
                "Swi": "Distance from 0 where the borders are defined.",
                "Adh": "Traction between neighbours.",
                "Act": "Activator auto-activation.",
                "Inh": "Inhibition of activator.",
                "Sec": "Growth factor secretion rate.",
                "Da": "Activator diffusion rate.",
                "Di": "Inhibitor diffusion rate.",
                "Ds": "Growth factor diffusion rate.",
                "Int": "Initial inhibitor threshold.",
                "Set": "Growth factor threshold.",
                "Boy": "Mesenchyme mechanic resistance.",
                "Dff": "Differentiation rate.",
                "Bgr": "Border growth, amount of mes. in ant.-post.",
                "Abi": "Anterior bias.",
                "Pbi": "Posterior bias.",
                "Lbi": "Lingual bias.",
                "Bbi": "Buccal bias.",
                "Rad": "Radius of initial conditions.",
                "Deg": "Protein degradation rate.",
                "Dgr": "Downward vector of growth.",
                "Ntr": "Mechanical traction from the borders to the nucleus.",
                "Bwi": "Width of border.",
                "Ina": "Initial activator concentration.",
                "uMgr": "Basal mesenchymal proliferation rate."
                }
