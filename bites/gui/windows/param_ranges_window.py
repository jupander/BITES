import logging

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QMessageBox,
                             QLabel, QLineEdit, QCheckBox, QPushButton,
                             QGridLayout, QFrame, QScrollArea, QWidget, QComboBox,
                             QApplication)
from PySide6.QtGui import QFont, QRegularExpressionValidator
from PySide6.QtCore import QRect, Qt, QRegularExpression

from bites.gui.mixins import DirtyTrackingMixin, GeometryPersistenceMixin
from bites.constants import (
    DIVIDER_AFTER_PARAMS_TOOTHMODEL,
    DIVIDER_AFTER_PARAMS_ALPHABETICAL
)

logger = logging.getLogger(__name__)


class OptimizationRangesWindow(GeometryPersistenceMixin, DirtyTrackingMixin, QDialog):
    GEOMETRY_KEY = "param_ranges"
    #DEFAULT_GEOMETRY = (865, 60, 240, 985)
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Search space')

        self.config = config
        try:
            self.optimization_ranges_dict = self.config['params_optimization_ranges'].copy()
            self.all_hyperparameter_names = self.config['param_names_all'].copy()
        except KeyError:
            raise KeyError("The config dictionary must contain the 'params_optimization_ranges' and 'param_names_all' keys.")
        try:
            self.Abi_value_for_Pbi_too = self.config['bias_param_config']['Abi_value_for_Pbi_too']
            self.Abi_Pbi_factor = self.config['bias_param_config']['Abi_Pbi_factor']
            self.Bbi_value_for_Lbi_too = self.config['bias_param_config']['Bbi_value_for_Lbi_too']
            self.Bbi_Lbi_factor = self.config['bias_param_config']['Bbi_Lbi_factor']
        except KeyError:
            logger.warning("One or more of the Abi_Pbi or Bbi_Lbi settings not found in the config dictionary.")
            self.Abi_value_for_Pbi_too = False
            self.Abi_Pbi_factor = 1.0
            self.Bbi_value_for_Lbi_too = False
            self.Bbi_Lbi_factor = 1.0
        try:
            self.params_to_optimize = self.config['params_to_optimize'].copy()
        except KeyError:
            logger.warning("'params_to_optimize' not found in the config dictionary. Using all parameters.")
            self.params_to_optimize = self.all_hyperparameter_names.copy()
        
        self.initUI()

        # Initialize dirty tracking after UI is built
        self._init_dirty_tracking()
        self._connect_dirty_signals(
            self.checkboxes,
            self.min_fields,
            self.max_fields,
            [self.Abi_Pbi_checkbox, self.Bbi_Lbi_checkbox,
             self.multiplier_Abi_Pbi, self.multiplier_Bbi_Lbi]
        )

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence()

    def initUI(self) -> None:
        # Main layout for the dialog
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Set a monospace font
        self.font = QFont("Courier New", 9)

        self.checkboxes = {}
        self.min_fields = {}
        self.max_fields = {}
        self.multipliers = {}

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
            'Choose how to sort the parameters in the list.\n\n'
            '"Tooth model" order groups related parameters together.\n'
            '"Alphabetical" sorts parameters by name A-Z.'
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

        # Toggle all checkbox
        self.toggle_all_checkbox = QCheckBox("Toggle all")
        self.toggle_all_checkbox.setTristate(True)  # Allow indeterminate state
        self.toggle_all_checkbox.setToolTip(
            'Quickly select or deselect all parameters.\n\n'
            'Click to toggle between:\n'
            '• All selected (checked)\n'
            '• None selected (unchecked)'
        )
        self.toggle_all_checkbox.clicked.connect(self.on_toggle_all_clicked)
        self._update_toggle_all_state()  # Set initial state
        scroll_layout.addWidget(self.toggle_all_checkbox)

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

        # Label for the Abi/Pbi multiplier
        label = QLabel("Link Ant.-Post. or Buccal-Lingual biases")
        label.setToolTip(
            'Link bias parameters to reduce the effective search space.\n\n'
            'WHAT ARE BIAS PARAMETERS?\n'
            '• Abi (Anterior bias) and Pbi (Posterior bias) control front-back asymmetry\n'
            '• Bbi (Buccal bias) and Lbi (Lingual bias) control cheek-tongue asymmetry\n\n'
            'WHY LINK THEM?\n'
            'Linking reduces the number of parameters to optimize:\n'
            '• Pbi = Abi × multiplier: Only optimize Abi, Pbi follows automatically\n'
            '• Lbi = Bbi × multiplier: Only optimize Bbi, Lbi follows automatically\n\n'
            'MULTIPLIER = 1.0: Creates symmetric shapes (Abi = Pbi, Bbi = Lbi)\n'
            'MULTIPLIER ≠ 1.0: Creates controlled asymmetry'
        )
        main_layout.addWidget(label)

        # Horizontal layout for Abi/Pbi multiplier
        abi_pbi_layout = QHBoxLayout()
        self.Abi_Pbi_checkbox = QCheckBox("Pbi = Abi * multiplier:")
        self.Abi_Pbi_checkbox.setToolTip(
            'Link Posterior bias (Pbi) to Anterior bias (Abi).\n\n'
            'When enabled, Pbi is automatically set based on Abi:\n'
            'Pbi = Abi × multiplier\n\n'
            'This reduces the search space by 1 dimension and ensures\n'
            'a consistent relationship between front and back growth.'
        )
        self.Abi_Pbi_checkbox.setChecked(self.Abi_value_for_Pbi_too and self.checkboxes['Abi'].isChecked())
        self.Abi_Pbi_checkbox.setEnabled(self.checkboxes['Abi'].isChecked())
        self.Abi_Pbi_checkbox.stateChanged.connect(self.toggle_Abi_Pbi)
        abi_pbi_layout.addWidget(self.Abi_Pbi_checkbox)

        # Add stretcher to push the multiplier field to the right
        abi_pbi_layout.addStretch(2)

        # Label for the multiplier field (hidden, kept for compatibility)
        self.abi_label = QLabel("Pbi = Abi * multiplier:")
        self.abi_label.setEnabled(self.Abi_value_for_Pbi_too)

        self.multiplier_Abi_Pbi = QLineEdit()
        self.multiplier_Abi_Pbi.setFixedWidth(40)
        self.multiplier_Abi_Pbi.setFont(self.font)
        self.multiplier_Abi_Pbi.setText(str(self.Abi_Pbi_factor))
        self.multiplier_Abi_Pbi.setEnabled(self.Abi_Pbi_checkbox.isChecked())
        self.multiplier_Abi_Pbi.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$")))
        self.multiplier_Abi_Pbi.setToolTip(
            'Multiplier for the Abi-Pbi relationship.\n\n'
            'Pbi = Abi × (this value)\n\n'
            '• 1.0 = Symmetric (Pbi equals Abi)\n'
            '• > 1.0 = Posterior bias is stronger than anterior\n'
            '• < 1.0 = Anterior bias is stronger than posterior'
        )
        self.multiplier_Abi_Pbi.textChanged.connect(self.sync_Abi_Pbi_multiplier)
        abi_pbi_layout.addWidget(self.multiplier_Abi_Pbi)

        main_layout.addLayout(abi_pbi_layout)

        # Horizontal layout for Bbi/Lbi multiplier
        bbi_lbi_layout = QHBoxLayout()
        self.Bbi_Lbi_checkbox = QCheckBox("Lbi = Bbi * multiplier:")
        self.Bbi_Lbi_checkbox.setToolTip(
            'Link Lingual bias (Lbi) to Buccal bias (Bbi).\n\n'
            'When enabled, Lbi is automatically set based on Bbi:\n'
            'Lbi = Bbi × multiplier\n\n'
            'This reduces the search space by 1 dimension and ensures\n'
            'a consistent relationship between cheek-side and tongue-side growth.'
        )
        self.Bbi_Lbi_checkbox.setChecked(self.Bbi_value_for_Lbi_too and self.checkboxes['Bbi'].isChecked())
        self.Bbi_Lbi_checkbox.setEnabled(self.checkboxes['Bbi'].isChecked())
        self.Bbi_Lbi_checkbox.stateChanged.connect(self.toggle_Bbi_Lbi)
        bbi_lbi_layout.addWidget(self.Bbi_Lbi_checkbox)

        # Add stretcher to push the multiplier field to the right
        bbi_lbi_layout.addStretch(2)

        # Label for the multiplier field (hidden, kept for compatibility)
        self.bbi_label = QLabel("Lbi = Bbi * multiplier:")
        self.bbi_label.setEnabled(self.Bbi_value_for_Lbi_too)

        self.multiplier_Bbi_Lbi = QLineEdit()
        self.multiplier_Bbi_Lbi.setFixedWidth(40)
        self.multiplier_Bbi_Lbi.setFont(self.font)
        self.multiplier_Bbi_Lbi.setText(str(self.Bbi_Lbi_factor))
        self.multiplier_Bbi_Lbi.setEnabled(self.Bbi_Lbi_checkbox.isChecked())
        self.multiplier_Bbi_Lbi.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$")))
        self.multiplier_Bbi_Lbi.setToolTip(
            'Multiplier for the Bbi-Lbi relationship.\n\n'
            'Lbi = Bbi × (this value)\n\n'
            '• 1.0 = Symmetric (Lbi equals Bbi)\n'
            '• > 1.0 = Lingual bias is stronger than buccal\n'
            '• < 1.0 = Buccal bias is stronger than lingual'
        )
        self.multiplier_Bbi_Lbi.textChanged.connect(self.sync_Bbi_Lbi_multiplier)
        bbi_lbi_layout.addWidget(self.multiplier_Bbi_Lbi)

        main_layout.addLayout(bbi_lbi_layout)

        # Explanatory label (not needed with tooltips)
        #label = QLabel("(e.g. multiplier 1 should result\nin symmetrical shapes, etc.)")
        #label.setAlignment(Qt.AlignRight)
        #main_layout.addWidget(label)

        # Button to save the settings - always visible at bottom
        save_button = QPushButton("Save Settings")
        save_button.setToolTip(
            'Save parameter selection and ranges to return to the Config Editor.'
        )
        save_button.clicked.connect(self.save_settings)
        main_layout.addWidget(save_button)

        self.toggle_Abi_Pbi()  # initial refresh
        self.toggle_Bbi_Lbi()  # initial refresh

        # Apply screen-aware sizing
        self._apply_screen_aware_sizing()

    def _apply_screen_aware_sizing(self) -> None:
        """Set window size constraints based on available screen space."""
        screen = QApplication.primaryScreen().availableGeometry()

        # Set minimum size so window remains usable
        self.setMinimumSize(240, 450)

        # Set maximum height to 85% of screen height
        max_height = int(screen.height() * 0.85)
        self.setMaximumHeight(max_height)

        # Calculate ideal height to fit scroll content without scrolling
        # Get the scroll content's size hint (actual content size)
        content_height = self._scroll_content.sizeHint().height()

        # Add height of fixed elements outside scroll area (buttons, labels, margins)
        # Use layout spacing and margins as approximation
        fixed_elements_height = 180  # Approximate height of bottom section + margins

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

        # Add header row
        optimize_header = QLabel("Optimize")
        optimize_header.setToolTip(
            'Check the box to include this parameter in the optimization.\n\n'
            'Selected parameters will be varied by the optimizer to find\n'
            'values that produce shapes matching your target.\n\n'
            'Unselected parameters stay fixed at their default values (set in another window).'
        )
        self.grid_layout.addWidget(optimize_header, 0, 0)
        min_header = QLabel("Range: Min")
        min_header.setToolTip(
            'The MINIMUM value the optimizer will try for this parameter.\n\n'
            'The optimizer searches within [Min, Max] for optimal values.\n'
            'Narrower ranges = faster search but may miss good solutions.\n'
            'Wider ranges = more thorough but needs more trials.'
        )
        self.grid_layout.addWidget(min_header, 0, 2)
        max_header = QLabel("Max")
        max_header.setToolTip(
            'The MAXIMUM value the optimizer will try for this parameter.\n\n'
            'Together with Min, this defines the search range.\n'
            'Max must be greater than Min.'
        )
        self.grid_layout.addWidget(max_header, 0, 4)

        row = 2
        for key in self._get_ordered_params():
            param_range = self.optimization_ranges_dict.get(key, [0, 1])
            enabled = key in self.params_to_optimize

            # Checkbox to enable/disable the key
            checkbox = QCheckBox(key)
            checkbox.setChecked(enabled)
            checkbox.setToolTip(full_param_name_dict[key])
            checkbox.stateChanged.connect(self.toggle_key)
            self.checkboxes[key] = checkbox
            self.grid_layout.addWidget(checkbox, row, 0)

            # Min value input
            min_field = QLineEdit(str(param_range[0]))
            min_field.setFixedWidth(60)
            min_field.setFont(self.font)
            min_field.setEnabled(enabled)
            self.min_fields[key] = min_field
            min_field.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$")))
            min_field.textChanged.connect(self.sync_Abi_Pbi_multiplier)
            min_field.textChanged.connect(self.sync_Bbi_Lbi_multiplier)
            self.grid_layout.addWidget(min_field, row, 2)

            # Max value input
            max_field = QLineEdit(str(param_range[1]))
            max_field.setFixedWidth(60)
            max_field.setFont(self.font)
            max_field.setEnabled(enabled)
            self.max_fields[key] = max_field
            max_field.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$")))
            max_field.textChanged.connect(self.sync_Abi_Pbi_multiplier)
            max_field.textChanged.connect(self.sync_Bbi_Lbi_multiplier)
            self.grid_layout.addWidget(QLabel("-"), row, 3)
            self.grid_layout.addWidget(max_field, row, 4)

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
        # Store current checkbox states and field values
        checkbox_states = {key: cb.isChecked() for key, cb in self.checkboxes.items()}
        min_values = {key: field.text() for key, field in self.min_fields.items()}
        max_values = {key: field.text() for key, field in self.max_fields.items()}

        # Clear existing widgets from grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Clear dictionaries
        self.checkboxes.clear()
        self.min_fields.clear()
        self.max_fields.clear()

        # Rebuild grid
        self._build_param_grid()

        # Restore checkbox states and field values
        for key in self.all_hyperparameter_names:
            if key in checkbox_states:
                self.checkboxes[key].setChecked(checkbox_states[key])
            if key in min_values:
                self.min_fields[key].setText(min_values[key])
            if key in max_values:
                self.max_fields[key].setText(max_values[key])

        # Refresh toggle all state and Abi/Pbi, Bbi/Lbi toggles
        self._update_toggle_all_state()
        self.toggle_Abi_Pbi()
        self.toggle_Bbi_Lbi()

    def toggle_key(self) -> None:
        for key in self.all_hyperparameter_names:
            state = self.checkboxes[key].isChecked()
            self.min_fields[key].setEnabled(state)
            self.max_fields[key].setEnabled(state)
        # disable the sync checkbox and multiplier field if Abi / Bbi is unchecked
        if self.checkboxes['Abi'].isChecked():
            self.Abi_Pbi_checkbox.setEnabled(True)
        else:
            self.Abi_Pbi_checkbox.setEnabled(False)
            self.Abi_Pbi_checkbox.setChecked(False)
        if self.checkboxes['Bbi'].isChecked():
            self.Bbi_Lbi_checkbox.setEnabled(True)
        else:
            self.Bbi_Lbi_checkbox.setEnabled(False)
            self.Bbi_Lbi_checkbox.setChecked(False)
        # Update toggle all checkbox state
        self._update_toggle_all_state()

    def _update_toggle_all_state(self) -> None:
        """Update the toggle all checkbox to reflect current selection state."""
        checked_count = sum(1 for cb in self.checkboxes.values() if cb.isChecked())
        total_count = len(self.checkboxes)

        self.toggle_all_checkbox.blockSignals(True)
        if checked_count == 0:
            self.toggle_all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        elif checked_count == total_count:
            self.toggle_all_checkbox.setCheckState(Qt.CheckState.Checked)
        else:
            self.toggle_all_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        self.toggle_all_checkbox.blockSignals(False)

    def on_toggle_all_clicked(self) -> None:
        """Handle toggle all checkbox click - select all or deselect all."""
        # Qt tristate cycle: Unchecked -> PartiallyChecked -> Checked -> Unchecked
        # After click, current_state is the NEW state:
        # - Unchecked means user clicked FROM Checked -> deselect all
        # - PartiallyChecked/Checked means user clicked FROM Unchecked/Partial -> select all
        current_state = self.toggle_all_checkbox.checkState()

        # Block signals to avoid multiple toggle_key calls
        for checkbox in self.checkboxes.values():
            checkbox.blockSignals(True)

        if current_state == Qt.CheckState.Unchecked:
            # User clicked from Checked state -> deselect all
            for key in self.all_hyperparameter_names:
                self.checkboxes[key].setChecked(False)
        else:
            # User clicked from Unchecked or PartiallyChecked -> select all
            for key in self.all_hyperparameter_names:
                self.checkboxes[key].setChecked(True)
            self.toggle_all_checkbox.setCheckState(Qt.CheckState.Checked)

        for checkbox in self.checkboxes.values():
            checkbox.blockSignals(False)

        self.toggle_key()
        self._mark_dirty()

    def toggle_Abi_Pbi(self) -> None:
        if self.Abi_Pbi_checkbox.isChecked():
            # Force the Pbi checkbox to be checked and disabled
            self.checkboxes['Pbi'].setChecked(True)
            self.checkboxes['Pbi'].setEnabled(False)

            # Disable min/max fields for Pbi
            #self.min_fields['Pbi'].setEnabled(False)
            #self.max_fields['Pbi'].setEnabled(False)
            # instead, change color of the min/max fields for Pbi to indicate that they cannot be edited
            self.min_fields['Pbi'].setStyleSheet("background-color: lightgrey;")
            self.max_fields['Pbi'].setStyleSheet("background-color: lightgrey;")

            # Enable multiplier field
            self.abi_label.setEnabled(True)
            self.multiplier_Abi_Pbi.setEnabled(True)
            self.sync_Abi_Pbi_multiplier()
            
        else:
            # Enable Pbi checkbox if checkbox is unchecked
            self.checkboxes['Pbi'].setEnabled(True)

            # Enable Pbi fields if checkbox is unchecked
            #self.min_fields['Pbi'].setEnabled(True)
            #self.max_fields['Pbi'].setEnabled(True)
            # instead, change color of the min/max fields for Pbi to indicate that they can be edited
            self.min_fields['Pbi'].setStyleSheet("")
            self.max_fields['Pbi'].setStyleSheet("")
            
            # Disable multiplier field
            self.abi_label.setEnabled(False)
            self.multiplier_Abi_Pbi.setEnabled(False)
    
    def toggle_Bbi_Lbi(self) -> None:
        if self.Bbi_Lbi_checkbox.isChecked():
            # Force the Lbi checkbox to be checked and disabled
            self.checkboxes['Lbi'].setChecked(True)
            self.checkboxes['Lbi'].setEnabled(False)
            
            # Disable min/max fields for Lbi
            #self.min_fields['Lbi'].setEnabled(False)
            #self.max_fields['Lbi'].setEnabled(False)
            # instead, change color of the min/max fields for Lbi to indicate that they cannot be edited
            self.min_fields['Lbi'].setStyleSheet("background-color: lightgrey;")
            self.max_fields['Lbi'].setStyleSheet("background-color: lightgrey;")

            # Enable multiplier field
            self.bbi_label.setEnabled(True)
            self.multiplier_Bbi_Lbi.setEnabled(True)
            self.sync_Bbi_Lbi_multiplier()
        else:
            # Enable Lbi checkbox if checkbox is unchecked
            self.checkboxes['Lbi'].setEnabled(True)

            # Enable Lbi fields if checkbox is unchecked
            #self.min_fields['Lbi'].setEnabled(True)
            #self.max_fields['Lbi'].setEnabled(True)
            # instead, change color of the min/max fields for Lbi to indicate that they can be edited
            self.min_fields['Lbi'].setStyleSheet("")
            self.max_fields['Lbi'].setStyleSheet("")
            
            # Disable multiplier field
            self.bbi_label.setEnabled(False)
            self.multiplier_Bbi_Lbi.setEnabled(False)
    
    def sync_Abi_Pbi_multiplier(self) -> None:
        if self.Abi_Pbi_checkbox.isChecked():
            multiplier = self.get_multiplier(self.multiplier_Abi_Pbi)
            try:
                # abi_min = float(self.min_fields['Abi'].text())
                min_field = self.min_fields['Abi'].text()
                if not min_field:  # Check if the field is empty
                    #print("Warning: min field for Abi is empty, setting value to 0")
                    abi_min = 0
                else:
                    abi_min = float(min_field)
            except ValueError:
                logger.warning("ValueError in sync_Abi_Pbi_multiplier: min_fields['Abi'].text() is not a number")
                abi_min = 0
            try:
                #abi_max = float(self.max_fields['Abi'].text())
                max_field = self.max_fields['Abi'].text()
                if not max_field:  # Check if the field is empty
                    #print("Warning: max field for Abi is empty, setting value to 0")
                    abi_max = 0
                else:
                    abi_max = float(max_field)
            except ValueError:
                logger.warning("ValueError in sync_Abi_Pbi_multiplier: max_fields['Abi'].text() is not a number")
                abi_max = 0
            self.min_fields['Pbi'].setText(str(abi_min * multiplier))
            self.max_fields['Pbi'].setText(str(abi_max * multiplier))
    
    def sync_Bbi_Lbi_multiplier(self) -> None:
        if self.Bbi_Lbi_checkbox.isChecked():
            multiplier = self.get_multiplier(self.multiplier_Bbi_Lbi)
            try:                
                #bbi_min = float(self.min_fields['Bbi'].text())
                min_field = self.min_fields['Bbi'].text()
                if not min_field:  # Check if the field is empty
                    #print("Warning: min field for Bbi is empty, setting value to 0")
                    bbi_min = 0
                else:
                    bbi_min = float(min_field)
            except ValueError:
                logger.warning("ValueError in sync_Bbi_Lbi_multiplier: min_fields['Bbi'].text() is not a number")
                bbi_min = 0
            try:
                #bbi_max = float(self.max_fields['Bbi'].text())
                max_field = self.max_fields['Bbi'].text()
                if not max_field:  # Check if the field is empty
                    #print("Warning: max field for Bbi is empty, setting value to 0")
                    bbi_max = 0
                else:
                    bbi_max = float(max_field)
            except ValueError:
                logger.warning("ValueError in sync_Bbi_Lbi_multiplier: max_fields['Bbi'].text() is not a number")
                bbi_max = 0
            self.min_fields['Lbi'].setText(str(bbi_min * multiplier))
            self.max_fields['Lbi'].setText(str(bbi_max * multiplier))
    
    def get_multiplier(self, multiplier_field) -> float:
        try:
            return float(multiplier_field.text()) if multiplier_field.isEnabled() else 1.0
        except ValueError:
            return 1.0
    
    def save_settings(self) -> None:
        valid = True

        self.Abi_value_for_Pbi_too = self.Abi_Pbi_checkbox.isChecked()
        self.Abi_Pbi_factor = self.get_multiplier(self.multiplier_Abi_Pbi)
        self.Bbi_value_for_Lbi_too = self.Bbi_Lbi_checkbox.isChecked()
        self.Bbi_Lbi_factor = self.get_multiplier(self.multiplier_Bbi_Lbi)

        # new version: optimization_ranges_dict always contains all keys, but the program only uses the ones that are checked (i.e. in params_to_optimize)
        self.params_to_optimize = [] # reset the list of params to optimize
        for key in self.all_hyperparameter_names:
            if self.checkboxes[key].isChecked():

                if key == 'Pbi' and self.Abi_value_for_Pbi_too: # NOTE Abi_value_for_Pbi_too should never be checked if Abi is unchecked. We assume it here too.
                    # Do not add Pbi to the list of params to optimize, since it is linked to Abi
                    # Also don't bother with the min/max fields for Pbi. Keep the old values.
                    continue
                elif key == 'Lbi' and self.Bbi_value_for_Lbi_too: # NOTE Same as above
                    # Same as above
                    continue

                self.params_to_optimize.append(key) # add the key to the list of params to optimize

                # Check if values are valid and update the optimization_ranges_dict
                try:
                    # Reset background color before checking validity (in case it was previously invalid)
                    for field in [self.min_fields[key], self.max_fields[key]]:
                        if field.styleSheet() == "background-color: pink;":
                            field.setStyleSheet("")

                    min_val = float(self.min_fields[key].text())
                    max_val = float(self.max_fields[key].text())
                    
                    if min_val >= max_val:
                        raise ValueError(f"Min value for {key} must be less than max value.")
                    
                    # if the error is not raised, Update the optimization_ranges_dict
                    self.optimization_ranges_dict[key] = [min_val, max_val]

                except ValueError as e:
                    QMessageBox.warning(self, "Invalid Input", str(e))
                    
                    # Highlight invalid fields with a red background color
                    self.min_fields[key].setStyleSheet("background-color: pink;")
                    self.max_fields[key].setStyleSheet("background-color: pink;")
                    valid = False
            else: # if the checkbox is unchecked, do not include the key in the params_to_optimize
                # also do not bother with the min/max fields if the checkbox is unchecked - might be invalid, let's just keep the old values
                continue 
            
        if self.params_to_optimize == []: # if all checkboxes are unchecked
            QMessageBox.warning(self, "Invalid Input", "At least one parameter must be enabled.")
            valid = False

        # debug print
        #print(f"params_to_optimize ({len(self.params_to_optimize)}): {self.params_to_optimize}")
        
        if valid:
            # Reset dirty state and close the dialog window
            self._reset_dirty()
            self.accept()

    def closeEvent(self, event) -> None:
        """Handle window close - prompt if unsaved changes."""
        if self._prompt_unsaved_changes(event):
            # GeometryPersistenceMixin.closeEvent saves geometry automatically
            super().closeEvent(event)

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
