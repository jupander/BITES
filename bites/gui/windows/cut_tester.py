#!/usr/bin/env python

# v6 - refactored to use cutting methods from statac module for consistency
# Following the successful shape_browser implementation pattern

"""
Mesh Cut Height Tester - Interactive tool for visualizing mesh cutting planes.

This module provides a dialog for testing cut height thresholds on 3D meshes:
- Interactive slider for adjusting cut height
- VisPy 3D visualization with color-coded parts (kept vs cut)
- Proper window management following shape_browser pattern

Uses cutting methods from bites.alignment.statac for consistency
with the actual alignment pipeline.
"""

import logging
import tempfile
import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QVBoxLayout, QSlider, QPushButton, QLabel,
                                QWidget, QHBoxLayout, QDialog,
                                QApplication)
from bites.paths import get_asset

import bites.mesh.read_target as read_target
from bites.gui.mixins import DirtyTrackingMixin, GeometryPersistenceMixin
from bites.gui.visualization.viewers.vispy_mesh_viewer import (
    VispyMeshViewer,
    DEFAULT_BG_COLOR,
    DEFAULT_LINE_WIDTH,
)
from bites.alignment.statac import perform_cut_on_target_mesh

logger = logging.getLogger(__name__)


class MeshCutVisualizer(QWidget):
    """Widget for visualizing mesh cuts with an interactive slider."""

    # Available cutting methods (maps display name -> statac method name)
    CUT_METHODS = {
        'Plane slice (clean edges)': 'plane_slice',
        'Face filter (rugged edges)': 'face_filter',
    }

    def __init__(self, mesh_path, default_cut_height):
        super().__init__()
        self.mesh_path = mesh_path
        self.default_cut_height = default_cut_height
        self.current_viewer = None
        self.target_mesh = None
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Slider for cut height
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setToolTip(
            'Drag to set the cut height limit (z-coordinate).\n\n'
            'During alignment, the optimizer may "cut off" parts of the target\n'
            'mesh below this height to better match shorter simulated shapes.\n\n'
            'The cut height is the MINIMUM z-value the cutting plane can reach.\n'
            '• Move LEFT: Allow cutting more of the mesh (lower z-value)\n'
            '• Move RIGHT: Allow cutting less (preserve more of the mesh)\n'
            '• Set to MAX: Disable cutting entirely (no mesh is cut)\n\n'
            'Click "Visualize cut" to see what will always be kept (yellow)\n'
            'versus what can be cut away if it improves alignment (blue).'
        )
        self.slider.valueChanged.connect(self.update_label)
        layout.addWidget(self.slider)

        # Button and label row
        controls_layout = QHBoxLayout()

        self.visualize_cut_button = QPushButton("Visualize cut")
        self.visualize_cut_button.setToolTip(
            'Visualize the maximum allowed crop on the target shape.\n\n'
            'Yellow: Part that will always be kept for shape similarity calculations\n'
            'Blue: Part that can be cut away if it improves alignment (root-side overshoot)'
        )
        self.visualize_cut_button.clicked.connect(self.visualize_cut)
        controls_layout.addWidget(self.visualize_cut_button)

        self.label = QLabel(f"Cut height: {self.default_cut_height}")
        self.label.setToolTip(
            'Set minimum z-coordinate for cutting.\n'
            'The cut position will not go below this value\n'
            '(i.e. toward tooth cusps, in the negative z direction).'
        )
        controls_layout.addWidget(self.label)

        controls_layout.addStretch()

        info_label = QLabel("(Set to max to disable cutting)")
        info_label.setAlignment(Qt.AlignRight)
        controls_layout.addWidget(info_label)

        layout.addLayout(controls_layout)

        # Method selector row (hidden for now - using plane_slice method)
        # To re-enable: uncomment this block and the method selection in visualize_cut()
        # method_layout = QHBoxLayout()
        # method_label = QLabel("Cut method:")
        # method_layout.addWidget(method_label)
        # self.method_combo = QComboBox()
        # self.method_combo.addItems(self.CUT_METHODS.keys())
        # self.method_combo.setToolTip(
        #     'Cutting method:\n\n'
        #     '• Plane slice: Creates new edges at cut boundary.\n'
        #     '  Clean cut, may add new vertices.\n\n'
        #     '• Face filter: Keeps/discards whole faces.\n'
        #     '  Rugged edge following original mesh topology.'
        # )
        # method_layout.addWidget(self.method_combo)
        # method_layout.addStretch()
        # layout.addLayout(method_layout)

        # Load initial mesh
        self.load_mesh(self.mesh_path, self.default_cut_height)

        self.setLayout(layout)
    
    def update_label(self) -> None:
        """Update label with current slider value."""
        self.label.setText(f"Cut height: {self.slider.value()}")


    def visualize_cut(self) -> None:
        """Visualize the cut using VisPy viewer."""
        if self.target_mesh is None:
            logger.warning("No mesh loaded")
            return

        cut_height = self.slider.value()

        # Use plane_slice method (same as actual alignment pipeline)
        # To use selectable method: uncomment method_combo in init_ui() and use:
        # method = self.CUT_METHODS[self.method_combo.currentText()]
        method = 'plane_slice'

        # Perform cut using statac module
        kept_mesh, cut_mesh = perform_cut_on_target_mesh(
            self.target_mesh, cut_height, method=method
        )

        # Close previous viewer
        if self.current_viewer is not None:
            try:
                self.current_viewer.close()
            except Exception as e:
                logger.debug(f"Error closing viewer: {e}")
            finally:
                self.current_viewer = None

        # Create new viewer with parent_handler
        self.current_viewer = VispyMeshViewer(
            parent_handler=self,
            title=f"Cut visualization [{method}] (Yellow = always matched, Blue = can be cut out)",
            size=(800, 800),
            show_axis=True,
            bgcolor=DEFAULT_BG_COLOR
        )

        # Display
        success = self.current_viewer.display_cut_visualization(
            kept_mesh=kept_mesh,
            cut_mesh=cut_mesh,
            kept_color=(1.0, 0.706, 0.0),  # Yellow for kept (cusp side)
            cut_color=(0.0, 0.651, 0.929),  # Blue for cut (root side)
            show_wireframe=True
        )

        if success:
            self.current_viewer.show_and_activate()
            logger.info(f"Visualization opened (method: {method}, cut_height: {cut_height})")
        else:
            logger.warning("Failed to display visualization")
            self.current_viewer = None

    def handle_key_press(self, key) -> None:
        """
        Handle key press events from the VisPy viewer.

        Parameters:
        -----------
        key : str
            The key name that was pressed
        """
        # Forward viewer-specific keys to the viewer's internal handler
        if self.current_viewer is not None:
            self.current_viewer._handle_viewer_key(key)

    
    def load_mesh(self, mesh_path, default_cut_height) -> None:
        """Load mesh from file and initialize slider."""
        self.mesh_path = mesh_path
        
        if not Path(mesh_path).exists():
            logger.warning(f"Mesh file not found: {mesh_path}")
            self.set_ui_enabled(False)
            return
        
        try:
            # Load mesh
            self.target_mesh = read_target.read_target_mesh_file(mesh_path)
            
            # Get bounds
            min_z = float(self.target_mesh.bounds[0][2])
            max_z = float(self.target_mesh.bounds[1][2])
            
            # Configure slider
            self.slider.setMinimum(int(np.floor(min_z)))
            self.slider.setMaximum(int(np.ceil(max_z)) + 1)
            
            # Set default value
            try:
                default_cut_height = int(default_cut_height)
            except (ValueError, TypeError):
                default_cut_height = int(np.ceil(max_z)) + 1 # default was None or invalid -> set to max (i.e. no cutting)
            
            self.slider.setValue(default_cut_height)
            self.update_label()
            self.set_ui_enabled(True)
            
            logger.info(f"Loaded mesh: {len(self.target_mesh.vertices)} vertices, "
                  f"{len(self.target_mesh.faces)} faces")
            logger.info(f"Z range: [{min_z:.2f}, {max_z:.2f}]")
            
        except Exception as e:
            logger.error(f"Error loading mesh: {e}")
            import traceback
            traceback.print_exc()
            self.set_ui_enabled(False)
    
    def set_ui_enabled(self, enabled) -> None:
        """Enable or disable UI elements."""
        self.slider.setEnabled(enabled)
        self.visualize_cut_button.setEnabled(enabled)
        self.label.setEnabled(enabled)
        
        if not enabled:
            self.label.setText("Mesh not loaded")
    
    def update_target_mesh_path(self, new_mesh_path, default_cut_height) -> None:
        """Update mesh path and reload."""
        self.load_mesh(new_mesh_path, default_cut_height)
    
    def closeEvent(self, event) -> None:
        """Clean up when widget is closed."""
        if self.current_viewer is not None:
            try:
                self.current_viewer.close()
            except RuntimeError as e:
                logger.debug(f"Viewer already closed: {e}")
            finally:
                self.current_viewer = None
        super().closeEvent(event)


class MeshCutDialog(GeometryPersistenceMixin, DirtyTrackingMixin, QDialog):
    """Dialog for interactive mesh cutting with visualization."""

    GEOMETRY_KEY = "mesh_cut_dialog"
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    def __init__(self, mesh_path, default_cut_height, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cut height limit tester")

        self.result_value = None  # Store the result
        self.initial_cut_height = default_cut_height  # Store initial value for dirty tracking

        self.visualizer = MeshCutVisualizer(mesh_path, default_cut_height)

        self.accept_button = QPushButton("Save and close")
        self.accept_button.setToolTip(
            'Save the current cut height setting and return to the Config Editor.\n\n'
            'The saved cut height will be used during optimization when STatac\n'
            'alignment algorithms are selected.'
        )
        self.accept_button.clicked.connect(self.on_accept)

        layout = QVBoxLayout()
        layout.addWidget(self.visualizer)
        layout.addWidget(self.accept_button)

        self.setLayout(layout)

        # Make it non-modal so VisPy works
        self.setModal(False)

        # Initialize dirty tracking - connect to slider
        self._init_dirty_tracking()
        self._connect_dirty_signals([self.visualizer.slider])

        # Apply screen-aware sizing
        self._apply_screen_aware_sizing()

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence()

    def _apply_screen_aware_sizing(self) -> None:
        """Set window size constraints based on available screen space."""
        screen = QApplication.primaryScreen().availableGeometry()

        # Set minimum size so window remains usable
        self.setMinimumSize(400, 150)

        # Set maximum height to 50% of screen height (this is a small tool dialog)
        self.setMaximumHeight(int(screen.height() * 0.50))

        # Let Qt calculate optimal size based on content
        self.adjustSize()

    def on_accept(self) -> None:
        """Handle accept button click with proper cleanup."""
        # Save result before closing
        self.result_value = self.visualizer.slider.value()

        # Write result to temporary file for IPC with main process
        try:
            ipc_file = Path(tempfile.gettempdir()) / 'bites_cut_height_result.txt'
            with open(ipc_file, 'w') as f:
                f.write(str(self.result_value))
            logger.info(f"Cut height saved to IPC file: {ipc_file}")
        except Exception as e:
            logger.error(f"Error writing cut height to IPC file: {e}")

        # Close the viewer first
        if self.visualizer.current_viewer is not None:
            try:
                self.visualizer.current_viewer.close()
            except Exception as e:
                logger.debug(f"Error closing viewer: {e}")
            finally:
                self.visualizer.current_viewer = None

        # Reset dirty state and accept the dialog
        self._reset_dirty()
        self.accept()

    def closeEvent(self, event) -> None:
        """Ensure viewer is closed when dialog is closed, prompt for unsaved changes."""
        # Check for unsaved changes first
        if not self._prompt_unsaved_changes(event):
            return  # User cancelled

        # Clean up viewer
        if self.visualizer.current_viewer is not None:
            try:
                self.visualizer.current_viewer.close()
            except Exception as e:
                logger.debug(f"Error closing viewer: {e}")
            finally:
                self.visualizer.current_viewer = None
        # GeometryPersistenceMixin.closeEvent saves geometry automatically
        super().closeEvent(event)


# Example usage
if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    from pathlib import Path
    
    qt_app = QApplication(sys.argv)
    
    mesh_path = Path(get_asset('benchmark_target.ply'))
    default_cut_height = None
    
    dialog = MeshCutDialog(mesh_path, default_cut_height)
    dialog.show()
    
    sys.exit(qt_app.exec())