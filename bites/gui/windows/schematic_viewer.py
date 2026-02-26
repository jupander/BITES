"""
Schematic viewer dialog for displaying the optimization process overview.

Extracted from config_editor.py for better separation of concerns.
"""

import logging

from PySide6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout,
    QPushButton, QDialog, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from bites.paths import get_asset


logger = logging.getLogger(__name__)

# Screen sizing constants
SCREEN_MAX_WIDTH_RATIO = 0.85
SCREEN_MAX_HEIGHT_RATIO = 0.90


class SchematicViewerDialog(QDialog):
    """Dialog for displaying the optimization process schematic."""

    SCHEMATIC_FILENAME = 'schematic_v6.png'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization process overview")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Load schematic image
        schematic_path = get_asset(self.SCHEMATIC_FILENAME)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.pixmap = None
        if schematic_path.exists():
            self.pixmap = QPixmap(str(schematic_path))
            if not self.pixmap.isNull():
                self._calculate_window_size(self.pixmap.width(), self.pixmap.height())
            else:
                logger.warning(f"Failed to load schematic image: {schematic_path}")
                self.image_label.setText("Failed to load schematic image")
                self.setMinimumSize(400, 300)
                self.pixmap = None
        else:
            logger.warning(f"Schematic file not found: {schematic_path}")
            self.image_label.setText(f"Schematic not found: {self.SCHEMATIC_FILENAME}")
            self.setMinimumSize(400, 300)

        # Add to scroll area for large images
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Close button
        #close_button = QPushButton("Close")
        #close_button.clicked.connect(self.accept)
        #layout.addWidget(close_button)

        self.setLayout(layout)

    def _calculate_window_size(self, img_width: int, img_height: int) -> None:
        """Calculate window size to fit screen while preserving image aspect ratio."""
        if img_width <= 0 or img_height <= 0:
            self.setMinimumSize(800, 600)
            return

        # Get available screen size
        screen = QApplication.primaryScreen().availableGeometry()
        max_width = int(screen.width() * SCREEN_MAX_WIDTH_RATIO)
        max_height = int(screen.height() * SCREEN_MAX_HEIGHT_RATIO)

        # Account for window decorations and button
        margin_height = 80

        aspect_ratio = img_width / img_height

        # Calculate size that fits within screen bounds while preserving aspect ratio
        available_height = max_height - margin_height
        if img_width <= max_width and img_height <= available_height:
            target_width = img_width
            target_height = img_height
        else:
            # Scale down to fit
            if max_width / available_height > aspect_ratio:
                target_height = available_height
                target_width = int(target_height * aspect_ratio)
            else:
                target_width = max_width
                target_height = int(target_width / aspect_ratio)

        self.setMinimumSize(400, int(400 / aspect_ratio) + margin_height)
        self.resize(target_width + 20, target_height + margin_height)

    def showEvent(self, event) -> None:
        """Set scaled pixmap when dialog is first shown."""
        super().showEvent(event)
        self._update_scaled_pixmap()

    def resizeEvent(self, event) -> None:
        """Rescale image when window is resized."""
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        """Update the label with a scaled version of the pixmap."""
        if self.pixmap is not None and not self.pixmap.isNull():
            # Get the scroll area's viewport size for scaling
            scroll_area = self.image_label.parent().parent()
            if isinstance(scroll_area, QScrollArea):
                available_size = scroll_area.viewport().size()
            else:
                available_size = self.image_label.size()

            scaled = self.pixmap.scaled(
                available_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
