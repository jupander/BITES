import yaml
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Union, Optional

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QWidget, QMessageBox

from PySide6.QtWidgets import QFileDialog

from bites.paths import get_asset


def get_file_dialog_options() -> QFileDialog.Options:
    """Return QFileDialog options based on the user's native dialog preference."""
    from bites.core.settings_manager import SettingsManager
    settings = SettingsManager()
    if not settings.use_native_dialog:
        return QFileDialog.Option.DontUseNativeDialog
    return QFileDialog.Options()


def load_yaml_config(config_file_path: Union[str, Path]) -> dict:
    """Load a YAML configuration file and return its contents."""
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)


def select_folder(title: str = "Select a Folder", output_folders_folder: Optional[str] = None) -> Path:
    """
    Opens folder selection window, returns path of the selected folder.

    Parameters:
    - title: The title of the folder selection window.
    - output_folders_folder: The name of the folder where the output folders to choose from are kept.

    Returns:
    - Path: The path of the selected folder as a Path object.
    """
    # Initialize Tkinter
    root = tk.Tk()
    # Hide the main window (we don't want to show the full GUI)
    root.withdraw()

    # Get the current working directory
    cwd = Path.cwd()

    # Determine initial directory
    if output_folders_folder:
        initial_directory = cwd / output_folders_folder
    else:
        initial_directory = cwd

    # Make the window appear on top of other windows
    root.attributes("-topmost", True)

    # Open the file dialog to select a folder
    folder_selected = filedialog.askdirectory(initialdir=str(initial_directory), title=title)

    if folder_selected == "" or folder_selected is None:
        _logger.info("No folder selected.")
        return Path()  # Return empty Path
    else:
        _logger.info(f"Folder selected: {folder_selected}")
        return Path(folder_selected)



def set_decimal_places(value) -> str:
    """
    Set the amount of decimal places according to the value. Return the value as a string.
    """
    if value == 0: value_str = '0'
    elif value < 0.0000001: value_str = f'{value:.9f}'
    elif value < 0.000001: value_str = f'{value:.8f}'
    elif value < 0.00001: value_str = f'{value:.7f}'
    elif value < 0.0001: value_str = f'{value:.6f}'
    elif value < 0.001: value_str = f'{value:.5f}'
    elif value < 0.01: value_str = f'{value:.4f}'
    elif value < 0.1: value_str = f'{value:.3f}'
    elif value < 1: value_str = f'{value:.2f}'
    elif value < 10:
        if value % 1 < 0.001:
            value_str = f'{value:.0f}'
        else:
            value_str = f'{value:.2f}'
    else: # value >= 10
        value_str = f'{value:.0f}'
    return value_str


def _parse_humppa_format(lines: list[str]) -> dict:
    """Parse humppa/tooth_model format: ``value name`` per line.

    Lines like ``25 number of cells`` or ``1000 number of iterations`` and
    placeholder names (``-``) are skipped.
    """
    skip_phrases = {'number of cells', 'number of iterations'}
    values = []
    strings = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        # Skip metadata lines like "25 number of cells"
        rest = ' '.join(parts[1:]).lower()
        if rest in skip_phrases:
            continue
        value_part = parts[0]
        string_part = parts[-1]
        try:
            float(string_part)  # skip lines where the last token is also numeric
        except ValueError:
            strings.append(string_part)
        try:
            values.append(float(value_part))
        except ValueError:
            _logger.warning(f"Could not convert '{value_part}' to a number.")

    result = {}
    for name, value in zip(strings, values):
        if name == '-':
            continue
        if name not in result:
            result[name] = value
        else:
            new_name = name
            num = 2
            while new_name in result:
                new_name = f'{name}_{num}'
                num += 1
            result[new_name] = value
    return result


def _parse_toothmaker_format(lines: list[str]) -> dict:
    """Parse ToothMaker/MorphoMaker format: ``Name==value`` per line.

    Metadata keys (model, viewthresh, viewmode, iter) are skipped so that only
    actual model parameters are returned.
    """
    metadata_keys = {'model', 'viewthresh', 'viewmode', 'iter'}
    result = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if '==' not in stripped:
            continue
        name, _, raw_value = stripped.partition('==')
        if name.lower() in metadata_keys:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            _logger.warning(f"Could not convert '{raw_value}' to a number for parameter '{name}'.")
            continue
        if name not in result:
            result[name] = value
        else:
            new_name = name
            num = 2
            while new_name in result:
                new_name = f'{name}_{num}'
                num += 1
            result[new_name] = value
    return result


def dict_from_txt_file(file_path: Union[str, Path]) -> dict:
    """Reads a parameter text file and returns a dictionary.

    Automatically detects the format:
    - **ToothMaker format**: lines contain ``==`` (e.g. ``Egr==0.025``)
    - **Humppa format**: ``value name`` per line (e.g. ``0.025 Egr``)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Detect format: if any non-comment, non-empty line contains '==', it's ToothMaker format
    is_toothmaker = any(
        '==' in line for line in lines
        if line.strip() and not line.strip().startswith('#')
    )

    if is_toothmaker:
        return _parse_toothmaker_format(lines)
    else:
        return _parse_humppa_format(lines)


import logging

_logger = logging.getLogger(__name__)


class ErrorNotifier:
    """
    Centralized error notification for GUI applications.

    Provides consistent error handling that:
    - Logs errors for debugging
    - Shows user-facing dialogs when appropriate
    - Handles cases where parent widget may be destroyed
    """

    @staticmethod
    def show_error(
        parent: Optional[QWidget],
        title: str,
        message: str,
        details: Optional[str] = None,
        log_level: int = logging.ERROR
    ) -> None:
        """
        Show an error dialog and log the error.

        Args:
            parent: Parent widget (can be None or destroyed)
            title: Dialog title
            message: User-friendly error message
            details: Technical details (shown in expandable section)
            log_level: Logging level (default: ERROR)
        """
        # Always log
        log_msg = f"{title}: {message}"
        if details:
            log_msg += f"\nDetails: {details}"
        _logger.log(log_level, log_msg)

        # Try to show dialog
        ErrorNotifier._show_dialog(parent, title, message, details, QMessageBox.Critical)

    @staticmethod
    def show_warning(
        parent: Optional[QWidget],
        title: str,
        message: str,
        details: Optional[str] = None
    ) -> None:
        """Show a warning dialog and log it."""
        log_msg = f"{title}: {message}"
        if details:
            log_msg += f"\nDetails: {details}"
        _logger.warning(log_msg)

        ErrorNotifier._show_dialog(parent, title, message, details, QMessageBox.Warning)

    @staticmethod
    def show_info(
        parent: Optional[QWidget],
        title: str,
        message: str
    ) -> None:
        """Show an info dialog."""
        _logger.info(f"{title}: {message}")
        ErrorNotifier._show_dialog(parent, title, message, None, QMessageBox.Information)

    @staticmethod
    def _show_dialog(
        parent: Optional[QWidget],
        title: str,
        message: str,
        details: Optional[str],
        icon: QMessageBox.Icon
    ) -> None:
        """Internal method to safely show a dialog."""
        # Check if parent is valid
        safe_parent = None
        if parent is not None:
            try:
                # Check if widget is destroyed
                if not getattr(parent, '_is_closed', False):
                    # Try to access a property to verify widget exists
                    _ = parent.isVisible()
                    safe_parent = parent
            except RuntimeError:
                # Widget's C++ object has been deleted
                pass

        try:
            msg_box = QMessageBox(safe_parent)
            msg_box.setIcon(icon)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            if details:
                msg_box.setDetailedText(details)
            msg_box.exec()
        except Exception as e:
            # Last resort: just print to console
            print(f"[{title}] {message}")
            if details:
                print(f"Details: {details}")
            _logger.debug(f"Could not show dialog: {e}")

    @staticmethod
    def log_exception(context: str, exception: Exception) -> str:
        """
        Log an exception with context and return formatted details.

        Args:
            context: Description of what was being attempted
            exception: The caught exception

        Returns:
            Formatted traceback string for use in dialogs
        """
        import traceback
        details = traceback.format_exc()
        _logger.error(f"{context}: {exception}\n{details}")
        return details


def open_users_guide(parent: QWidget = None, page: int = None) -> bool:
    """
    Open the user's guide PDF in the system's default PDF viewer.

    Args:
        parent: Parent widget for error dialogs (optional).
        page: Page number to open to (1-indexed). Uses #page=N URL fragment
              which works with most PDF viewers (Adobe Reader, Chrome, Edge, etc.)

    Returns:
        True if the PDF was opened successfully, False otherwise.
    """
    guide_path = get_asset('BITES_users_guide.pdf')
    if guide_path.exists():
        url = QUrl.fromLocalFile(str(guide_path))
        if page is not None:
            url.setFragment(f"page={page}")
        if QDesktopServices.openUrl(url):
            return True
        else:
            if parent:
                QMessageBox.warning(parent, "Error", "Failed to open PDF viewer.")
            return False
    else:
        if parent:
            QMessageBox.warning(
                parent, "File Not Found",
                f"User's guide not found:\n{guide_path}"
            )
        return False
