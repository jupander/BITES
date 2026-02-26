import logging
import pandas as pd
from pathlib import Path
from typing import Union

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class FolderAnalysisWorker(QThread):
    """
    Background worker for analyzing results folder contents.

    Signals:
        finished_analysis: Emitted when analysis is complete (result_text)
        error: Emitted on error (error_message)
    """

    finished_analysis = Signal(str)  # Analysis result text
    error = Signal(str)  # Error message

    def __init__(self, folder_path: Union[str, Path], parent=None):
        """
        Initialize the worker.

        Args:
            folder_path: Path to the folder to analyze
            parent: Parent QObject
        """
        super().__init__(parent)
        self.folder_path = folder_path
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the worker."""
        self._cancelled = True

    def run(self) -> None:
        """Main worker thread execution."""
        try:
            if self._cancelled:
                return

            result = analyze_folder(self.folder_path)

            if not self._cancelled:
                self.finished_analysis.emit(result)

        except Exception as e:
            logger.error(f"Error analyzing folder: {e}", exc_info=True)
            if not self._cancelled:
                self.error.emit(str(e))


def analyze_folder(path: Union[str, Path]) -> str:
    """Analyze the given folder for subfolders and CSV files.
    Args:
        path: The path to the folder to analyze.
    Returns:
        str: A summary of the folder's contents.
    """
    folder_path = Path(path)

    # Ensure the path exists and is a directory
    if not folder_path.exists():
        logger.warning(f"Path does not exist: {path}")
        return ""
    if not folder_path.is_dir():
        logger.warning(f"Path is not a directory: {path}")
        return "Selected path is not a directory."

    # Validate the folder looks like a completed optimization output folder
    from bites.gui.visualization.shape_browser.utils import validate_folder_for_shape_browser
    problems = validate_folder_for_shape_browser(folder_path)
    if problems:
        bullet_list = "\n".join(f"  • {p}" for p in problems)
        return (
            "The selected folder does not look like a completed "
            "optimization output folder:\n\n"
            f"{bullet_list}\n\n"
            f"Folder: {folder_path}\n\n"
            "Please select a folder that contains the results of a "
            "completed optimization run."
        )

    output_lines = []

    # Collect subfolder names and their file counts
    subfolders_info = []
    optuna_vis_files = []
    csv_completed_trials = []

    for item_path in folder_path.iterdir():
        item = item_path.name

        # If it's a directory
        if item_path.is_dir():
            file_count = len([f for f in item_path.iterdir() if f.is_file()])
            subfolders_info.append(f"'{item}': {file_count} files")

            if item == "optuna_visualizations":
                optuna_vis_files = [f.name for f in item_path.iterdir()]

        # If it's a CSV file
        elif item.lower().endswith('.csv'):
            # skip backup CSV files (name includes 'ongoing')
            if 'ongoing' in item.lower():
                continue

            # Read CSV once with pandas to count completed trials
            try:
                df = pd.read_csv(item_path, encoding='utf-8', delimiter=';')
                if 'state' in df.columns:
                    complete_count = df[df['state'] == 'COMPLETE'].shape[0]
                    total_count = df.shape[0]
                    csv_completed_trials.append(f"'{item}':\n{complete_count} / {total_count}  trials completed\n")
                else:
                    csv_completed_trials.append(f"'{item}': 'state' column not found")
            except Exception as e:
                csv_completed_trials.append(f"CSV File: '{item}' could not be read ({e})")

    # Compose the output
    if csv_completed_trials:
        output_lines.append("Trials completed/started:")
        output_lines.extend(csv_completed_trials)

    if optuna_vis_files:
        output_lines.append("\nGenerated visualizations:")
        output_lines.extend(optuna_vis_files)
    else:
        output_lines.append("No 'optuna_visualizations' folder found or it's empty.")

    output_lines.append("\nSubfolders:")
    output_lines.extend(subfolders_info)

    return '\n'.join(output_lines)

if __name__ == "__main__":

    # Example usage
    # Replace this with your actual path
    folder_path = "/path/to/your/folder"
    analyze_folder(folder_path)
