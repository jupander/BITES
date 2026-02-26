import sys
import os

# Set VisPy backend BEFORE any other imports
# This must happen before PySide6 or any module that imports VisPy
os.environ['VISPY_BACKEND'] = 'pyside6'

import pstats
import cProfile
import traceback
import logging
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QTextEdit, QPushButton
from PySide6.QtGui import QIcon
from PySide6.QtCore import QSettings
from pathlib import Path

# Import core application infrastructure
from bites.core.logging_config import setup_default_logging
from bites.core.application_controller import ApplicationController
from bites.paths import get_results_folder

# Legacy imports (for CLI subprocess modes)
from bites.core.run_optimization import run_optimization

from time import sleep

performance_analysis = False


def _set_app_icon(app: QApplication) -> None:
    """Set the application-wide window icon so all windows and dialogs inherit it."""
    try:
        from bites.paths import get_asset
        icon_path = get_asset('bites_icon.png')
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass

def show_exception_dialog(exc_type, exc_value, exc_traceback) -> None:
    # Only show GUI dialog if QApplication exists, otherwise print to console
    if QApplication.instance() is not None:
        # Create a custom dialog with a QTextEdit for selectable error display
        error_dialog = QDialog()
        error_dialog.setWindowTitle("An unexpected error occurred")

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        tb_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        text_edit.setPlainText(tb_text)
        text_edit.setMinimumSize(600, 300)

        layout.addWidget(text_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(error_dialog.accept)
        layout.addWidget(close_button)

        error_dialog.setLayout(layout)
        error_dialog.exec()
    else:
        # Fallback to console output if QApplication doesn't exist
        tb_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print("An unexpected error occurred:", file=sys.stderr)
        print(tb_text, file=sys.stderr)


# Install global exception hook before QApplication is created
sys.excepthook = show_exception_dialog


# ----------------

def start_performance_analysis() -> cProfile.Profile:
    # Create a profiler object (for performance analysis, not required for normal operation)
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    return profiler

def end_performance_analysis(profiler, print_profiler_stats = True) -> None:
    profiler.disable()  # Stop profiling
    # Create Stats object to store and analyze profiler data
    stats = pstats.Stats(profiler)
    #stats.sort_stats('cumulative').print_stats(100)  # Sort by 'cumulative' time and print the top 100 entries
    #stats.sort_stats('time').print_stats(100)  # Sort by 'time' and print the top 10 entries
    if print_profiler_stats:
        stats.print_stats('(run_optimization|objective_function|run_tooth_model_as_subprocess|align_meshes|perform_CPD_on_mesh|calculate_chamfer_distances)')  # Filter by function name pattern
    # Save the profiler data to a file
    stats.dump_stats('profile_results.prof')  # Save the profiler data to a file

# ----------------


def main() -> None:
    # CRITICAL: freeze_support() must be called at the start for multiprocessing
    # to work correctly in frozen executables (PyInstaller, cx_Freeze, etc.)
    from multiprocessing import freeze_support
    freeze_support()

    # Initialize logging
    # Check if we're in subprocess mode (different logging for subprocesses)
    is_subprocess = "--optimize" in sys.argv or "--shape-browser" in sys.argv or "--simplifier" in sys.argv or "--cut-tester" in sys.argv

    if not is_subprocess:
        # GUI mode - set up full logging
        setup_default_logging(debug=False)  # Set to True for debug logging
        logger = logging.getLogger(__name__)
        logger.info("Starting BITES application...")

    # Legacy settings initialization (for subprocess modes)
    settings = QSettings("Jupander", "BITES")

    if "--optimize" in sys.argv:
        try:
            print("Initiating optimization...")

            # Expect config path after --optimize
            try:
                path_index = sys.argv.index("--optimize") + 1
                if path_index >= len(sys.argv):
                    print("No config path provided after --optimize.")
                    sys.exit(2)
                config_path = sys.argv[path_index]
            except (ValueError, IndexError):
                print("Usage: main.py --optimize <config_path>")
                sleep(5)
                sys.exit(2)

            if performance_analysis:
                profiler = start_performance_analysis()

            results_root = get_results_folder()  # default output root folder
            study, results_folder_path, optimization_run_id = run_optimization(Path(config_path), results_root)

            # Save the output folder path to QSettings (stored as string, read back as string via SettingsManager)
            settings.setValue("results_folder_path", str(results_folder_path))
            settings.sync()  # <--- This is needed to ensure the settings are saved immediately
            #print(f"Optimization completed. Output folder: {results_folder_path}")

            print("\nThis window can be closed now.")
        except Exception as e:
            print(f"Optimization failed: {e}", file=sys.stderr)
            traceback.print_exc()
            sleep(10)
        finally:
            if performance_analysis and 'profiler' in locals():
                end_performance_analysis(profiler, print_profiler_stats=True)
            sys.exit(0) # NOTE only exits the current (sub)process (started in the OptimizationOutputWindow class) - not the whole QApplication


    # Add shape browser argument handling
    if "--shape-browser" in sys.argv:
        # Set up file logging for subprocess (stdout/stderr may not be visible)
        from bites.paths import get_logs_folder, ensure_folder_exists
        logs_folder = ensure_folder_exists(get_logs_folder())
        subprocess_log_path = logs_folder / "shape_browser_subprocess_output.log"

        # Configure logging to file for subprocess
        import logging as subprocess_logging
        subprocess_logging.basicConfig(
            level=subprocess_logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                subprocess_logging.FileHandler(subprocess_log_path, mode='w', encoding='utf-8'),
                subprocess_logging.StreamHandler(sys.stderr)  # Also print to stderr if visible
            ]
        )
        subprocess_logger = subprocess_logging.getLogger("shape_browser_subprocess")

        try:
            subprocess_logger.info("=" * 60)
            subprocess_logger.info("Shape browser subprocess starting...")
            subprocess_logger.info(f"Python executable: {sys.executable}")
            subprocess_logger.info(f"sys.argv: {sys.argv}")
            subprocess_logger.info(f"Frozen: {getattr(sys, 'frozen', False)}")
            subprocess_logger.info("=" * 60)

            path_index = sys.argv.index("--shape-browser") + 1
            if path_index >= len(sys.argv):
                subprocess_logger.error("No folder path provided after --shape-browser.")
                print("No folder path provided after --shape-browser.", file=sys.stderr)
                sys.exit(2)
            folder_path = sys.argv[path_index]
            subprocess_logger.info(f"Folder path: {folder_path}")

            # Verify folder exists
            if not Path(folder_path).exists():
                subprocess_logger.error(f"Folder does not exist: {folder_path}")
                print(f"Folder does not exist: {folder_path}", file=sys.stderr)
                sys.exit(2)

            subprocess_logger.info("Creating QApplication...")
            # Create QApplication for shape browser subprocess
            app = QApplication(sys.argv)
            _set_app_icon(app)
            subprocess_logger.info("QApplication created successfully")

            # Import and run shape browser directly
            subprocess_logger.info("Importing shape browser module...")
            import bites.gui.visualization.shape_browser as shape_browser
            subprocess_logger.info("Shape browser module imported successfully")

            subprocess_logger.info("Starting shape browser viewer...")
            shape_browser.run_results_viewer(folder_path)
            subprocess_logger.info("Shape browser viewer started, entering event loop...")

            # Run the event loop and exit properly
            exit_code = app.exec()
            subprocess_logger.info(f"Event loop exited with code: {exit_code}")
            sys.exit(exit_code)
        except Exception as e:
            error_msg = f"Shape browser failed: {e}"
            subprocess_logger.error(error_msg, exc_info=True)
            print(error_msg, file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    # Add simplifier argument handling
    if "--simplifier" in sys.argv:
        try:
            path_index = sys.argv.index("--simplifier") + 1
            if path_index >= len(sys.argv):
                print("No mesh path provided after --simplifier.")
                sys.exit(2)
            mesh_path = sys.argv[path_index]

            # Create QApplication for simplifier subprocess
            app = QApplication(sys.argv)
            _set_app_icon(app)

            # Import and run simplifier dialog
            from bites.gui.windows.simplifier import MeshSimplifierDialog
            dialog = MeshSimplifierDialog(mesh_path)
            dialog.show()

            # Run the event loop and exit properly
            sys.exit(app.exec())
        except Exception as e:
            print(f"Simplifier failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Add cut tester argument handling
    if "--cut-tester" in sys.argv:
        try:
            path_index = sys.argv.index("--cut-tester") + 1
            if path_index + 1 >= len(sys.argv):
                print("Insufficient arguments after --cut-tester (need mesh_path and cut_height).")
                sys.exit(2)
            mesh_path = sys.argv[path_index]
            cut_height = sys.argv[path_index + 1]

            # Create QApplication for cut tester subprocess
            app = QApplication(sys.argv)
            _set_app_icon(app)

            # Import and run cut tester dialog
            from bites.gui.windows.cut_tester import MeshCutDialog
            dialog = MeshCutDialog(mesh_path, cut_height)
            dialog.show()

            # Run the event loop and exit properly
            sys.exit(app.exec())
        except Exception as e:
            print(f"Cut tester failed: {e}")
            traceback.print_exc()
            sys.exit(1)


    # Normal GUI mode - use new architecture
    logger.info("Starting GUI mode with ApplicationController")
    app = QApplication(sys.argv)
    _set_app_icon(app)

    # Create application controller (handles all coordination)
    controller = ApplicationController()

    # Run the application (creates and shows main window)
    controller.run()

    # Execute Qt event loop
    exit_code = app.exec()

    # Clean shutdown
    logger.info("Application exiting, performing cleanup...")
    controller.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Fallback in case of non-Gui error before QApplication is created
        show_exception_dialog(*sys.exc_info())
        """
        "Why sys.excepthook is Better for GUI Apps:
        - When a PyInstaller .exe crashes without a visible console, only sys.excepthook ensures all uncaught exceptions are handled in a consistent, centralized way.
        - It captures even errors from slots, signals, or startup issues not wrapped in try/except.
        - You only have to define it once."
        """