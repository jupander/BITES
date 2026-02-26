import optuna
import optuna.visualization as vis
from pathlib import Path
import plotly.io as pio
pio.renderers.default = "browser"

import bites.gui.gui_utils as gui_utils
from bites.config.config_loader import Config
from bites.database.pg_db_manager import ensure_database_for_target


def find_storage_url_from_config(output_folder: Path) -> str:
    """
    Infers the storage URL from the config YAML file in the given output folder.
    """
    # there can be multiple config files, so we need to find the most recent one
    config_files = list(output_folder.glob("config_*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No config file found in {output_folder}")
    # get the last one when sorted by name
    config_file = sorted(config_files)[-1]
    print(f"Reading config file: {config_file.name}")
    
    config = Config.from_yaml(str(config_file))

    if config.storage_mode == "shared":
        return ensure_database_for_target(config)
    ''' # NOTE TODO currently there are no alternatives, so this is not needed
    else:
        db_dir = output_folder.parent / "databases"
        db_name = f"optuna_{config.study_name}.db"
        return f"sqlite:///{db_dir / db_name}"
    '''

def load_study(output_folder: Path) -> optuna.Study:
    """
    Loads the Optuna study using the config and storage settings.
    """
    #config_file = next(output_folder.glob("config_*.yaml"), None)
    # there can be multiple config files, so we need to find the most recent one
    config_files = list(output_folder.glob("config_*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No config file found in {output_folder}")
    # get the last one when sorted by name
    config_file = sorted(config_files)[-1]
    print(f"Reading config file: {config_file.name}")

    #from bites.config.config_loader import Config # moved to top
    config = Config.from_yaml(str(config_file))
    storage_url = find_storage_url_from_config(output_folder)

    return optuna.load_study(
        study_name=config.study_name,
        storage=storage_url
    )

def show_visualizations(study: optuna.Study) -> None:
    """
    Opens interactive visualizations in a web browser.
    """
    print("Opening interactive visualizations...")
    # with warnings.catch_warnings(): # NOTE to suppress warnings about unique value length (did not work)
    #     warnings.filterwarnings("ignore", message="Param iter_max unique value length is less than 2.")
    print("Opening optimization history...")
    vis.plot_optimization_history(study).show()
    print("Done.\nOpening slice plot...")
    vis.plot_slice(study).show()
    print("Done.\nOpening parallel coordinate plot...")
    vis.plot_parallel_coordinate(study).show()
    print("Done.\nOpening parameter importances plot...")
    vis.plot_param_importances(study).show()
    print("Done.\nOpening contour plot...")
    vis.plot_contour(study).show()
    print("Done.\n")

def save_visualizations(study: optuna.Study, output_folder: Path) -> None:
    """
    Saves visualizations as HTML files inside the output folder.
    """
    output_folder = Path(output_folder).resolve()
    vis_dir = output_folder / "optuna_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "optimization_history.html": vis.plot_optimization_history,
        "slice.html": vis.plot_slice,
        "parallel_coordinate.html": vis.plot_parallel_coordinate,
        "param_importances.html": vis.plot_param_importances,
        "contour.html": vis.plot_contour,
    }

    for filename, plot_func in plots.items():
        try:
            fig = plot_func(study)
            #fig.write_html(str(vis_dir / filename))
            fig.write_html(str(vis_dir / filename), validate=False)  # Suppress validation errors
            print(f"Saved: {vis_dir / filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")

def save_and_show_visualizations(study: optuna.Study, output_folder: Path) -> None:
    """
    Saves and shows visualizations one by one.
    Each plot is saved first, then opened in a browser.
    Suppresses unnecessary warnings and gives detailed feedback.
    """
    output_folder = Path(output_folder).resolve()

    # Force plotly to use browser renderer
    pio.renderers.default = "browser"

    # Directory to save plots
    vis_dir = output_folder / "optuna_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Mapping of filenames to Optuna visualization functions
    plots = {
        "optimization_history.html": vis.plot_optimization_history,
        "slice.html": vis.plot_slice,
        "parallel_coordinate.html": vis.plot_parallel_coordinate,
        "param_importances.html": vis.plot_param_importances,
        "contour.html": vis.plot_contour,
    }

    print("\n=== Saving and Opening Optuna Visualizations ===")

    for filename, plot_func in plots.items():
        try:
            print(f"\nGenerating {filename.replace('.html', '').replace('_', ' ').title()}...")

            ''' # disabled this warning suppression, as it did not work
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", 
                    message=".*unique value length is less than 2.*", 
                    category=UserWarning
                )

                fig = plot_func(study)
            '''
            fig = plot_func(study)

            filepath = vis_dir / filename
            #fig.write_html(str(filepath))
            fig.write_html(str(filepath), validate=False)
            print(f"Saved: {filepath}")

            print(f"Opening {filename} in browser...")
            fig.show()
            print(f"Done.")

        except Exception as e:
            print(f"Failed to save/show {filename}: {e}")

    print("\n=== All visualizations processed ===\n")

def main(output_folder: Path) -> None:
    #while output_folder is None: # avoid loops
    if output_folder is None:
        print("Error: The folder path is None. Please select a valid output folder.")
        output_folder = Path(gui_utils.select_folder())
    if output_folder is None:
        print("Error: No folder selected. Exiting.")
        return

    if not output_folder.exists() or not output_folder.is_dir():
        print(f"Error: The folder path '{output_folder}' does not exist. Please select a valid output folder.")
        output_folder = Path(gui_utils.select_folder())
    if not output_folder.exists() or not output_folder.is_dir():
        print("Error: No valid folder selected. Exiting.")
        return
    
    study = load_study(output_folder)
    # Instead of these:
    # show_visualizations(study)
    # save_visualizations(study, output_folder)

    # Use the new combined function:
    save_and_show_visualizations(study, output_folder)

if __name__ == "__main__":
    # Example usage
    output_folder = None  # Set to None to prompt folder selection
    
    main(output_folder)
