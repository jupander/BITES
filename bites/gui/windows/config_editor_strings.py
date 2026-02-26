"""
Tooltip and label text constants for the configuration menu.

Extracted from config_editor.py to reduce file size and improve maintainability.
All tooltip strings are organized by their UI group/section.
"""

from bites.config.config_validation import MAX_STUDY_NAME_LENGTH
from bites.constants import DEFAULT_ITER_INTERVAL
from bites.paths import DEFAULT_RESULTS_FOLDER


# =============================================================================
# Study Group
# =============================================================================

STUDY_GROUP_TOOLTIP = (
    'Settings that identify and control the overall optimization study.\n\n'
    'WHAT IS A STUDY?\n'
    'A "study" is the overall optimization project. It includes:\n'
    '• All the trials (parameter evaluations) you\'ve run (across one or more optimization runs)\n'
    '• The database that stores these trials, and thus allows continuing optimization later\n'
    '• Effectively, the learned information about which parameters work well\n\n'
    'Separate optimization runs that share the same study name share information\n'
    'and build on each other\'s results.'
)

STUDY_NAME_LABEL_TOOLTIP = (
    "A unique name identifying this optimization study.\n\n"
    "WHY THIS MATTERS:\n"
    "• Optimization runs with the same study name share a database and learn from each other\n"
    "• The optimizer uses results from all previous runs with this name\n"
    "• This allows you to continue optimization across multiple sessions\n\n"
    "NAMING RECOMMENDATIONS:\n"
    "• Use descriptive names: 'seal_tooth_v1', 'human_molar_narrow_range'\n"
    "• Include version numbers etc. if iterating on the same target\n\n"
    "• Start a NEW study name when:\n"
    "  - Changing the target mesh (ALWAYS change the study name when doing this)\n"
    "  - Testing significantly different parameter sets and/or ranges (will mess up the learning)\n"
    "  - Changing the min/max iterations or the model timeout significantly (will also mess up the learning)\n"
    "  - Changing the fixed parameter values (will definitely mess up the learning)\n"
    "  - Starting fresh without prior information (try this when additional trials stop improving results)\n\n"
    "KEEP the same study name when:\n"
    "• Continuing a previous optimization,\n"
    "• i.e. running more trials on the same target,\n"
    "• i.e. wanting to build on previous results.\n"
    "• Only settings such as the number of trials, parallelism, optimization run timeout, etc. have changed."
)

STUDY_NAME_WIDGET_TOOLTIP = (
    f'Unique identifier for this study (max {MAX_STUDY_NAME_LENGTH} characters).\n'
    'Use letters, numbers, and underscores only.\n\n'
    'Same name = shared learning between optimization runs.\n'
    'Different name = independent optimization from scratch.'
)

STUDY_TIMEOUT_LABEL_TOOLTIP = (
    "Maximum total time for the optimization run, in hours (use . for decimals).\n\n"
    "WHAT THIS DOES:\n"
    "The optimization will stop after this many hours, even if it hasn't\n"
    "completed all requested trials. Set to 0 for no time limit.\n\n"
    "NOTE: The optimization will finish the current trial before stopping,\n"
    "so actual runtime may slightly exceed this limit."
)

STUDY_TIMEOUT_WIDGET_TOOLTIP = (
    'Maximum total runtime in hours. 0 = no limit.\n'
    'Useful for overnight runs or limited computing time.'
)


# =============================================================================
# Target Group
# =============================================================================

TARGET_GROUP_TOOLTIP = (
    'The 3D mesh that you want BITES to replicate.\n\n'
    'The optimization process will search for model parameters that produce\n'
    'simulated teeth most similar to this target shape.\n\n'
    'SUPPORTED FORMATS: .ply, .off, .stl, .obj, .glb, .gltf, .3mf\n\n'
    'TIPS FOR BEST RESULTS:\n'
    '• Ensure the mesh is oriented correctly: \n'
    '  (cusps pointing towards negative z-axis, ant.-post.-axis aligns with x-axis)\n'
    '• Simplify dense meshes for faster optimization\n'
    '• Set a cut height limit if the target includes parts that are less important to match'
)

TARGET_PATH_WIDGET_TOOLTIP = (
    'The path to the target mesh file (.ply, .off, .stl, .obj, .glb, .gltf, or .3mf format).\n\n'
    'This is the 3D shape that BITES will try to replicate by finding\n'
    'the best combination of model parameters.\n\n'
    'You can use a scan of a real tooth or any other 3D mesh you want the model\n'
    'to recreate. The mesh should ideally be simplified for faster processing -\n'
    'use the "Simplify target mesh" button if needed.'
)

SIMPLIFIER_BUTTON_TOOLTIP = (
    'OPTIONAL STEP: Simplify target mesh before optimization.\n\n'
    'Opens a tool for creating a simplified copy of the target mesh, which can then be used instead of the original.\n\n'
    'WHY SIMPLIFY?\n'
    'Dense meshes (many vertices/faces) can significantly slow down the optimization.\n'
    'Adequate simplifying reduces this computation time without losing important shape details.\n\n'
    'The tool shows how vertex/face count changes with different simplification levels\n'
    'and lets you preview the result in 3D before saving the new mesh.'
)

CUT_HEIGHT_BUTTON_TOOLTIP = (
    'OPTIONAL STEP: Set a cut height limit for the target shape.\n\n'
    "WHAT IS CUT HEIGHT?\n"
    "When using STatac alignment algorithms, the optimizer can 'cut off' the bottom\n"
    "portion of the target mesh (the root side) to better match simulated shapes\n"
    "that are shorter than the target.\n\n"
    "WHY IS THIS USEFUL?\n"
    "Real tooth scans often include parts of the root that the tooth model doesn't\n"
    "simulate. Without cutting, the optimizer would penalize the model output for not matching\n"
    "these root portions, even if it perfectly matches the crown.\n\n"
    "WHY SET A LIMIT?\n"
    "Without a limit, the algorithm could 'cheat' by cutting away most of the target,\n"
    "leaving only a tiny tip that's easy to match but meaningless.\n\n"
    "HOW TO CHOOSE A GOOD CUT HEIGHT:\n"
    "• Set the limit at the lowest point you care about matching\n"
    "• Usually this is where the crown meets the root\n"
    "• The visualization shows yellow (kept) vs blue (can be cut)\n\n"
    "DEFAULT: No cutting allowed (slider at maximum value).\n\n"
    "Only relevant when using STatac alignment algorithms."
)


# =============================================================================
# Parameter Group
# =============================================================================

PARAMETER_GROUP_TOOLTIP = (
    'Configure which model parameters the optimizer will vary.\n\n'
    'This is one of the most important settings. The optimizer searches through\n'
    'different combinations of parameter values to find those that produce\n'
    'simulated tooth shapes most similar to your target.\n\n'
    'SEARCH SPACE SIZE:\n'
    'The "search space" is the set of all possible parameter combinations.\n'
    'More parameters and wider ranges = larger search space.'
)

PARAMS_TO_OPT_LABEL_TOOLTIP = (
    'Shows which parameters are currently selected for optimization.\n\n'
    'Click "Select parameters to optimize" to change this selection.'
)

OPT_BUTTON_TOOLTIP = (
    'Opens a menu for selecting the model parameters to optimize and their search ranges.\n\n'
    'WHAT ARE PARAMETERS?\n'
    'The tooth model has 27 parameters that control different aspects of tooth development:\n'
    '• Growth rates (how fast different parts grow)\n'
    '• Diffusion rates (how signals spread through tissue)\n'
    '• Mechanical properties (stiffness, adhesion)\n'
    '• Initial conditions (starting shape and chemical concentrations)\n\n'
    'SELECTING PARAMETERS TO OPTIMIZE:\n'
    'The optimizer will try different values for selected parameters to find combinations\n'
    'that produce shapes matching your target. Parameters NOT selected stay fixed.\n\n'
    'TRADEOFF: More parameters/wider ranges -> larger search space, which means:\n'
    '+ More flexibility to match complex shapes\n'
    '- More trials needed to find specific solutions\n\n'
    'If you have prior knowledge about reasonable parameter values,\n'
    'use narrower ranges to speed up the search.'
)

DEFAULT_VALUES_BUTTON_TOOLTIP = (
    'Opens a menu for setting the fixed values for parameters not being optimized.\n\n'
    'These values are used for all trials as constants while the optimizer\n'
    'varies the selected parameters.\n\n'
    'You can also import parameter values from a text file (e.g., output from\n'
    'a previous tooth model run) using the import feature in the window.\n\n'
    'Parameters selected for optimization (in the other window) are\n'
    'disabled here since their values will be determined by the optimizer.'
)


# =============================================================================
# Iteration Group
# =============================================================================

ITER_GROUP_TOOLTIP = (
    'Controls how far each tooth model simulation runs.\n\n'
    'WHAT ARE ITERATIONS?\n'
    'Each "iteration" is one time step in the tooth development simulation.\n'
    'More iterations = longer simulated development = potentially larger/more complex teeth.\n'
    '(But the simulation also becomes exponentially more computationally expensive\n'
    'and slow as the number of iterations increases.)\n\n'
    'The optimizer will effectively treat the simulation length as another\n'
    'parameter to optimize: The model outputs the current state of the simulation\n'
    f'each time its iteration count reaches a multiple of {DEFAULT_ITER_INTERVAL}.\n'
    'Each of these output shapes is evaluated, and the best one is chosen as the trial result.'
)

MIN_ITERATIONS_TOOLTIP = (
    'Minimum number of iterations considered.\n\n'
    'Shapes output by the tooth model at earlier simulation stages than this\n'
    'will not be evaluated (aligned and matched), which saves computation time.'
)

MAX_ITERATIONS_TOOLTIP = (
    'For how many iterations the tooth model will run its simulation before stopping.\n\n'
    'Higher values allow longer simulated development, potentially producing\n'
    'larger or more complex tooth shapes, but increase computation time.\n\n'
    'Note that the tooth model timeout (set in the Persistence section below)\n'
    'may terminate simulations before they reach the maximum iteration count.'
)


# =============================================================================
# Duration/Persistence Group
# =============================================================================

DURATION_GROUP_TOOLTIP = (
    'Controls how long the optimization runs and how many trials to perform.\n\n'
    'WHAT IS A TRIAL?\n'
    'Each trial runs the tooth model once with a specific set of parameter values,\n'
    'compares the resulting shape to your target, and records the loss score.\n\n'
    'More trials = Better chance of finding good solutions, but takes longer.\n'
    'The optimizer learns from previous trials to make smarter guesses over time.'
)

TM_TIMEOUT_LABEL_TOOLTIP = (
    "Maximum time allowed for each tooth model simulation (one per trial) before it's terminated.\n\n"
    "Iteration stages already reached by the simulation will still be evaluated,\n"
    "but if the simulation doesn't finish within this time, it won't continue further.\n\n"
    "WHY IS THIS NEEDED?\n"
    "Some parameter combinations cause the simulation to run extremely slowly\n"
    "or get stuck in infinite loops. Without a timeout, a single bad trial could\n"
    "block the entire optimization for hours.\n\n"
    "CHOOSING A GOOD VALUE DEPENDS ON:\n"
    "• How many simulation iterations you're aiming for (see section above)\n"
    "• How powerful your system is (i.e. how many iterations per second it can run the model)\n"
    "• Your willingness to sacrifice speed for thoroughness\n\n"
    "TRADEOFF:\n"
    "• Too short: May kill legitimate slow simulations\n"
    "• Too long: Bad parameter combinations waste time\n\n"
    "TIP: See earlier results or run test simulations to estimate typical runtimes for each iteration value."
)

TM_TIMEOUT_WIDGET_TOOLTIP = (
    'Maximum seconds per trial before the simulation is terminated.\n\n'
    "Typical range: I've used 30-300 seconds, but HEAVILY depends on your system and settings.\n"
    "See label tooltip for guidance."
)

MAX_TRIALS_LABEL_TOOLTIP = (
    "Total number of trials (parameter value combination evaluations) to run.\n\n"
    "WHAT IS A TRIAL?\n"
    "Each trial tests one set of parameter values by running the tooth model\n"
    "and comparing the output shape(s) to your target. The optimizer learns from\n"
    "each trial to make better guesses in future trials.\n\n"
    "HOW MANY TRIALS DO I NEED?\n"
    "• Depends on how many parameters you're optimizing and their search ranges.\n"
    "• Rule of thumb: More optimized parameters = more trials needed.\n"
    "• Personally, I would not give up on a study before running at least (n_parameters * 20) trials.\n\n"
    "• You can always continue a study later with more trials if needed.\n\n"
    "EXTRA TIP: Starting a new study (i.e. changing the study name but keeping other settings the same) will yield\n"
    "different results, due to the stochastic/random nature of the optimization algorithm.\n"
    "Thus running e.g. 5 separate 2000 trial studies may yield better results than a single 10 000 trial study."
)

MAX_TRIALS_WIDGET_TOOLTIP = (
    'Number of trials to run. More trials = better results, but longer runtime (and often diminishing returns).\n'
    'Typical range: 100-1000 trials. See label tooltip for detailed guidance.'
)

ETA_LABEL_TOOLTIP = (
    'Estimated maximum time for the optimization to complete.\n\n'
    'This is a worst-case estimate assuming every tooth simulation runs until timeout.\n'
    'In practice, given most trials complete faster, the actual runtime is usually less.\n\n'
    'This estimate does not account for evaluation (alignment + loss calculation) time,\n'
    'but that is usually small compared to simulation time.\n\n'
    'If you have set a timeout for the optimization run itself,\n'
    'that time limit will always apply regardless of this estimate.'
)

ETA_VALUE_TOOLTIP = (
    'Estimated maximum runtime based on your current settings.\n\n'
    'CALCULATION:\n'
    '(Trials x Tooth model timeout per trial) / Parallel CPUs\n\n'
    'ACTUAL RUNTIME:\n'
    'Usually much less, given most trials complete before tooth model timeout.'
)


# =============================================================================
# Parallelization Group
# =============================================================================

PARALLEL_GROUP_TOOLTIP = (
    'Run multiple tooth model simulations simultaneously to speed up optimization.\n\n'
    'WHAT IS PARALLELIZATION?\n'
    'Instead of running one simulation at a time, the optimizer can run several\n'
    'simulations in parallel on different CPU cores. This significantly reduces\n'
    'the total time needed to complete all trials.\n\n'
    'IMPORTANT NOTES:\n'
    '• Leave 1-2 CPUs free for system tasks if using your computer during optimization\n'
    '• More parallelization uses more memory (RAM)'
)

def cpu_info_label_tooltip(num_cpus: int) -> str:
    return (
        'The number of CPU cores detected on this computer.\n\n'
        'This shows "logical" cores, which includes hyperthreading.\n'
        'Physical cores may be half this number, but using all logical\n'
        'cores is usually fine for this workload.\n\n'
        'NOTE: If running optimization on a different system (e.g. a computing\n'
        'cluster), you may want to set parallelization based on that system\'s\n'
        'resources, not this computer\'s.'
    )

PARALLEL_SEARCH_TOOLTIP = (
    'Enable parallel processing to run multiple simulations simultaneously.\n\n'
    'WHEN TO ENABLE:\n'
    '• Your computer has multiple CPU cores (most modern computers do)\n'
    '• You want faster optimization at the cost of higher CPU/memory usage'
)

def parallel_trials_tooltip(num_cpus: int) -> str:
    return (
        'Number of simulations to run simultaneously.\n\n'
        'RECOMMENDED SETTINGS:\n'
        f'• Your system has {num_cpus} logical CPUs\n'
        f'• Recommended: leave 1-2 CPUs for other system tasks\n'
        f'• Maximum useful: {num_cpus} (may slow down other applications)\n\n'
        'MEMORY CONSIDERATION:\n'
        'Each parallel simulation uses additional memory.\n'
        'If you experience slowdowns or crashes, reduce this number.\n\n'
        'NOTE: Values above your CPU count provide no benefit and may actually\n'
        'slow things down due to context switching overhead.'
    )


# =============================================================================
# Alignment Algorithms Group
# =============================================================================

ALIGNMENT_GROUP_TOOLTIP = (
    'Select which alignment algorithms to use when comparing simulated shapes to the target.\n'
    'Each selected algorithm will be tried for each trial, and the best loss reported.\n\n'
    'WHAT IS ALIGNMENT?\n'
    'Before calculating how different a simulated tooth is from the target, we need to\n'
    'align them (position, rotate, and scale them to match).\n'
    'Different algorithms do this in different ways with various tradeoffs.\n\n'
    'ABOUT ABBREVIATIONS:\n'
    'ST stands for "Scale and Translate".\n'
    'STatac stands for "Scale and Translate, align tops and cut".\n'
    'AB and C refer to different ways for defining the center of the shape during alignment, either based on\n'
    'the center of the axis-aligned bounding box (AB) or the mass center (centroid) of the mesh vertices (C).\n\n'
    'AVAILABLE ALGORITHMS:\n'
    '• ST: scale by average of all 3 axes, translate to align centroids. Fast.\n'
    '• STatac-AB: scale by average of xy-axes, translate to align top cusps, cut overshooting target if allowed. Fast.\n'
    '• STatac-C: scale by average of xy-axes, translate to align top cusps, cut overshooting target if allowed. Fast.\n'
    '• STatac-AB-CPD: STatac-AB followed by refinement using the Coherent Point Drift algorithm. Slow but accurate.\n'
    '• STatac-C-CPD: STatac-C followed by CPD refinement. Slow but accurate.\n\n'
    'ABOUT VARIANTS:\n'
    'The "-CPD" variants add a more sophisticated (but significantly slower) alignment step.\n'
    'The AB-centering method might work better for some complex shapes,\n'
    'but generally the C-centering is more robust and works better with asymmetric shapes.\n\n'
    'RECOMMENDATION:\n'
    '• Use STatac-C-CPD for best overall results (cutting can also be disabled by setting tha max cut height limit)\n'
    '• Other non-CPD methods can be selected as extras for comparison - they require basically no extra time,\n'
    '  but do create extra aligned mesh files in the output folder, which uses more disk space.\n'
    '• We would advice against using multiple CPD-based methods simultaneously, as they are slow and similar to each other.'
)

ALGO_TOOLTIPS = {
    'ST': 'Scale by average of all 3 axes, translate to align centroids. Fast.',
    'STatac-AB': 'Scale by average of xy-axes, translate to align top cusps, cut overshooting target if allowed. Fast.',
    'STatac-C': 'Scale by average of xy-axes, translate to align top cusps, cut overshooting target if allowed. Fast.',
    'STatac-AB-CPD': 'STatac-AB followed by Coherent Point Drift. Slow but accurate.',
    'STatac-C-CPD': 'STatac-C followed by Coherent Point Drift. Slow but accurate.',
}


# =============================================================================
# Output Folder Group
# =============================================================================

OUTPUT_GROUP_TOOLTIP = (
    'Options for what gets saved to the output folder.\n\n'
    'WHAT GETS SAVED BY DEFAULT:\n'
    '• Aligned mesh files (.ply): Simulated shapes after alignment\n'
    '• Parameter files (.txt): The parameter values used for each trial\n'
    '• CSV file: Summary of all trials with loss scores, parameters, etc.\n'
    '• Etc.\n\n'
    f'The output folder is created automatically in the "{DEFAULT_RESULTS_FOLDER}/" directory\n'
    'with a name based on your study name.'
)

SAVE_ORIGINAL_MESHES_TOOLTIP = (
    'Also save the original (unaligned) meshes from each tooth model simulation.\n\n'
    'WHEN TO DISABLE (recommended):\n'
    '• For most optimization runs - saves disk space\n'
    '• Aligned meshes and parameter files are usually sufficient\n'
    '• You can always re-run the tooth model using the saved parameter values if needed\n\n'
    'WHEN TO ENABLE:\n'
    '• When you need to analyze the raw simulation output'
)

OPEN_RESULTS_FOLDER_TOOLTIP = (
    'Automatically open the output folder in your file explorer when the optimization run completes.'
)


# =============================================================================
# Technical Settings Group
# =============================================================================

TECHNICAL_GROUP_TOOLTIP = (
    'Advanced configuration options for power users.\n\n'
    'These settings allow you to:\n'
    '• Specify custom path for the tooth model executable\n'
    '• Override default folder paths\n'
    '• Etc.\n\n'
    'MOST USERS DON\'T NEED THESE - the defaults work well for typical use.\n\n'
    'Enable this checkbox to access the advanced settings buttons.'
)

TECHNICAL_ENABLE_TOOLTIP = (
    'Check this box to enable access to advanced technical settings.\n\n'
    'Most users can leave this disabled - the default settings work\n'
    'well for typical optimization runs on a single computer.'
)

TECHNICAL_BUTTON_TOOLTIP = (
    'Opens advanced technical settings:\n\n'
    '• Model executable path: Custom location for the tooth model program\n'
    '• Results backup folder: Centralized location to collect copies of results\n'
    '• Config/Assets folders: Override default folder locations\n'
    '• Iteration interval: Step size for simulation length (usually 1000)\n\n'
    'These are typically only needed when running on non-standard setups\n'
    'or when organizing results across multiple projects.'
)

POSTGRES_BUTTON_TOOLTIP = (
    'Configure PostgreSQL database connection.'
)


# =============================================================================
# Save / Import / Export
# =============================================================================

SAVE_BUTTON_TOOLTIP = (
    'Save all your changes to the configuration file.\n\n'
    'After saving, you can close this window and launch the optimization\n'
    'from the Main Menu.\n\n'
    'If you want to save these settings to a DIFFERENT config file, use "Save as..." instead.'
)

EXPORT_BUTTON_TOOLTIP = (
    'Save the current settings as a new config file.'
)

IMPORT_BUTTON_TOOLTIP = (
    'Load settings from another configuration file.\n\n'
    'NOTE: This will replace ALL current settings with those from the\n'
    'imported file. Your current unsaved changes will be lost.\n\n'
    'The imported settings will appear in this window but won\'t be saved\n'
    'until you click "Save changes".'
)


# =============================================================================
# Help / Diagram buttons
# =============================================================================

HELP_BUTTON_TOOLTIP = (
    "Show optimization process overview\n\n"
    "Click to view a schematic diagram explaining\n"
    "how the optimization process works."
)

GUIDE_BUTTON_TOOLTIP = (
    "Open User's Guide\n\n"
    "Opens the PDF documentation in your\n"
    "system's default PDF viewer."
)
