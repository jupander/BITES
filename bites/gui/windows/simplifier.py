#!/usr/bin/env python

"""
Mesh simplifier tool - Interactive mesh simplification with visualization and benchmarking.

This module provides a GUI dialog for simplifying 3D meshes with:
- Interactive epsilon threshold selection
- Real-time visualization with VisPy 3D viewer
- Optional benchmarking with CPD algorithm
- Matplotlib-based statistics plotting

Refactored to use trimesh and VisPy instead of open3d.
"""

from __future__ import annotations
import logging
import time
import tempfile
import trimesh
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QSlider, QPushButton,
    QDoubleSpinBox, QHBoxLayout, QFileDialog, QMessageBox,
    QLineEdit, QCheckBox, QSpinBox, QProgressBar, QWidget,
    QApplication, QFrame, QToolButton
)
from bites.paths import get_asset
from PySide6.QtCore import Qt, QThread, Signal
import bites.gui.gui_utils as gui_utils

from bites.gui.mixins import GeometryPersistenceMixin

logger = logging.getLogger(__name__)

# Import after PySide6 to avoid conflicts
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import VisPy mesh viewer for 3D visualization
try:
    from bites.gui.visualization.viewers.vispy_mesh_viewer import (
        VispyMeshViewer,
        DEFAULT_BG_COLOR,
        DEFAULT_WIREFRAME_COLOR,
        DEFAULT_LINE_WIDTH,
    )
    VISPY_AVAILABLE = True
except ImportError:
    VISPY_AVAILABLE = False
    DEFAULT_BG_COLOR = '#2E3440'  # Fallback if import fails
    logger.warning("vispy_mesh_viewer not available. 3D visualization disabled.")

# Note: These imports may need adjustment based on your project structure
# from bites.alignment.coherent_point_drift import perform_CPD_on_mesh
# import bites.mesh.read_target as read_target


# ============================================================================
# IPC Writer for communicating saved mesh path back to parent process
# ============================================================================

class SimplifierIPCWriter:
    """Handles writing simplified mesh path to IPC file for parent process."""

    IPC_FILENAME = 'bites_simplifier_result.txt'

    @classmethod
    def get_ipc_path(cls) -> str:
        return str(Path(tempfile.gettempdir()) / cls.IPC_FILENAME)

    @classmethod
    def write_result(cls, output_path: str) -> bool:
        """
        Write the simplified mesh path to IPC file.

        Args:
            output_path: Path where the simplified mesh was saved

        Returns:
            True if write succeeded, False otherwise
        """
        ipc_file = cls.get_ipc_path()
        try:
            with open(ipc_file, 'w') as f:
                f.write(output_path)
            logger.info(f"Simplified mesh path saved to IPC file: {ipc_file}")
            return True
        except Exception as e:
            logger.error(f"Error writing simplified mesh path to IPC file: {e}")
            return False


# ============================================================================
# Constants
# ============================================================================

class BenchmarkAlgorithm(Enum):
    """Available benchmark algorithms."""
    CPD = "CPD"
    CHAMFER_DISTANCE = "CD"


# Default configuration values
DEFAULT_MIN_EPSILON = 0.0
DEFAULT_MAX_EPSILON = 20.0
DEFAULT_N_SELECTABLE = 11      # Number of candidate meshes user can select from
DEFAULT_N_PLOT_SAMPLES = 200   # Number of samples for the plot curve (each is a simplified mesh)
DEFAULT_WINDOW_SIZE = (600, 400)
DEFAULT_VISUALIZER_SIZE = (400, 400)
DEFAULT_VISUALIZER_POSITION = (650, 50)
VISUALIZER_UPDATE_INTERVAL_MS = 30
WINDOW_ACTIVATION_DELAY = 0.05  # seconds

# Progress bar phases
# Plotting always takes 10%, the rest is divided between simplification and benchmarking
PROGRESS_PLOTTING = 10
PROGRESS_SIMPLIFICATION_ONLY = 90  # When no benchmark
PROGRESS_SIMPLIFICATION_WITH_BENCH = 45  # When benchmark is enabled
PROGRESS_BENCHMARKING = 45

# File filters
MESH_FILE_FILTER = "Mesh Files (*.ply *.off *.stl *.obj *.glb *.gltf *.3mf)"
PLY_FILE_FILTER = "PLY Files (*.ply);;All Files (*)"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SimplificationConfig:
    """
    Configuration for mesh simplification.

    Attributes:
    -----------
    min_epsilon : float
        Minimum simplification threshold (0 = original mesh)
    max_epsilon : float
        Maximum simplification threshold (higher = more aggressive)
    n_selectable : int
        Number of candidate meshes the user can select from via slider
    n_plot_samples : int
        Total number of samples for the plot curve (each requires mesh simplification)
    """
    min_epsilon: float
    max_epsilon: float
    n_selectable: int
    n_plot_samples: int

    def validate(self) -> Tuple[bool, str]:
        """
        Validate configuration parameters.

        Returns:
        --------
        tuple
            (is_valid, error_message) where error_message is empty if valid
        """
        # Check for NaN or infinity
        if not np.isfinite(self.min_epsilon) or not np.isfinite(self.max_epsilon):
            return False, "Epsilon values must be finite numbers."

        if self.min_epsilon < 0 or self.max_epsilon < 0:
            return False, "Epsilon values cannot be negative."

        if self.min_epsilon >= self.max_epsilon:
            return False, "Min epsilon must be less than max epsilon."

        if not isinstance(self.n_selectable, int) or self.n_selectable < 2:
            return False, "Number of selectable meshes must be an integer >= 2."

        if not isinstance(self.n_plot_samples, int) or self.n_plot_samples < 2:
            return False, "Number of plot samples must be an integer >= 2."

        return True, ""

    def get_effective_n_selectable(self) -> int:
        """Get the effective number of selectable meshes (clamped to n_plot_samples if needed)."""
        return min(self.n_selectable, self.n_plot_samples)


# ============================================================================
# Utility Functions
# ============================================================================

def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from file with proper normal computation.
    
    Parameters:
    -----------
    path : str
        Path to mesh file
    
    Returns:
    --------
    trimesh.Trimesh
        Loaded mesh with computed normals
    """
    mesh = trimesh.load(path, force='mesh')
    
    # Ensure it's a Trimesh object (not Scene)
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, concatenate all geometries
        mesh = trimesh.util.concatenate(
            [geom for geom in mesh.geometry.values() 
             if isinstance(geom, trimesh.Trimesh)]
        )
    
    # Ensure vertex normals are computed (accessing the property triggers computation)
    _ = mesh.vertex_normals

    return mesh


def simplify_mesh(mesh: trimesh.Trimesh,
                 epsilon: float) -> trimesh.Trimesh:
    """
    Simplify a mesh by merging close vertices.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh
    epsilon : float
        Distance threshold for merging vertices

    Returns:
    --------
    trimesh.Trimesh
        Simplified mesh (never returns None values for vertices/faces)
    """
    # Return empty mesh with proper arrays (not None) for edge cases
    def empty_mesh() -> None:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64)
        )

    if epsilon <= 0:
        return mesh.copy()

    # Handle empty or invalid input mesh
    if mesh.vertices is None or mesh.faces is None:
        return empty_mesh()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return empty_mesh()

    from scipy.spatial import cKDTree

    mesh_copy = mesh.copy()
    vertices = mesh_copy.vertices
    faces = mesh_copy.faces

    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(vertices)

    # Find groups of vertices within epsilon distance
    groups = tree.query_ball_tree(tree, epsilon)

    # Create mapping from old vertex indices to new vertex indices
    vertex_map = np.arange(len(vertices))
    visited = np.zeros(len(vertices), dtype=bool)

    for i, group in enumerate(groups):
        if not visited[i]:
            # Mark all vertices in group as visited and map to first vertex
            group_indices = [j for j in group if j >= i]  # Only consider j >= i to avoid duplicates
            for j in group_indices:
                vertex_map[j] = i
                visited[j] = True

    # Get unique vertices and remap faces
    unique_indices = np.unique(vertex_map)
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

    new_vertices = vertices[unique_indices]
    new_faces = np.array([[index_mapping[vertex_map[v]] for v in face] for face in faces])

    # Remove degenerate faces (where any two vertices are the same)
    valid_faces = []
    for face in new_faces:
        if face[0] != face[1] and face[1] != face[2] and face[0] != face[2]:
            valid_faces.append(face)

    if len(valid_faces) == 0:
        # Return empty mesh with proper arrays (not None)
        return empty_mesh()

    new_faces = np.array(valid_faces)

    # Create new mesh with merged vertices
    simplified = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False  # We already cleaned up manually
    )

    return simplified


# ============================================================================
# Worker Threads
# ============================================================================

class SimplificationWorker(QThread):
    """Worker thread for mesh simplification without blocking UI."""

    progress = Signal(int, str)  # (progress_value, status_text)
    # Signal: (candidate_meshes, candidate_epsilons, sample_epsilons, vertex_counts, face_counts)
    done = Signal(list, list, list, list, list)

    def __init__(self, original_mesh: trimesh.Trimesh,
                 sample_epsilons: np.ndarray,
                 candidate_indices: List[int],
                 progress_weight: int):
        """
        Initialize simplification worker.

        Parameters:
        -----------
        original_mesh : trimesh.Trimesh
            The original mesh to simplify
        sample_epsilons : ndarray
            All epsilon values to compute (for plot curve samples)
        candidate_indices : list
            Indices into sample_epsilons that should be kept as selectable candidate meshes
        progress_weight : int
            Portion of progress bar (0-100) this operation should occupy
        """
        super().__init__()
        self.original_mesh = original_mesh
        self.sample_epsilons = sample_epsilons
        self.candidate_indices = set(candidate_indices)
        self.progress_weight = progress_weight

    def run(self) -> None:
        """Execute simplification for all epsilon values."""
        candidate_meshes = []
        candidate_epsilons = []
        vertex_counts = []
        face_counts = []

        n_samples = len(self.sample_epsilons)

        for i, eps in enumerate(self.sample_epsilons):
            try:
                mesh = simplify_mesh(self.original_mesh, eps)

                # Safely get vertex and face counts (handle None or empty arrays)
                try:
                    vert_count = len(mesh.vertices) if mesh.vertices is not None else 0
                    face_count = len(mesh.faces) if mesh.faces is not None else 0
                except (TypeError, AttributeError):
                    vert_count = 0
                    face_count = 0

                vertex_counts.append(vert_count)
                face_counts.append(face_count)

                # Store candidate meshes (the ones user can select from)
                if i in self.candidate_indices:
                    if vert_count > 0 and face_count > 0:
                        candidate_meshes.append(mesh)
                        candidate_epsilons.append(eps)
                    else:
                        logger.info(f"Skipping candidate mesh at eps={eps:.2f}: invalid mesh")

            except Exception as e:
                logger.error(f"Error simplifying at eps={eps:.2f}: {e}")
                vertex_counts.append(0)
                face_counts.append(0)

            percentage = int((i + 1) / n_samples * self.progress_weight)
            status = f"Simplifying mesh {i + 1}/{n_samples}..."
            self.progress.emit(percentage, status)

        self.done.emit(candidate_meshes, candidate_epsilons,
                       list(self.sample_epsilons), vertex_counts, face_counts)


class BenchmarkWorker(QThread):
    """Worker thread for running benchmarks without blocking UI."""

    progress = Signal(int, str)  # (percentage_within_phase, status_text)
    done = Signal(list)  # benchmark_times

    def __init__(self, meshes: List[trimesh.Trimesh],
                 benchmark_func: Callable,
                 progress_weight: int = 100):
        """
        Initialize benchmark worker.

        Parameters:
        -----------
        meshes : list
            Meshes to benchmark
        benchmark_func : callable
            Function to call for each mesh
        progress_weight : int
            Portion of progress bar (0-100) this operation should occupy
        """
        super().__init__()
        self.meshes = meshes
        self.benchmark_func = benchmark_func
        self.progress_weight = progress_weight

    def run(self) -> None:
        """Execute benchmark for all meshes."""
        times = []
        total = len(self.meshes)

        for i, mesh in enumerate(self.meshes):
            start = time.perf_counter()
            self.benchmark_func(mesh)
            end = time.perf_counter()

            times.append(end - start)
            percentage = int((i + 1) / total * self.progress_weight)
            status = f"Benchmarking mesh {i + 1}/{total}..."
            self.progress.emit(percentage, status)

        self.done.emit(times)


# ============================================================================
# Main Dialog
# ============================================================================

class MeshSimplifierDialog(GeometryPersistenceMixin, QDialog):
    """
    Interactive mesh simplification dialog with visualization and benchmarking.

    Features:
    ---------
    - Adjust epsilon thresholds to control simplification
    - Generate multiple simplified versions
    - Visualize vertex/triangle counts
    - Optional CPD benchmarking
    - Interactive mesh preview with matplotlib
    """

    GEOMETRY_KEY = "mesh_simplifier"
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    def __init__(self, mesh_path: str, parent: Optional[QWidget] = None):
        """
        Initialize the mesh simplifier dialog.

        Parameters:
        -----------
        mesh_path : str
            Path to the mesh file to simplify
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Mesh simplifier tool")
        self.setMinimumSize(*DEFAULT_WINDOW_SIZE)

        # Core state
        self.mesh_path = mesh_path
        self.original_mesh = load_mesh(mesh_path)
        self.meshes: List[trimesh.Trimesh] = []
        self.epsilons: List[float] = []
        self.current_index = 0
        self.output_path: Optional[str] = None

        # Benchmark configuration
        self.bench_algo = BenchmarkAlgorithm.CPD.value
        self.benchmark_times: List[float] = []

        # Worker threads
        self.simplification_worker: Optional[SimplificationWorker] = None
        self.benchmark_worker: Optional[BenchmarkWorker] = None

        # Processing state
        self._pending_benchmark = False  # Whether to run benchmark after simplification
        self._progress_base = 0  # Current base progress for phased operations

        # Plot data storage for updating current epsilon highlight
        self._plot_thresholds: Optional[np.ndarray] = None
        self._plot_verts: List[int] = []
        self._plot_tris: List[int] = []
        self._current_epsilon_line = None  # matplotlib line object

        # VisPy 3D viewer (standalone window)
        self.viewer: Optional[VispyMeshViewer] = None

        # UI initialization
        self._init_ui()

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence()
    
    # ------------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------------
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # === TOP SECTION (fixed height) ===
        layout.addLayout(self._create_path_section())
        layout.addLayout(self._create_epsilon_section())
        layout.addLayout(self._create_button_section())

        # Progress and status
        layout.addWidget(self._create_progress_section())
        layout.addWidget(self._create_status_label())

        # Divider before plot
        divider1 = QFrame()
        divider1.setFrameShape(QFrame.HLine)
        divider1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider1)

        # === MIDDLE SECTION (expanding) ===
        # Plot canvas - this is the only thing that should expand
        self._create_plot_canvas()
        layout.addWidget(self.stats_canvas, stretch=1)

        # Divider after plot
        divider2 = QFrame()
        divider2.setFrameShape(QFrame.HLine)
        divider2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider2)

        # === BOTTOM SECTION (fixed height) ===
        layout.addLayout(self._create_slider_section())
        layout.addLayout(self._create_bottom_action_section())

        self.setLayout(layout)

        # Apply screen-aware sizing
        self._apply_screen_aware_sizing()

    def _apply_screen_aware_sizing(self) -> None:
        """Set window size constraints based on available screen space."""
        screen = QApplication.primaryScreen().availableGeometry()

        # Set minimum size so window remains usable
        self.setMinimumSize(550, 500)

        # Set maximum height to 90% of screen height
        self.setMaximumHeight(int(screen.height() * 0.90))

        # Let Qt calculate optimal size based on content
        self.adjustSize()

    def _create_path_section(self) -> QHBoxLayout:
        """Create mesh path input section."""
        layout = QHBoxLayout()

        self.path_input = QLineEdit(self.mesh_path)
        self.path_input.setToolTip(
            'Path to the mesh file to simplify.\n\n'
            'Supported formats: .ply, .off, .stl, .obj, .glb, .gltf, .3mf'
        )
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setToolTip('Select a mesh file to simplify.')
        self.browse_btn.clicked.connect(self._on_browse_mesh)

        layout.addWidget(QLabel("Mesh:"))
        layout.addWidget(self.path_input)
        layout.addWidget(self.browse_btn)

        # Guide button
        guide_button = QToolButton()
        guide_button.setText("Guide")
        guide_button.setToolTip(
            "Open User's Guide\n\n"
            "Opens the PDF documentation in your\n"
            "system's default PDF viewer."
        )
        guide_button.clicked.connect(lambda: gui_utils.open_users_guide(self, page=32))
        layout.addWidget(guide_button)

        return layout
    
    def _create_epsilon_section(self) -> QHBoxLayout:
        """Create epsilon configuration section."""
        layout = QHBoxLayout()

        # Epsilon range inputs
        self.min_eps_spin = self._create_epsilon_spinbox(DEFAULT_MIN_EPSILON)
        self.min_eps_spin.setToolTip(
            'Minimum epsilon (simplification threshold).\n\n'
            'Epsilon = 0 means no simplification (original mesh).\n'
            'The tool will generate meshes at multiple epsilon values\n'
            'between min and max for comparison.'
        )
        self.max_eps_spin = self._create_epsilon_spinbox(DEFAULT_MAX_EPSILON)
        self.max_eps_spin.setToolTip(
            'Maximum epsilon (simplification threshold).\n\n'
            'Higher epsilon = more aggressive simplification = fewer vertices.\n'
            'The optimal value depends on your mesh size and detail requirements.\n\n'
            'TIP: Start with max=20 and adjust based on results.'
        )

        min_label = QLabel("Min epsilon:")
        min_label.setToolTip('Minimum epsilon value (usually 0 for original mesh)')
        #layout.addWidget(min_label)
        #layout.addWidget(self.min_eps_spin)
        #layout.addStretch()
        inner_layout = QVBoxLayout()
        min_layout = QHBoxLayout()
        min_layout.addWidget(min_label)
        min_layout.addWidget(self.min_eps_spin)
        inner_layout.addLayout(min_layout)

        max_label = QLabel("Max epsilon:")
        max_label.setToolTip('Maximum epsilon value (higher = more simplification)')
        #layout.addWidget(max_label)
        #layout.addWidget(self.max_eps_spin)
        #layout.addStretch()
        max_layout = QHBoxLayout()
        max_layout.addWidget(max_label)
        max_layout.addWidget(self.max_eps_spin)
        inner_layout.addLayout(max_layout)
        layout.addLayout(inner_layout)

        # Advanced options
        self._add_advanced_options(layout)

        return layout
    
    def _create_epsilon_spinbox(self, default_value: float) -> QDoubleSpinBox:
        """Create a configured epsilon spinbox."""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(0.0, 100.0)
        spinbox.setValue(default_value)
        spinbox.setSingleStep(1.0)
        spinbox.setStyleSheet(
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 16px; }"
        )
        return spinbox
    
    def _add_advanced_options(self, layout: QHBoxLayout) -> None:
        """Add advanced options to the layout."""
        self.advanced_checkbox = QCheckBox("Advanced options:")
        self.advanced_checkbox.setToolTip(
            'Enable advanced simplification options.\n\n'
            'Most users can leave these at their defaults.'
        )
        self.advanced_checkbox.toggled.connect(self._toggle_advanced_options)
        self.advanced_checkbox.setChecked(False)

        self.n_selectable_spin = QSpinBox()
        self.n_selectable_spin.setRange(2, 50)
        self.n_selectable_spin.setValue(DEFAULT_N_SELECTABLE)
        self.n_selectable_spin.setEnabled(False)
        self.n_selectable_spin.setStyleSheet(
            "QSpinBox::up-button, QSpinBox::down-button { width: 16px; }"
        )
        self.n_selectable_spin.setToolTip(
            'Number of candidate meshes you can select from.\n\n'
            'These are evenly spaced across the epsilon range.\n'
            'More candidates = finer granularity for selection.\n\n'
            f'Default: {DEFAULT_N_SELECTABLE} candidates'
        )

        self.n_plot_samples_spin = QSpinBox()
        self.n_plot_samples_spin.setRange(10, 1000)
        self.n_plot_samples_spin.setValue(DEFAULT_N_PLOT_SAMPLES)
        self.n_plot_samples_spin.setEnabled(False)
        self.n_plot_samples_spin.setStyleSheet(
            "QSpinBox::up-button, QSpinBox::down-button { width: 16px; }"
        )
        self.n_plot_samples_spin.setToolTip(
            'Total number of samples for the plot curve.\n\n'
            'Each sample requires simplifying the mesh at that epsilon.\n'
            'Higher values create smoother curves but take longer.\n\n'
            f'Default: {DEFAULT_N_PLOT_SAMPLES} samples'
        )

        layout.addStretch()
        layout.addWidget(self.advanced_checkbox)
        layout.addStretch()
        n_selectable_label = QLabel("Candidates:")
        n_selectable_label.setToolTip('Number of candidate meshes to select from')
        layout.addWidget(n_selectable_label)
        layout.addWidget(self.n_selectable_spin)
        layout.addStretch()
        n_samples_label = QLabel("Plot samples:")
        n_samples_label.setToolTip('Number of samples for the plot curve')
        layout.addWidget(n_samples_label)
        layout.addWidget(self.n_plot_samples_spin)
        layout.addStretch()
    
    def _create_button_section(self) -> QHBoxLayout:
        """Create action button section."""
        layout = QHBoxLayout()
        
        self.simplify_btn = QPushButton("Simplify")
        self.simplify_btn.setToolTip(
            'Generate simplified meshes at different epsilon values.\n\n'
            'This creates multiple versions of your mesh with varying\n'
            'levels of simplification. Use the slider below to browse\n'
            'through them and find a good balance between detail and size.\n\n'
            'The plot shows vertex/face count vs. epsilon value.'
        )
        self.simplify_btn.clicked.connect(lambda: self._on_simplify(benchmark=False))

        benchmark_text = f"Simplify + Benchmark {self.bench_algo} (slow)"
        self.benchmark_btn = QPushButton(benchmark_text)
        self.benchmark_btn.setToolTip(
            'Generate simplified meshes AND run CPD alignment benchmarks.\n\n'
            'This is SLOW but useful for finding the optimal simplification:\n'
            '• Generates meshes at different epsilon values\n'
            '• Runs CPD alignment on each mesh\n'
            '• Plots alignment time vs. simplification level\n\n'
            'Use this to find the "sweet spot" where alignment is fast\n'
            'enough but the mesh retains sufficient detail.\n\n'
            'NOTE: always take the CPD-benchmark results with a grain of salt;\n'
            'depending on the mesh, overhead and noise may dominate timings.'
        )
        self.benchmark_btn.clicked.connect(lambda: self._on_simplify(benchmark=True))
        
        layout.addStretch()
        layout.addWidget(self.simplify_btn)
        layout.addWidget(self.benchmark_btn)
        
        return layout
    
    def _create_progress_section(self) -> QProgressBar:
        """Create progress bar."""
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        return self.progress_bar
    
    def _create_status_label(self) -> QLabel:
        """Create status label."""
        self.status_label = QLabel("")
        return self.status_label
    
    def _create_plot_canvas(self) -> FigureCanvas:
        """Create matplotlib canvas for statistics plotting."""
        self.stats_fig = Figure(figsize=(5, 3))
        self.stats_canvas = FigureCanvas(self.stats_fig)
        return self.stats_canvas

    def _create_slider_section(self) -> QVBoxLayout:
        """Create mesh selection slider section."""
        layout = QVBoxLayout()

        eps_label = QLabel("Mesh selector:")
        eps_label.setToolTip(
            'Browse through the generated simplified meshes.\n\n'
            'Move the slider to select different simplification levels\n'
            'and preview them in the 3D viewer.'
        )
        layout.addWidget(eps_label)

        self.slider_label = QLabel("Click 'Simplify' to generate simplified meshes")
        self.slider_label.setToolTip(
            'Shows information about the currently selected mesh:\n'
            '• Epsilon value (simplification strength)\n'
            '• Vertex count\n'
            '• Face count'
        )
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(9)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.setToolTip(
            'Select a simplified mesh to preview.\n\n'
            'LEFT = Less simplification (more detail, more vertices)\n'
            'RIGHT = More simplification (less detail, fewer vertices)\n\n'
            'Use the 3D preview to visually assess quality at each level.'
        )
        self.slider.valueChanged.connect(self._on_slider_changed)

        layout.addWidget(self.slider_label)
        layout.addWidget(self.slider)

        return layout

    def _create_bottom_action_section(self) -> QHBoxLayout:
        """Create bottom action buttons (3D preview and save)."""
        layout = QHBoxLayout()

        self.view_3d_btn = QPushButton("Open 3D mesh preview")
        self.view_3d_btn.clicked.connect(self._on_open_3d_viewer)
        self.view_3d_btn.setEnabled(False)
        self.view_3d_btn.setToolTip(
            'Opens a 3D interactive viewer for the currently selected mesh.\n\n'
            'KEYBOARD CONTROLS:\n'
            '• X/Y/Z: Snap to axis views\n'
            '• A: Toggle axis indicator\n'
            '• W: Toggle wireframe\n'
            '• B: Toggle background color\n'
            '• F: Fit mesh to view\n'
            '• H: Print help\n'
            '• Esc: Close viewer\n\n'
            'MOUSE CONTROLS:\n'
            '• Drag: Rotate view\n'
            '• Scroll: Zoom'
        )

        self.save_btn = QPushButton("Save selected simplified mesh")
        self.save_btn.clicked.connect(self._on_save_mesh)
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip(
            'Save the currently selected simplified mesh to a file.\n\n'
            'The mesh will be saved in PLY format. You can then use this\n'
            'simplified mesh as the target shape for optimization.\n\n'
            'After saving, the path will be passed to the Config Editor.'
        )

        layout.addStretch()
        layout.addWidget(self.view_3d_btn)
        layout.addWidget(self.save_btn)
        layout.addStretch()

        return layout
    
    # ------------------------------------------------------------------------
    # Mesh Visualization
    # ------------------------------------------------------------------------

    def _update_mesh_selection(self, index: int) -> None:
        """
        Update current mesh selection and display info.

        Parameters:
        -----------
        index : int
            Mesh index to select
        """
        # Bounds checking
        if not self.meshes:
            self.slider_label.setText("No meshes available")
            return

        # Clamp index to valid range
        index = max(0, min(index, len(self.meshes) - 1))
        self.current_index = index

        # Safely get epsilon value
        eps_val = self.epsilons[index] if index < len(self.epsilons) else 0.0

        # Safely get mesh stats
        mesh = self.meshes[index]
        try:
            vertex_count = len(mesh.vertices) if mesh.vertices is not None else 0
            face_count = len(mesh.faces) if mesh.faces is not None else 0
        except (TypeError, AttributeError):
            vertex_count = 0
            face_count = 0

        # Format epsilon display
        eps_display = f"{eps_val:.2f}" if isinstance(eps_val, (int, float)) else "?"

        # Build label text
        label_text = (
            f"Mesh {index + 1} (ε={eps_display}): "
            f"{vertex_count} vertices, {face_count} faces"
        )

        # Add benchmark time if available
        if self.benchmark_times and index < len(self.benchmark_times):
            bench_time = self.benchmark_times[index]
            label_text += f", {self.bench_algo} {bench_time:.2f}s"

        self.slider_label.setText(label_text)

        # Update the plot to highlight current epsilon
        self._update_plot_highlight()

        # Update viewer if it's open
        if self.viewer is not None and hasattr(self.viewer, 'native'):
            try:
                self._update_viewer_mesh()
            except Exception as e:
                logger.warning(f"Error updating viewer: {e}")

    def _on_open_3d_viewer(self) -> None:
        """Open or update the 3D VisPy viewer window."""
        if not VISPY_AVAILABLE:
            QMessageBox.information(
                self,
                "Viewer Not Available",
                "VisPy mesh viewer is not available.\n"
                "Please check the vispy_mesh_viewer module."
            )
            return

        if not self.meshes:
            return

        # Always create a fresh viewer instance
        # CRITICAL: Reusing a closed viewer causes responsiveness issues in multiprocess mode.
        # When a VisPy viewer is closed, its internal state becomes corrupted and cannot be
        # reliably reused. Creating a new instance each time ensures clean state.
        if self.viewer is not None:
            try:
                self.viewer.close()
            except RuntimeError as e:
                logger.debug(f"Viewer already closed: {e}")
            self.viewer = None

        try:
            self.viewer = VispyMeshViewer(
                parent_handler=self,
                title="Mesh Simplifier - 3D Preview",
                size=(800, 800),
                show_axis=True,
                bgcolor=DEFAULT_BG_COLOR
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Viewer Error",
                f"Failed to create 3D viewer:\n{str(e)}"
            )
            return

        # Update the mesh in the viewer
        self._update_viewer_mesh()

        # Show the viewer window
        self.viewer.show_and_activate()

    def _update_viewer_mesh(self) -> None:
        """Update the mesh displayed in the VisPy viewer."""
        if self.viewer is None or not self.meshes:
            return

        # Bounds checking
        if self.current_index < 0 or self.current_index >= len(self.meshes):
            logger.warning(f"current_index {self.current_index} out of range [0, {len(self.meshes)})")
            return

        mesh = self.meshes[self.current_index]

        # Validate mesh
        if mesh is None or mesh.vertices is None or mesh.faces is None:
            logger.warning("Invalid mesh for viewer")
            return

        # Safely get epsilon value
        eps_val = self.epsilons[self.current_index] if self.current_index < len(self.epsilons) else 0.0

        try:
            # Clear existing visuals
            self.viewer.clear_all_visuals()

            # Add the current mesh using trimesh integration
            self.viewer.load_mesh_from_trimesh(
                mesh,
                color='lightblue',
                shading='flat',
                name=f"Mesh ε={eps_val:.2f}",
            )

            # Add wireframe (toggleable with 'W' key)
            self.viewer.add_wireframe_from_trimesh(
                mesh,
                color=DEFAULT_WIREFRAME_COLOR,
                #line_width=1.0,
                line_width=DEFAULT_LINE_WIDTH,
                is_target=False,  # Source wireframe - toggleable with 'W'
                name=f"Wireframe ε={eps_val:.2f}",
            )

            # Fit the camera to the mesh
            self.viewer.fit_to_meshes()

        except Exception as e:
            logger.warning(f"Error updating viewer mesh: {e}")

    def handle_key_press(self, key) -> None:
        """
        Handle key press events from the VisPy viewer.

        Parameters:
        -----------
        key : str
            The key name that was pressed
        """
        # Forward viewer-specific keys to the viewer's internal handler
        if self.viewer is not None:
            self.viewer._handle_viewer_key(key)
    
    # ------------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------------
    
    def _toggle_advanced_options(self, checked: bool) -> None:
        """Toggle advanced options visibility."""
        self.n_selectable_spin.setEnabled(checked)
        self.n_plot_samples_spin.setEnabled(checked)
    
    def _on_browse_mesh(self) -> None:
        """Handle browse mesh button click."""
        mesh_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mesh File",
            "",
            MESH_FILE_FILTER, "",
            gui_utils.get_file_dialog_options()
        )

        if mesh_path:
            try:
                mesh_path = str(Path(mesh_path).relative_to(Path.cwd()))
            except ValueError:
                pass  # Keep absolute path if relative path not possible
            self.mesh_path = mesh_path
            self.path_input.setText(mesh_path)

            try:
                self.original_mesh = load_mesh(mesh_path)
                self.status_label.setText("Loaded new mesh.")
                self.meshes = []
                self.stats_canvas.figure.clf()
                self.stats_canvas.draw()

                # Clear viewer if it exists
                if self.viewer is not None:
                    self.viewer.clear_all_visuals()

                # Disable buttons until new simplification
                self.slider.setEnabled(False)
                self.save_btn.setEnabled(False)
                self.view_3d_btn.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to load mesh: {str(e)}"
                )
    
    def _on_simplify(self, benchmark: bool = False) -> None:
        """
        Handle simplify button click. Starts async simplification.

        Parameters:
        -----------
        benchmark : bool
            Whether to run benchmarking after simplification
        """
        # Validate original mesh exists and is valid
        if self.original_mesh is None:
            QMessageBox.warning(
                self,
                "No Mesh Loaded",
                "Please load a mesh file first."
            )
            return

        if (self.original_mesh.vertices is None or
            self.original_mesh.faces is None or
            len(self.original_mesh.vertices) == 0 or
            len(self.original_mesh.faces) == 0):
            QMessageBox.warning(
                self,
                "Invalid Mesh",
                "The loaded mesh is empty or invalid."
            )
            return

        config = self._get_simplification_config()

        is_valid, error_msg = config.validate()
        if not is_valid:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                error_msg or "Please check your epsilon and mesh count settings."
            )
            return

        # Store benchmark flag for later
        self._pending_benchmark = benchmark

        # Disable UI during processing
        self._set_processing_ui_state(enabled=False)

        # Calculate epsilon values and start async simplification
        sample_epsilons, candidate_indices = self._compute_epsilon_values(config)

        # Final safety check on computed values
        if len(sample_epsilons) == 0:
            QMessageBox.warning(
                self,
                "Configuration Error",
                "Could not generate epsilon values. Please check settings."
            )
            self._set_processing_ui_state(enabled=True)
            return

        if len(candidate_indices) == 0:
            QMessageBox.warning(
                self,
                "Configuration Error",
                "Could not determine candidate mesh indices. Please check settings."
            )
            self._set_processing_ui_state(enabled=True)
            return

        # Determine progress weight for simplification phase
        if benchmark:
            simplification_weight = PROGRESS_SIMPLIFICATION_WITH_BENCH
        else:
            simplification_weight = PROGRESS_SIMPLIFICATION_ONLY

        # Reset state
        self._progress_base = 0
        self.progress_bar.setValue(0)
        self.benchmark_times = []

        # Start simplification worker
        self.simplification_worker = SimplificationWorker(
            self.original_mesh,
            sample_epsilons,
            candidate_indices,
            simplification_weight
        )
        self.simplification_worker.progress.connect(self._on_simplification_progress)
        self.simplification_worker.done.connect(self._on_simplification_complete)
        self.simplification_worker.start()

    def _get_simplification_config(self) -> SimplificationConfig:
        """Get current simplification configuration from UI."""
        return SimplificationConfig(
            min_epsilon=self.min_eps_spin.value(),
            max_epsilon=self.max_eps_spin.value(),
            n_selectable=self.n_selectable_spin.value(),
            n_plot_samples=self.n_plot_samples_spin.value()
        )

    def _compute_epsilon_values(self, config: SimplificationConfig) -> Tuple[np.ndarray, List[int]]:
        """
        Compute all epsilon values needed for plot samples and candidate selection.

        Parameters:
        -----------
        config : SimplificationConfig
            Simplification parameters

        Returns:
        --------
        tuple
            (sample_epsilons, candidate_indices) where:
            - sample_epsilons: ndarray of all epsilon values for the plot curve
            - candidate_indices: list of indices into sample_epsilons for selectable candidates
        """
        # Defensive: ensure we have valid n_plot_samples
        n_plot_samples = max(2, config.n_plot_samples)

        # Generate epsilon values for all plot samples
        sample_epsilons = np.linspace(
            config.min_epsilon,
            config.max_epsilon,
            n_plot_samples
        )

        # Get effective number of selectable candidates (clamped to n_plot_samples)
        n_selectable = config.get_effective_n_selectable()

        # Defensive: ensure n_selectable is valid
        n_selectable = max(1, min(n_selectable, n_plot_samples))

        # Calculate which sample indices correspond to selectable candidates
        if n_selectable >= n_plot_samples:
            # All samples are selectable candidates
            candidate_indices = list(range(n_plot_samples))
        elif n_selectable == 1:
            # Only one candidate: select the middle sample
            candidate_indices = [n_plot_samples // 2]
        else:
            # Select evenly spaced indices from the samples
            # Defensive: ensure divisor is never zero
            divisor = max(1, n_selectable - 1)
            candidate_indices = [
                min(n_plot_samples - 1, int(round(i * (n_plot_samples - 1) / divisor)))
                for i in range(n_selectable)
            ]
            # Ensure uniqueness, proper order, and valid range
            candidate_indices = sorted(set(
                idx for idx in candidate_indices
                if 0 <= idx < n_plot_samples
            ))

        # Final safety: ensure we have at least one candidate index
        if not candidate_indices:
            candidate_indices = [0] if n_plot_samples > 0 else []

        return sample_epsilons, candidate_indices

    def _set_processing_ui_state(self, enabled: bool) -> None:
        """
        Enable or disable UI elements during processing.

        Parameters:
        -----------
        enabled : bool
            True to enable UI, False to disable
        """
        self.simplify_btn.setEnabled(enabled)
        self.benchmark_btn.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        self.slider.setEnabled(enabled and len(self.meshes) > 0)
        self.save_btn.setEnabled(enabled and len(self.meshes) > 0)
        self.view_3d_btn.setEnabled(enabled and len(self.meshes) > 0)

    def _on_simplification_progress(self, value: int, text: str) -> None:
        """
        Handle simplification progress updates from worker thread.

        Parameters:
        -----------
        value : int
            Progress value (0 to progress_weight)
        text : str
            Status text
        """
        self.progress_bar.setValue(self._progress_base + value)
        self.status_label.setText(text)

    def _on_simplification_complete(self,
                                    candidate_meshes: List[trimesh.Trimesh],
                                    candidate_epsilons: List[float],
                                    sample_epsilons: List[float],
                                    vertex_counts: List[int],
                                    face_counts: List[int]) -> None:
        """
        Handle simplification completion.

        Parameters:
        -----------
        candidate_meshes : list
            Candidate meshes the user can select from
        candidate_epsilons : list
            Epsilon values for the candidate meshes
        sample_epsilons : list
            All epsilon values used for plot samples
        vertex_counts : list
            Vertex counts for each plot sample
        face_counts : list
            Face counts for each plot sample
        """
        self.meshes = candidate_meshes
        self.epsilons = candidate_epsilons

        # Store plot data
        self._plot_thresholds = np.array(sample_epsilons)
        self._plot_verts = vertex_counts
        self._plot_tris = face_counts

        if not self.meshes:
            self.status_label.setText("No valid meshes were generated.")
            self._set_processing_ui_state(enabled=True)
            self.progress_bar.setValue(100)
            return

        # Check if benchmark is pending
        if self._pending_benchmark:
            self._start_benchmark()
        else:
            self._finalize_simplification()

    def _start_benchmark(self) -> None:
        """Start benchmarking in background thread."""
        self.status_label.setText("Starting benchmark...")

        # Update progress base to after simplification phase
        self._progress_base = PROGRESS_SIMPLIFICATION_WITH_BENCH

        self.benchmark_worker = BenchmarkWorker(
            self.meshes,
            self._benchmark_function,
            PROGRESS_BENCHMARKING
        )
        self.benchmark_worker.progress.connect(self._on_benchmark_progress)
        self.benchmark_worker.done.connect(self._on_benchmark_complete)
        self.benchmark_worker.start()

    def _finalize_simplification(self) -> None:
        """Finalize simplification by rendering plot and updating UI."""
        self.status_label.setText("Rendering plot...")

        # Render plot (uses pre-computed data, no re-simplification)
        self._render_plot(self._plot_thresholds, self._plot_verts, self._plot_tris)

        # Update progress to 100%
        self.progress_bar.setValue(100)

        # Update UI
        self.slider.setMaximum(len(self.meshes) - 1)
        self.slider.setValue(0)
        self._set_processing_ui_state(enabled=True)
        self._update_mesh_selection(0)

        self.status_label.setText("Simplification complete")

        # Automatically open the 3D preview
        self._on_open_3d_viewer()

    def _on_slider_changed(self, index: int) -> None:
        """Handle slider value change."""
        self._update_mesh_selection(index)
    
    def _on_save_mesh(self) -> None:
        """Handle save mesh button click."""
        base_name = Path(self.mesh_path).stem
        default_path = f"{base_name}_simplified.ply"

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simplified Mesh",
            default_path,
            PLY_FILE_FILTER, "",
            gui_utils.get_file_dialog_options()
        )

        if not output_path:
            return

        try:
            output_path = str(Path(output_path).relative_to(Path.cwd()))
        except ValueError:
            pass  # Keep absolute path if relative path not possible

        if Path(output_path).exists():
            reply = QMessageBox.question(
                self,
                "File Exists",
                f"The file '{output_path}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        try:
            self.meshes[self.current_index].export(output_path)
            QMessageBox.information(
                self,
                "Saved",
                f"Simplified mesh saved to: {output_path}"
            )
            self.output_path = output_path
            # Write the saved path to IPC file for parent process to read
            SimplifierIPCWriter.write_result(output_path)
            self.accept()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save mesh: {str(e)}"
            )
    
    # ------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------

    def _render_plot(self, thresholds: np.ndarray,
                    verts: List[int], tris: List[int]) -> None:
        """
        Render the statistics plot using pre-computed data.

        Parameters:
        -----------
        thresholds : ndarray
            Epsilon threshold values
        verts : list
            Vertex counts
        tris : list
            Triangle counts
        """
        self.stats_fig.clf()
        ax1 = self.stats_fig.add_subplot(111)

        # Validate inputs
        if thresholds is None or len(thresholds) == 0:
            ax1.text(0.5, 0.5, "No data to plot", ha='center', va='center',
                     transform=ax1.transAxes)
            self.stats_canvas.draw()
            return

        # Ensure array lengths match (use minimum length for safety)
        min_len = min(len(thresholds), len(verts) if verts else 0, len(tris) if tris else 0)
        if min_len == 0:
            ax1.text(0.5, 0.5, "No data to plot", ha='center', va='center',
                     transform=ax1.transAxes)
            self.stats_canvas.draw()
            return

        # Truncate to matching length if needed
        plot_thresholds = thresholds[:min_len]
        plot_verts = verts[:min_len]
        plot_tris = tris[:min_len]

        # Plot vertex and triangle counts
        ax1.plot(plot_thresholds, plot_verts, label="Vertices", linewidth=2)
        ax1.plot(plot_thresholds, plot_tris, label="Faces", linewidth=2)

        # Mark all selectable epsilon values (light dotted lines)
        for eps in self.epsilons:
            if isinstance(eps, (int, float)) and np.isfinite(eps):
                ax1.axvline(x=eps, color='g', linestyle=':', alpha=0.3)

        # Highlight current epsilon with a prominent line
        if self.epsilons and 0 <= self.current_index < len(self.epsilons):
            current_eps = self.epsilons[self.current_index]
            if isinstance(current_eps, (int, float)) and np.isfinite(current_eps):
                self._current_epsilon_line = ax1.axvline(
                    x=current_eps, color='green', linestyle='-', linewidth=1.5,
                    alpha=0.8,
                )

        ax1.set_title("Mesh density vs epsilon")
        ax1.set_xlabel("Epsilon (simplification threshold)")
        ax1.set_ylabel("Count")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Add benchmark times if available
        self._add_benchmark_to_plot(ax1)

        self.stats_fig.tight_layout()
        self.stats_canvas.draw()

    def _update_plot_highlight(self) -> None:
        """Update the highlighted epsilon line on the plot without full re-render."""
        if self._plot_thresholds is None or not self.epsilons:
            return

        # Re-render the plot with updated current index
        self._render_plot(self._plot_thresholds, self._plot_verts, self._plot_tris)

    def _add_benchmark_to_plot(self, ax1) -> None:
        """
        Add benchmark times to existing plot.
        
        Parameters:
        -----------
        ax1 : matplotlib.axes.Axes
            Primary plot axis
        """
        if not hasattr(self, 'benchmark_times'):
            return
        
        if len(self.benchmark_times) != len(self.epsilons):
            return
        
        ax2 = ax1.twinx()
        ax2.plot(
            self.epsilons,
            self.benchmark_times,
            'rx',
            markersize=10,
            label=f"{self.bench_algo} duration"
        )
        ax2.set_ylabel(f"{self.bench_algo} alignment time (s)", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    # ------------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------------
    
    def _benchmark_function(self, mesh: trimesh.Trimesh) -> None:
        """
        Run benchmark algorithm on a mesh.
        
        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Mesh to benchmark
        """
        try:
            target_path = get_asset('benchmark_target.ply')
            target_mesh = load_mesh(str(target_path))
            
            if self.bench_algo == BenchmarkAlgorithm.CPD.value:
                
                import bites.alignment.coherent_point_drift as cpd
                cpd.perform_CPD_on_mesh(source=mesh, target=target_mesh)
                
                logger.info(f"Benchmark {self.bench_algo} on mesh with {len(mesh.vertices)} vertices")
            else:
                logger.warning(f"Benchmark algorithm {self.bench_algo} not implemented")
                
        except Exception as e:
            logger.error(f"Error during {self.bench_algo} benchmark: {e}")
    
    def _on_benchmark_progress(self, value: int, text: str) -> None:
        """
        Handle benchmark progress updates.

        Parameters:
        -----------
        value : int
            Progress value (0 to progress_weight)
        text : str
            Status text
        """
        self.progress_bar.setValue(self._progress_base + value)
        self.status_label.setText(text)

    def _on_benchmark_complete(self, times: List[float]) -> None:
        """
        Handle benchmark completion.

        Parameters:
        -----------
        times : list
            Benchmark times for each mesh
        """
        self.benchmark_times = times

        # Now finalize (renders plot with benchmark data)
        self._finalize_simplification()
    
    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Handle dialog close event."""
        # Clean up worker threads
        for worker in [self.simplification_worker, self.benchmark_worker]:
            if worker is not None and worker.isRunning():
                worker.quit()
                if not worker.wait(1000):
                    # Worker didn't finish in time - detach so it can finish safely
                    worker.finished.connect(worker.deleteLater)
                    worker.setParent(None)

        # Clean up VisPy viewer if it exists
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as e:
                logger.debug(f"Error closing viewer: {e}")
            self.viewer = None

        # GeometryPersistenceMixin.closeEvent saves geometry automatically
        super().closeEvent(event)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    mesh_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Mesh File",
        "",
        "Mesh Files (*.ply *.off *.stl *.obj *.glb *.gltf *.3mf)"
    )
    
    if mesh_path:
        dialog = MeshSimplifierDialog(mesh_path)
        dialog.show()
        sys.exit(app.exec())