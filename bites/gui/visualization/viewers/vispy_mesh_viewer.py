#!/usr/bin/env python

"""
Enhanced VisPy mesh viewer with keyboard controls and flexible visual management.
Designed for integration with PySide6 applications and standalone use.

This module provides a 3D mesh viewer with support for:
- Multiple mesh and wireframe rendering
- Interactive camera controls
- Keyboard shortcuts for view manipulation
- Integration with trimesh for mesh loading
"""


from __future__ import annotations

# Force Qt backend for VisPy (must be set before importing vispy)
import os
if 'VISPY_BACKEND' not in os.environ:
    os.environ['VISPY_BACKEND'] = 'pyside6'

import logging
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from vispy import scene
from vispy.scene import visuals
import trimesh
import bites.mesh.trimesh_utils as trimesh_utils
from bites.paths import get_asset

_logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

class VisualType(Enum):
    """Types of visual objects that can be rendered."""
    MESH = "mesh"
    SOURCE_WIREFRAME = "source_wireframe"
    TARGET_WIREFRAME = "target_wireframe"


class CameraPreset(Enum):
    """Predefined camera view angles with corresponding light directions."""
    # (name, elevation, azimuth, light_dir)
    # light_dir points FROM the light source toward the scene
    X_AXIS = ("x", 0, 90, (1, 0, 0))
    Y_AXIS = ("y", 0, 0, (0, 1, 0))
    Z_AXIS = ("z", 90, 0, (0, 0, 1))

    def __init__(self, name: str, elevation: float, azimuth: float, light_dir: Tuple[float, float, float]):
        self.view_name = name
        self.elevation = elevation
        self.azimuth = azimuth
        self.light_dir = light_dir


# Default viewer settings
DEFAULT_CAMERA_DISTANCE = 200
DEFAULT_CAMERA_TYPE = 'turntable'
DEFAULT_BG_COLOR = 'black'
DEFAULT_BG_COLOR_ALT = 'white'  # Alternative background color (toggled with 'B')
DEFAULT_BG_COLOR_GREY = '#2E3440'  # Nord theme dark background
DEFAULT_WINDOW_SIZE = (800, 800)
DEFAULT_WINDOW_POSITION = (40, 20)
#DEFAULT_LINE_WIDTH = 1.5
DEFAULT_LINE_WIDTH = 2.0
DEFAULT_WIREFRAME_VISIBLE = False

# Mesh color options - edit DEFAULT_MESH_COLOR to use any of these alternatives
MESH_COLOR_LIGHTBLUE = 'lightblue'
MESH_COLOR_CORAL = '#FF7F50'       # Warm coral - good contrast on dark bg
MESH_COLOR_MINT = '#98FF98'        # Pale mint green
MESH_COLOR_LAVENDER = '#E6E6FA'    # Soft lavender
MESH_COLOR_PEACH = '#FFDAB9'       # Peach puff - warm neutral
MESH_COLOR_GOLD = '#FFD700'        # Gold - high visibility
MESH_COLOR_SALMON = '#FA8072'      # Salmon pink
MESH_COLOR_SKYBLUE = '#87CEEB'     # Sky blue - lighter than lightblue
MESH_COLOR_IVORY = '#FFFFF0'       # Ivory - works well on dark bg
MESH_COLOR_SILVER = '#C0C0C0'      # Neutral silver/gray

DEFAULT_MESH_COLOR = MESH_COLOR_LIGHTBLUE
#DEFAULT_WIREFRAME_COLOR = (1, 1, 1, 1) # White wireframe
#DEFAULT_WIREFRAME_COLOR = (0, 0, 0, 1) # Black wireframe
DEFAULT_WIREFRAME_COLOR = '#2E3440' # Nord theme dark color
DEFAULT_MARGIN_FACTOR = 1.1  # Margin factor when fitting to view
DEFAULT_FOV = 45.0  # Default field of view for perspective camera, degrees
# Shading mode: controls normal interpolation (S key to cycle)
# 'smooth' = interpolated normals (smoother look), 'flat' = face normals (faceted look)
DEFAULT_MESH_SHADING = 'flat'  # 'smooth' or 'flat'

# Diffuse presets: controls directional light intensity (L key to cycle)
# Higher diffuse = brighter + more contrast, lower = darker + flatter
DIFFUSE_PRESETS = [
    ('bright', 1.0),    # Full directional lighting - brightest
    ('medium', 0.7),    # Medium brightness
    ('soft', 0.5),      # Softer lighting
    ('dim', 0.25),      # Dim, flatter appearance
]
DEFAULT_DIFFUSE_INDEX = 0  # Start at 'bright' for light appearance

# Specular settings: controls the "luster" highlight effect (like Open3D)
DEFAULT_SPECULAR_ENABLED = True
DEFAULT_SPECULAR_COEFFICIENT = (0.3, 0.3, 0.3, 1.0)  # RGB + alpha
#DEFAULT_SHININESS = 32  # Higher = tighter/smaller highlights
DEFAULT_SHININESS = 0.001  # Higher = tighter/smaller highlights

# Ambient coefficient: base brightness level (higher = brighter overall)
# VisPy default is ~0.3, Open3D uses higher values for brighter appearance
DEFAULT_AMBIENT_COEFFICIENT = 1.0  # Adjust this for overall brightness (0-1 range)

# Light direction: points FROM the light source toward the scene
# Default matches Y-axis view (light comes from +Y direction)
DEFAULT_LIGHT_DIR = (0, 1, 0)

# Keyboard controls help text
HELP_TEXT = """
=============================================
VIEWER KEYBOARD CONTROLS
=============================================
X       : View along X-axis (light follows)
Y       : View along Y-axis (light follows)
Z       : View along Z-axis (light follows)
F       : Fit objects in view
W       : Toggle source wireframe on/off
T       : Toggle target wireframe on/off
S       : Cycle shading (smooth/flat/uniform)
D       : Cycle diffuse intensity (shadow strength)
G       : Toggle specular highlights (luster effect)
B       : Toggle background (black/white)
C       : Toggle backface culling on/off
A       : Toggle coordinate axis on/off
H       : Show this help message
ESC     : Close viewer

MOUSE CONTROLS
=============================================
Left button + drag   : Rotate
Right button + drag  : Zoom
Shift + Left button + drag : Pan
=============================================
"""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MeshVisual:
    """Container for a mesh visual with metadata."""
    visual: visuals.Visual
    visual_type: VisualType
    name: Optional[str] = None
    visible: bool = True
    
    def set_visibility(self, visible: bool) -> None:
        """Set the visibility of this visual."""
        self.visible = visible
        self.visual.visible = visible
    
    def is_mesh(self) -> bool:
        """Check if this is a mesh visual."""
        return self.visual_type == VisualType.MESH
    
    def is_source_wireframe(self) -> bool:
        """Check if this is a source wireframe."""
        return self.visual_type == VisualType.SOURCE_WIREFRAME
    
    def is_target_wireframe(self) -> bool:
        """Check if this is a target wireframe."""
        return self.visual_type == VisualType.TARGET_WIREFRAME



# ============================================================================
# Main Viewer Class
# ============================================================================

class VispyMeshViewer(scene.SceneCanvas):
    """
    Interactive 3D mesh viewer using VisPy with keyboard controls.
    
    Features:
    ---------
    - Multiple mesh and wireframe support
    - Separate control of source/target wireframes
    - Smooth/flat shading toggle
    - Camera presets (X/Y/Z axis views)
    - Backface culling control
    - Coordinate axis toggle
    
    See HELP_TEXT constant for keyboard and mouse controls.
    """
    
    def __init__(
        self,
        parent_handler: Optional[object] = None,
        bgcolor: Union[str, Tuple] = DEFAULT_BG_COLOR,
        camera: str = DEFAULT_CAMERA_TYPE,
        show_axis: bool = False,
        title: str = '3D Mesh Viewer',
        position: Tuple[int, int] = DEFAULT_WINDOW_POSITION,
        size: Tuple[int, int] = DEFAULT_WINDOW_SIZE
    ):
        """
        Initialize the mesh viewer.
        
        Parameters:
        -----------
        parent_handler : object, optional
            Parent object that handles key events
        bgcolor : str or tuple
            Background color
        camera : str
            Camera type ('turntable', 'arcball', 'fly', etc.)
        show_axis : bool
            Whether to show XYZ axis initially
        title : str
            Window title
        position : tuple
            Window position (x, y)
        size : tuple
            Window size (width, height)
        """

        # Use default VisPy backend detection
        # In complex Qt applications, VisPy works best when launched in subprocess
        # This viewer will work in-process for simple cases (like shape browser subprocess)
        # but may have event routing issues in complex nested dialog scenarios
        super().__init__(
            keys='interactive',
            show=False,  # Don't show immediately
            bgcolor=bgcolor,
            title=title,
            size=size
            # Let VisPy auto-detect the application
        )

        self.unfreeze()

        # Configuration
        self.parent_handler = parent_handler
        self.position = position

        # Lifecycle state
        self._is_closed = False
        self._close_callbacks: List[callable] = []
        self._error_callbacks: List[callable] = []

        # Initialize view and camera
        self.view = self.central_widget.add_view()
        self.view.camera = camera

        # Create coordinate axis
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
        self.axis_visible = show_axis

        # Storage for visuals
        self.mesh_visuals: List[MeshVisual] = []

        # State flags
        self.source_wireframe_visible = DEFAULT_WIREFRAME_VISIBLE
        self.target_wireframe_visible = True
        self.show_backface = True
        self.bg_is_black = True  # Track current background state

        # Lighting state (independent controls)
        self.shading_mode = DEFAULT_MESH_SHADING  # 'smooth', 'flat', or None
        self.diffuse_index = DEFAULT_DIFFUSE_INDEX  # Index into DIFFUSE_PRESETS
        self.specular_enabled = DEFAULT_SPECULAR_ENABLED  # Specular highlights on/off
        self.light_dir = DEFAULT_LIGHT_DIR  # Light direction (synced with camera view)

        # Set initial camera settings
        self._configure_camera(distance=DEFAULT_CAMERA_DISTANCE, axis_visible=show_axis)

        # Connect keyboard event
        self.events.key_press.connect(self.on_key_press)

        self.freeze()

        # Ensure it's a completely independent top-level window
        # CRITICAL: This makes the VisPy window independent and responsive
        # Without these settings, VisPy windows launched from nested dialogs
        # (like ConfigWindow) can become unresponsive to mouse/keyboard events
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QIcon

        try:
            # CRITICAL: Completely orphan the window from any parent widget hierarchy
            # This prevents event filtering/blocking from parent dialogs
            self.native.setParent(None)

            # Set window flag to make it a proper top-level window
            self.native.setWindowFlag(Qt.WindowType.Window, True)

            # Ensure it's not a child widget that would be affected by parent's event loop
            self.native.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

            # Ensure mouse events are properly tracked and not filtered
            self.native.setMouseTracking(True)
            self.native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.native.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

            # Set window icon
            icon_path = get_asset('bites_icon.png')
            if icon_path.exists():
                self.native.setWindowIcon(QIcon(str(icon_path)))
        except Exception as e:
            _logger.warning(f"Could not set VisPy window flags: {e}")

    # ------------------------------------------------------------------------
    # Lifecycle Management
    # ------------------------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        """
        Check if the viewer is still usable.

        Returns False if the viewer has been closed or the underlying
        Qt widget has been destroyed.
        """
        if self._is_closed:
            return False
        try:
            # Check if native Qt widget still exists and is valid
            if self.native is None:
                return False
            # Accessing isVisible() will raise RuntimeError if C++ object deleted
            _ = self.native.isVisible()
            return True
        except (RuntimeError, AttributeError):
            self._is_closed = True
            return False

    def on_close_callback(self, callback: callable) -> None:
        """
        Register a callback to be called when the viewer closes.

        Args:
            callback: Function to call with no arguments when viewer closes.
        """
        self._close_callbacks.append(callback)

    def on_error_callback(self, callback: callable) -> None:
        """
        Register a callback to be called when an error occurs.

        Args:
            callback: Function to call with error message string.
        """
        self._error_callbacks.append(callback)

    def _notify_closed(self) -> None:
        """Notify all registered close callbacks."""
        for callback in self._close_callbacks:
            try:
                callback()
            except Exception as e:
                _logger.debug(f"Close callback error: {e}")

    def _notify_error(self, message: str) -> None:
        """Notify all registered error callbacks."""
        _logger.error(f"Viewer error: {message}")
        for callback in self._error_callbacks:
            try:
                callback(message)
            except Exception as e:
                _logger.debug(f"Error callback error: {e}")

    def close(self) -> None:
        """Close the viewer and notify callbacks."""
        if self._is_closed:
            return

        self._is_closed = True
        _logger.debug("VispyMeshViewer closing")

        try:
            self._notify_closed()
        except Exception as e:
            _logger.debug(f"Error in close callbacks: {e}")

        try:
            super().close()
        except Exception as e:
            _logger.debug(f"Error closing SceneCanvas: {e}")

    def _safe_update(self) -> bool:
        """
        Safely update the viewer, catching OpenGL/Qt errors.

        Returns:
            True if update succeeded, False otherwise.
        """
        if not self.is_alive:
            return False
        try:
            self.update()
            return True
        except (RuntimeError, Exception) as e:
            error_msg = f"Render error: {e}"
            _logger.warning(error_msg)
            self._notify_error(error_msg)
            return False

    # ------------------------------------------------------------------------
    # Scene Management
    # ------------------------------------------------------------------------
    
    def clear_all_visuals(self) -> None:
        """Remove all visuals from the scene."""
        for mesh_visual in self.mesh_visuals:
            if mesh_visual.visual.parent is not None:
                mesh_visual.visual.parent = None
        self.mesh_visuals.clear()
    
    def _configure_camera(self, distance: float = DEFAULT_CAMERA_DISTANCE,
                         axis_visible: bool = True) -> None:
        """
        Configure camera settings.

        Parameters:
        -----------
        distance : float
            Camera distance from origin
        axis_visible : bool
            Whether to show coordinate axis
        """
        self.view.camera.up = '-z'
        self.view.camera.distance = distance
        # Default to X-axis view (same as pressing 'X')
        self.view.camera.elevation = 0
        self.view.camera.azimuth = 0
        self.axis.visible = axis_visible
    
    # ------------------------------------------------------------------------
    # Adding Visuals (internal methods)
    # ------------------------------------------------------------------------

    def _add_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: Optional[np.ndarray] = None,
        color: Union[str, Tuple] = DEFAULT_MESH_COLOR,
        shading: str = DEFAULT_MESH_SHADING,
        name: Optional[str] = None
    ) -> MeshVisual:
        """
        Add a mesh visual to the scene (internal method).

        Use load_mesh_from_trimesh() for the public API.

        Parameters:
        -----------
        vertices : ndarray
            Vertex positions (N, 3)
        faces : ndarray
            Face indices (M, 3)
        vertex_colors : ndarray, optional
            Vertex colors (N, 3) or (N, 4)
        color : str or tuple, optional
            Default color if vertex_colors not provided
        shading : str, optional
            'smooth' or 'flat'
        name : str, optional
            Name for this visual

        Returns:
        --------
        MeshVisual object
        """
        use_vertex_colors = (vertex_colors is not None and 
                           vertex_colors.size > 0)
        
        mesh_visual = visuals.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors if use_vertex_colors else None,
            color=color if not use_vertex_colors else None,
            shading=shading
        )
        mesh_visual.parent = self.view.scene

        # Apply current lighting preset
        self._apply_lighting_to_mesh(mesh_visual)
        
        mv = MeshVisual(
            visual=mesh_visual,
            visual_type=VisualType.MESH,
            name=name
        )
        self.mesh_visuals.append(mv)
        return mv
    
    def _add_wireframe(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Tuple[float, ...] = DEFAULT_WIREFRAME_COLOR,
        line_width: float = DEFAULT_LINE_WIDTH,
        is_target: bool = False,
        name: Optional[str] = None
    ) -> MeshVisual:
        """
        Add a wireframe visual to the scene (internal method).

        Use add_wireframe_from_trimesh() for the public API.

        Parameters:
        -----------
        vertices : ndarray
            Vertex positions (N, 3)
        faces : ndarray
            Face indices (M, 3) or edge indices (M, 2)
        color : tuple
            RGBA color (0-1 range)
        line_width : float
            Width of the lines
        is_target : bool
            If True, wireframe is a target (not toggleable with 'w' key)
        name : str, optional
            Name for this visual

        Returns:
        --------
        MeshVisual object
        """
        # Convert faces to edges if needed
        if faces.shape[1] == 3:
            edges = trimesh_utils.compute_edges_from_faces(faces)
        else:
            edges = faces
        
        # Create Line visual using edges as connectivity
        line_visual = visuals.Line(
            pos=vertices,
            connect=edges,
            color=color,
            width=line_width,
            method='gl'
        )
        line_visual.parent = self.view.scene
        
        visual_type = (VisualType.TARGET_WIREFRAME if is_target 
                      else VisualType.SOURCE_WIREFRAME)
        
        mv = MeshVisual(
            visual=line_visual,
            visual_type=visual_type,
            name=name
        )
        self.mesh_visuals.append(mv)

        # Sync initial visibility with the viewer's toggle state
        if visual_type == VisualType.SOURCE_WIREFRAME:
            mv.set_visibility(self.source_wireframe_visible)
        elif visual_type == VisualType.TARGET_WIREFRAME:
            mv.set_visibility(self.target_wireframe_visible)

        return mv
    
    # ------------------------------------------------------------------------
    # Trimesh Integration
    # ------------------------------------------------------------------------
    
    def load_mesh_from_trimesh(
        self,
        trimesh_mesh: trimesh.Trimesh,
        color: Union[str, Tuple] = DEFAULT_MESH_COLOR,
        shading: str = DEFAULT_MESH_SHADING,
        vertex_colors: bool = True,
        name: Optional[str] = None
    ) -> Optional[MeshVisual]:
        """
        Load a trimesh mesh and add it to the scene.
        
        Parameters:
        -----------
        trimesh_mesh : trimesh.Trimesh
            Trimesh mesh object
        color : str or tuple
            Default color if vertex colors not available
        shading : str
            'smooth' or 'flat'
        vertex_colors : bool
            Whether to use vertex colors if available
        name : str, optional
            Name for this visual
        
        Returns:
        --------
        MeshVisual object or None if failed
        """
        if not trimesh_utils.validate_trimesh(trimesh_mesh, "mesh loading"):
            return None
        
        # Ensure normals are computed
        _ = trimesh_mesh.vertex_normals
        
        vertices = trimesh_mesh.vertices
        faces = trimesh_mesh.faces
        
        vcolors = self._extract_vertex_colors(trimesh_mesh) if vertex_colors else None

        return self._add_mesh(
            vertices, faces,
            vertex_colors=vcolors,
            color=color,
            shading=shading,
            name=name
        )

    def add_wireframe_from_trimesh(
            self,
            trimesh_mesh: trimesh.Trimesh,
            color: Tuple[float, ...] = DEFAULT_WIREFRAME_COLOR,
            line_width: float = DEFAULT_LINE_WIDTH,
            is_target: bool = False,
            name: Optional[str] = None
        ) -> Optional[MeshVisual]:
            """
            Create and add wireframe from a trimesh mesh.
            
            Parameters:
            -----------
            trimesh_mesh : trimesh.Trimesh
                Trimesh mesh object
            color : tuple
                RGBA color (0-1 range)
            line_width : float
                Width of the lines
            is_target : bool
                If True, wireframe is a target (not toggleable with 'w' key)
            name : str, optional
                Name for this visual
            
            Returns:
            --------
            MeshVisual object or None if failed
            """
            try:
                edges = trimesh_mesh.edges_unique
                
                if len(edges) == 0:
                    print("Error: Failed to create edges from mesh")
                    return None
                
                # Validate edges reference valid vertices
                max_vertex_index = len(trimesh_mesh.vertices) - 1
                if np.any(edges > max_vertex_index) or np.any(edges < 0):
                    print("Error: Edge indices out of bounds")
                    return None
                
                return self._add_wireframe(
                    trimesh_mesh.vertices, edges,
                    color=color,
                    line_width=line_width,
                    is_target=is_target,
                    name=name
                )
            except Exception as e:
                print(f"Error creating wireframe: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    @staticmethod
    def _extract_vertex_colors(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        """
        Extract vertex colors from a trimesh mesh.
        
        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Mesh with potential vertex colors
        
        Returns:
        --------
        ndarray or None : Vertex colors in 0-1 range, or None if not available
        """
        if mesh.visual.kind != 'vertex':
            return None
        
        vcolors = mesh.visual.vertex_colors
        
        # Convert to 0-1 range if needed
        if vcolors.dtype == np.uint8:
            vcolors = vcolors.astype(np.float32) / 255.0
        
        return vcolors
    
    # ------------------------------------------------------------------------
    # High-level Loading Methods
    # ------------------------------------------------------------------------
    
    def load_and_display_aligned_with_target(
        self,
        aligned_file_path: str,
        target_mesh_path: str,
        target_color: Union[Tuple, List],
        show_source_wireframe: bool = True,
        source_wireframe_color: Tuple[float, ...] = DEFAULT_WIREFRAME_COLOR
    ) -> bool:
        """
        Load and display both an aligned mesh and target wireframe.
        This is the main method for the shape browser integration.

        Parameters:
        -----------
        aligned_file_path : str
            Path to aligned mesh file (.ply format)
        target_mesh_path : str
            Path to target mesh file (.off or .ply format)
        target_color : tuple or list
            RGB color for target wireframe (0-1 range)
        show_source_wireframe : bool
            Whether to add a source wireframe (toggleable with 'W' key)
        source_wireframe_color : tuple
            RGBA color for source wireframe (0-1 range)

        Returns:
        --------
        bool : True if successful, False otherwise
        """
        self.clear_all_visuals()

        try:
            # Load aligned mesh
            try:
                aligned_mesh = trimesh.load(aligned_file_path, process=False)
            except Exception as e:
                print(f"Error loading aligned mesh from {aligned_file_path}: {e}")
                return False
            if not trimesh_utils.validate_trimesh(aligned_mesh, f"loading from {aligned_file_path}"):
                return False

            self.load_mesh_from_trimesh(aligned_mesh, name='aligned_mesh')

            # Add source wireframe for the aligned mesh (toggleable with 'W' key)
            if show_source_wireframe:
                wf = self.add_wireframe_from_trimesh(
                    aligned_mesh,
                    color=source_wireframe_color,
                    line_width=DEFAULT_LINE_WIDTH,
                    is_target=False,  # Source wireframe - toggleable with 'W'
                    name='source_wireframe'
                )
                # Set initial visibility to match the wireframe state
                if wf is not None:
                    wf.set_visibility(self.source_wireframe_visible)

            # Load target mesh for wireframe
            try:
                target_mesh = trimesh.load(target_mesh_path, process=False)
            except Exception as e:
                print(f"Error loading target mesh from {target_mesh_path}: {e}")
                return False
            if not trimesh_utils.validate_trimesh(target_mesh, f"loading from {target_mesh_path}"):
                return False

            # Add target wireframe
            rgba = trimesh_utils.convert_color_to_rgba(target_color)
            self.add_wireframe_from_trimesh(
                target_mesh,
                color=rgba,
                is_target=True,
                name='target_wireframe'
            )

            # Reset camera to fit mesh
            self.fit_to_meshes()

            return True

        except Exception as e:
            print(f"Error loading mesh: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ------------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------------
    
    def on_key_press(self, event) -> None:
        """Handle keyboard events."""
        key = event.key.name if hasattr(event.key, 'name') else str(event.key)
        
        # Check if parent handler wants to handle this key
        if self.parent_handler is not None:
            self.parent_handler.handle_key_press(key)
            return
        
        # Otherwise, handle internally
        self._handle_viewer_key(key)
    
    def _handle_viewer_key(self, key: str) -> None:
        """
        Internal key handler for viewer-specific controls.
        
        Parameters:
        -----------
        key : str
            Key name or character
        """
        key_lower = key.lower()
        
        handlers = {
            'w': self._toggle_source_wireframe,
            't': self._toggle_target_wireframe,
            'a': self._toggle_axis,
            'b': self._toggle_background,
            'c': self._toggle_backface_culling,
            'f': self._fit_view,
            'd': self._cycle_diffuse,        # Cycle diffuse/shadow intensity
            's': self._toggle_shading,       # Toggle smooth/flat shading
            'g': self._toggle_specular,      # Toggle specular/luster effect
            'x': lambda: self._set_camera_view(CameraPreset.X_AXIS),
            'y': lambda: self._set_camera_view(CameraPreset.Y_AXIS),
            'z': lambda: self._set_camera_view(CameraPreset.Z_AXIS),
            'h': self.print_help,
        }
        
        if key_lower in handlers:
            handlers[key_lower]()
        elif key == 'Escape':
            print("Closing viewer...")
            self.close()
        else:
            print(f"Unhandled key press: {key}")
    
    def _toggle_source_wireframe(self) -> None:
        """Toggle visibility of source wireframes."""
        self.source_wireframe_visible = not self.source_wireframe_visible
        
        for mv in self.mesh_visuals:
            if mv.is_source_wireframe():
                mv.set_visibility(self.source_wireframe_visible)
        
        status = 'ON' if self.source_wireframe_visible else 'OFF'
        print(f"Source Wireframe: {status}")
        self.update()

    def _toggle_target_wireframe(self) -> None:
        """Toggle visibility of target wireframes."""
        self.target_wireframe_visible = not self.target_wireframe_visible

        for mv in self.mesh_visuals:
            if mv.is_target_wireframe():
                mv.set_visibility(self.target_wireframe_visible)

        status = 'ON' if self.target_wireframe_visible else 'OFF'
        print(f"Target Wireframe: {status}")
        self.update()

    def _toggle_axis(self) -> None:
        """Toggle visibility of coordinate axis."""
        self.axis_visible = not self.axis_visible
        self.axis.visible = self.axis_visible
        
        status = 'ON' if self.axis_visible else 'OFF'
        print(f"Coordinate Axis: {status}")
        self.update()
    
    def _toggle_background(self) -> None:
        """Toggle background color between black and white."""
        self.bg_is_black = not self.bg_is_black
        new_color = DEFAULT_BG_COLOR if self.bg_is_black else DEFAULT_BG_COLOR_ALT
        self.bgcolor = new_color

        color_name = 'BLACK' if self.bg_is_black else 'WHITE'
        print(f"Background: {color_name}")
        self.update()

    def _apply_lighting_to_mesh(self, mesh_visual) -> None:
        """Apply current lighting settings to a single mesh visual.

        Lighting is controlled by four independent settings:
        - shading_mode: 'smooth', 'flat', or None (controls normal interpolation)
        - diffuse: shadow intensity (from DIFFUSE_PRESETS)
        - specular: highlight/luster effect (like Open3D default)
        - light_dir: direction the light comes from (synced with camera view)
        """
        # Set the shading mode (smooth/flat/none)
        mesh_visual.shading = self.shading_mode

        # Apply lighting coefficients if shading is enabled
        if self.shading_mode is not None and mesh_visual.shading_filter is not None:
            # Light direction (synced with camera view presets)
            mesh_visual.shading_filter.light_dir = self.light_dir

            # Ambient controls base brightness (higher = brighter overall)
            mesh_visual.shading_filter.ambient_coefficient = DEFAULT_AMBIENT_COEFFICIENT

            # Diffuse controls shadow intensity
            _, diffuse = DIFFUSE_PRESETS[self.diffuse_index]
            mesh_visual.shading_filter.diffuse_coefficient = diffuse

            # Specular creates the "luster" highlight effect (like Open3D)
            if self.specular_enabled:
                mesh_visual.shading_filter.specular_coefficient = DEFAULT_SPECULAR_COEFFICIENT
                mesh_visual.shading_filter.shininess = DEFAULT_SHININESS
            else:
                mesh_visual.shading_filter.specular_coefficient = (0, 0, 0, 1)
                mesh_visual.shading_filter.shininess = 1

    def _cycle_diffuse(self) -> None:
        """Cycle through diffuse intensity presets (shadow strength)."""
        # Cycle to next preset
        self.diffuse_index = (self.diffuse_index + 1) % len(DIFFUSE_PRESETS)
        preset_name, diffuse = DIFFUSE_PRESETS[self.diffuse_index]

        # Apply to all mesh visuals
        for mv in self.mesh_visuals:
            if mv.is_mesh():
                try:
                    self._apply_lighting_to_mesh(mv.visual)
                except Exception as e:
                    print(f"Warning: Could not apply lighting to mesh: {e}")

        print(f"Diffuse: {preset_name.upper()} ({diffuse})")
        self.update()

    def _toggle_backface_culling(self) -> None:
        """Toggle backface culling for mesh visuals."""
        self.show_backface = not self.show_backface

        for mv in self.mesh_visuals:
            if mv.is_mesh():
                try:
                    state = 'translucent' if self.show_backface else 'opaque'
                    mv.visual.set_gl_state(state, cull_face=not self.show_backface)
                except Exception:
                    pass

        status = 'OFF' if self.show_backface else 'ON'
        print(f"Backface Culling: {status}")
        self.update()
    
    def _fit_view(self) -> None:
        """Fit all objects in view (keyboard handler)."""
        self.fit_to_meshes()
        print("Fit objects in view")
    
    def _toggle_shading(self) -> None:
        """Toggle between smooth, flat, and uniform (no shading) modes.

        This only affects normal interpolation, not diffuse/specular settings.
        - smooth: interpolated normals (smoother look)
        - flat: face normals (faceted look)
        - None (uniform): no shading, flat color only
        """
        # Cycle through: smooth -> flat -> None (uniform) -> smooth
        if self.shading_mode == 'smooth':
            self.shading_mode = 'flat'
        elif self.shading_mode == 'flat':
            self.shading_mode = None
        else:  # None
            self.shading_mode = 'smooth'

        # Apply to all mesh visuals
        for mv in self.mesh_visuals:
            if mv.is_mesh():
                try:
                    self._apply_lighting_to_mesh(mv.visual)
                except Exception:
                    pass

        mode_name = self.shading_mode.upper() if self.shading_mode else 'UNIFORM'
        print(f"Shading: {mode_name}")
        self.update()

    def _toggle_specular(self) -> None:
        """Toggle specular highlights (luster effect) on/off."""
        self.specular_enabled = not self.specular_enabled

        # Apply to all mesh visuals
        for mv in self.mesh_visuals:
            if mv.is_mesh():
                try:
                    self._apply_lighting_to_mesh(mv.visual)
                except Exception:
                    pass

        status = 'ON' if self.specular_enabled else 'OFF'
        print(f"Specular (luster): {status}")
        self.update()
    
    def _set_camera_view(self, preset: CameraPreset) -> None:
        """
        Set camera to a predefined view angle and sync light direction.

        Parameters:
        -----------
        preset : CameraPreset
            Camera preset to use (includes matching light direction)
        """
        if not hasattr(self.view.camera, 'elevation'):
            print(f"Camera does not support preset views")
            return

        self.view.camera.elevation = preset.elevation
        self.view.camera.azimuth = preset.azimuth

        # Sync light direction with camera view
        self.light_dir = preset.light_dir

        # Apply updated light direction to all mesh visuals
        for mv in self.mesh_visuals:
            if mv.is_mesh():
                try:
                    self._apply_lighting_to_mesh(mv.visual)
                except Exception:
                    pass

        print(f"View: {preset.view_name.upper()}-axis (light synced)")
        self.update()
    
    def print_help(self) -> None:
        """Print keyboard controls help."""
        print(HELP_TEXT)


    def display_cut_visualization(self, kept_mesh, cut_mesh, kept_color=(1, 0.706, 0), 
                                cut_color=(0, 0.651, 0.929), show_wireframe=True):
        """
        Display two meshes for cut visualization (kept part and cut part).
        FIXED VERSION with better error handling for shape mismatches.
        
        Parameters:
        -----------
        kept_mesh : trimesh.Trimesh or None
            The mesh part that will be kept (shown in kept_color)
        cut_mesh : trimesh.Trimesh or None
            The mesh part that will be cut (shown in cut_color)
        kept_color : tuple
            RGB color for kept mesh (0-1 range), default yellow-orange
        cut_color : tuple
            RGB color for cut mesh (0-1 range), default blue
        show_wireframe : bool
            Whether to show wireframe on the meshes
        
        Returns:
        --------
        bool : True if successful, False otherwise
        """
        # Clear existing visuals
        self.clear_all_visuals()
        
        def is_valid_mesh(mesh):
            """Check if a mesh is valid and renderable."""
            if mesh is None:
                return False
            if not hasattr(mesh, 'faces') or not hasattr(mesh, 'vertices'):
                return False
            if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
                return False
            # Check for degenerate geometry
            if len(mesh.vertices) < 3:
                return False
            try:
                # Try to compute a basic property to ensure mesh is valid
                _ = mesh.bounds
                return True
            except Exception:
                return False
        
        def clean_mesh_for_display(mesh):
            """
            Clean a mesh to ensure it can be displayed properly.
            Removes vertex colors if they cause issues.
            """
            try:
                # Create a copy to avoid modifying the original
                cleaned = mesh.copy()
                
                # Validate vertex colors
                if cleaned.visual.kind == 'vertex':
                    vc = cleaned.visual.vertex_colors
                    
                    # Check if vertex colors match vertex count
                    if not isinstance(vc, np.ndarray) or len(vc) != len(cleaned.vertices):
                        print(f"  Warning: Vertex color count mismatch, removing colors")
                        # Create mesh without vertex colors
                        cleaned = trimesh.Trimesh(
                            vertices=cleaned.vertices,
                            faces=cleaned.faces,
                            process=False
                        )
                
                # Ensure edges can be computed
                try:
                    _ = cleaned.edges_unique
                except Exception as e:
                    print(f"  Warning: Cannot compute edges: {e}")
                    return None
                
                return cleaned
                
            except Exception as e:
                print(f"  Error cleaning mesh: {e}")
                return None
        
        success = False
        
        try:
            # Add kept mesh (yellow part - always spared)
            if is_valid_mesh(kept_mesh):
                print("Adding kept mesh...")
                cleaned_kept = clean_mesh_for_display(kept_mesh)
                
                if cleaned_kept is not None:
                    try:
                        # Try to load with vertex colors first
                        self.load_mesh_from_trimesh(
                            cleaned_kept, 
                            color=kept_color,
                            vertex_colors=True,
                            name='kept_mesh'
                        )
                        success = True
                        print("   Kept mesh loaded with vertex colors")
                    except Exception as e:
                        print(f"  Warning: Could not load with vertex colors: {e}")
                        try:
                            # Try without vertex colors
                            mesh_no_colors = trimesh.Trimesh(
                                vertices=cleaned_kept.vertices,
                                faces=cleaned_kept.faces,
                                process=False
                            )
                            self.load_mesh_from_trimesh(
                                mesh_no_colors,
                                color=kept_color,
                                vertex_colors=False,
                                name='kept_mesh'
                            )
                            success = True
                            print("   Kept mesh loaded without vertex colors")
                        except Exception as e2:
                            print(f"  Error loading kept mesh: {e2}")
                            import traceback
                            traceback.print_exc()
                    
                    # Add wireframe (source wireframe - toggleable with 'W' key)
                    if show_wireframe and success:
                        try:
                            rgba = trimesh_utils.convert_color_to_rgba(kept_color)
                            self.add_wireframe_from_trimesh(
                                cleaned_kept,
                                color=rgba,
                                line_width=1.0,
                                is_target=False,  # Source wireframe - toggleable with 'W'
                                name='kept_wireframe'
                            )
                            print("   Kept wireframe added")
                        except Exception as e:
                            print(f"  Warning: Could not add kept wireframe: {e}")
                else:
                    print("  Warning: Could not clean kept mesh")
            else:
                print("Warning: Kept mesh is invalid or empty")
            
            # Add cut mesh (blue part - will be cut off)
            if is_valid_mesh(cut_mesh):
                print("Adding cut mesh...")
                cleaned_cut = clean_mesh_for_display(cut_mesh)
                
                if cleaned_cut is not None:
                    try:
                        # Try to load with vertex colors first
                        self.load_mesh_from_trimesh(
                            cleaned_cut,
                            color=cut_color,
                            vertex_colors=True,
                            name='cut_mesh'
                        )
                        success = True
                        print("   Cut mesh loaded with vertex colors")
                    except Exception as e:
                        print(f"  Warning: Could not load with vertex colors: {e}")
                        try:
                            # Try without vertex colors
                            mesh_no_colors = trimesh.Trimesh(
                                vertices=cleaned_cut.vertices,
                                faces=cleaned_cut.faces,
                                process=False
                            )
                            self.load_mesh_from_trimesh(
                                mesh_no_colors,
                                color=cut_color,
                                vertex_colors=False,
                                name='cut_mesh'
                            )
                            success = True
                            print("   Cut mesh loaded without vertex colors")
                        except Exception as e2:
                            print(f"  Error loading cut mesh: {e2}")
                            import traceback
                            traceback.print_exc()
                    
                    # Add wireframe (source wireframe - toggleable with 'W' key)
                    if show_wireframe and success:
                        try:
                            rgba = trimesh_utils.convert_color_to_rgba(cut_color)
                            self.add_wireframe_from_trimesh(
                                cleaned_cut,
                                color=rgba,
                                line_width=1.0,
                                is_target=False,  # Source wireframe - toggleable with 'W'
                                name='cut_wireframe'
                            )
                            print("   Cut wireframe added")
                        except Exception as e:
                            print(f"  Warning: Could not add cut wireframe: {e}")
                else:
                    print("  Warning: Could not clean cut mesh")
            else:
                print("Warning: Cut mesh is invalid or empty")
            
            # Only reset camera if we successfully added at least one mesh
            if success:
                try:
                    #self.view.camera.set_range()
                    self.fit_to_meshes()
                    print("Camera reset")
                except Exception as e:
                    print(f"Warning: Could not reset camera: {e}")
            
            return success
            
        except Exception as e:
            print(f"Error displaying cut visualization: {e}")
            import traceback
            traceback.print_exc()
            return False

    def fit_to_meshes(self) -> None:
        """
        Fit camera view to all loaded meshes.

        This method computes the combined bounding box of all visible visuals
        and adjusts the TurntableCamera's center and distance to frame them.

        Note: Unlike set_range() which works well for 2D cameras, TurntableCamera
        requires explicit setting of 'center' and 'distance' properties for proper
        framing of 3D content.
        """
        try:
            # Collect vertices from all visible visuals
            all_vertices = []

            for mv in self.mesh_visuals:
                if not mv.visible:
                    continue

                vertices = self._get_visual_vertices(mv.visual)
                if vertices is not None and len(vertices) > 0:
                    all_vertices.append(vertices)

            if not all_vertices:
                # No visible geometry - nothing to fit
                return

            # Combine all vertices and compute bounding box
            combined_vertices = np.vstack(all_vertices)
            min_bounds = combined_vertices.min(axis=0)
            max_bounds = combined_vertices.max(axis=0)

            # Compute center and extent
            center = (min_bounds + max_bounds) / 2.0
            extent = max_bounds - min_bounds
            max_extent = float(np.max(extent))

            # Handle degenerate cases (flat or point geometry)
            if max_extent < 1e-6:
                max_extent = 1.0

            # Set camera center to the center of the bounding box
            self.view.camera.center = center

            # Compute appropriate distance based on FOV
            # Distance = (extent/2) / tan(fov/2) gives the distance where the extent
            # exactly fills the view. We add a margin factor for comfortable framing.
            fov = getattr(self.view.camera, 'fov', DEFAULT_FOV)
            margin_factor = DEFAULT_MARGIN_FACTOR  # margin for comfortable viewing

            if fov > 0:
                fov_rad = np.radians(fov)
                half_fov_tan = np.tan(fov_rad / 2.0)
                if half_fov_tan > 1e-6:
                    distance = (max_extent / 2.0) / half_fov_tan * margin_factor
                else:
                    distance = max_extent * 2.0
            else:
                # Orthographic projection (fov=0) - use scale_factor instead
                distance = max_extent * 2.0

            # Ensure minimum distance to avoid clipping issues
            min_distance = max_extent * 0.1
            distance = max(distance, min_distance)

            self.view.camera.distance = distance
            self.update()

        except Exception as e:
            print(f"Error fitting view: {e}")
            import traceback
            traceback.print_exc()

    def _get_visual_vertices(self, visual) -> Optional[np.ndarray]:
        """
        Extract vertex positions from a VisPy visual.

        Parameters
        ----------
        visual : vispy.scene.visuals.Visual
            The visual to extract vertices from (Mesh or Line)

        Returns
        -------
        np.ndarray or None
            Vertex positions as (N, 3) array, or None if extraction failed
        """
        try:
            # Try Mesh visual first (has mesh_data attribute)
            if hasattr(visual, 'mesh_data') and visual.mesh_data is not None:
                vertices = visual.mesh_data.get_vertices()
                if vertices is not None and len(vertices) > 0:
                    return np.asarray(vertices)

            # Try Line visual (has pos property or _pos attribute)
            # The 'pos' property may trigger computation, so try _pos first
            if hasattr(visual, '_pos') and visual._pos is not None:
                pos = visual._pos
                if len(pos) > 0:
                    return np.asarray(pos)

            # Fallback: try the pos property directly
            if hasattr(visual, 'pos'):
                pos = visual.pos
                if pos is not None and len(pos) > 0:
                    return np.asarray(pos)

            return None

        except Exception:
            # Silently fail for individual visuals - other visuals may still work
            return None

    def show_and_activate(self) -> None:
        """
        Show the viewer window and bring it to the foreground.

        This is the recommended way to display the viewer window, as it
        ensures proper window activation across different platforms.
        """
        try:
            self.show()
            self.native.raise_()
            self.native.activateWindow()
            self.print_help()
        except Exception as e:
            print(f"Error showing viewer: {e}")

# ============================================================================
# Standalone Viewer Factory
# ============================================================================

def create_standalone_viewer(
    meshes: Optional[List[Tuple]] = None,
    wireframes: Optional[List[Tuple]] = None,
    title: str = '3D Mesh Viewer'
) -> VispyMeshViewer:
    """
    Create a standalone viewer with meshes and wireframes.
    
    Parameters:
    -----------
    meshes : list of tuples, optional
        List of (trimesh_mesh, color, name) tuples
    wireframes : list of tuples, optional
        List of (trimesh_mesh, color, line_width, is_target, name) tuples
    title : str
        Window title
    
    Returns:
    --------
    VispyMeshViewer object
    
    Examples:
    ---------
    >>> mesh = trimesh.load('model.ply')
    >>> viewer = create_standalone_viewer(
    ...     meshes=[(mesh, 'lightblue', 'my_mesh')],
    ...     title='My Viewer'
    ... )
    """
    viewer = VispyMeshViewer(title=title)
    
    if meshes:
        for mesh_data in meshes:
            trimesh_mesh = mesh_data[0]
            color = mesh_data[1] if len(mesh_data) > 1 else DEFAULT_MESH_COLOR
            name = mesh_data[2] if len(mesh_data) > 2 else None
            viewer.load_mesh_from_trimesh(trimesh_mesh, color=color, name=name)
    
    if wireframes:
        for wf_data in wireframes:
            trimesh_mesh = wf_data[0]
            color = wf_data[1] if len(wf_data) > 1 else DEFAULT_WIREFRAME_COLOR
            line_width = wf_data[2] if len(wf_data) > 2 else DEFAULT_LINE_WIDTH
            is_target = wf_data[3] if len(wf_data) > 3 else False
            name = wf_data[4] if len(wf_data) > 4 else None
            viewer.add_wireframe_from_trimesh(
                trimesh_mesh,
                color=color,
                line_width=line_width,
                is_target=is_target,
                name=name
            )
    
    viewer.fit_to_meshes()
    viewer.print_help()
    
    return viewer


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    """Example standalone usage."""
    from vispy import app
    from pathlib import Path

    example_source_path = Path(get_asset('example_source.ply'))
    example_target_path = Path(get_asset('benchmark_target.ply'))

    
    # Example: Load and display meshes
    try:
        # Load your mesh files here
        source_mesh = trimesh.load(example_source_path.as_posix(), process=False)
        target_mesh = trimesh.load(example_target_path.as_posix(), process=False)

        # validate meshes
        if not trimesh_utils.validate_trimesh(source_mesh, f"loading from {example_source_path}"):
            raise ValueError(f"Invalid source mesh: {example_source_path}")
        if not trimesh_utils.validate_trimesh(target_mesh, f"loading from {example_target_path}"):
            raise ValueError(f"Invalid target mesh: {example_target_path}")
        
        viewer = create_standalone_viewer(
            meshes=[(source_mesh, 'lightblue', 'source')],
            wireframes=[(target_mesh, (0, 1, 1, 1), 1.5, True, 'target')],
            title='Mesh Viewer Example'
        )
        
        app.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please provide valid mesh files for standalone testing")