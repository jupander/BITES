"""
Tests for bites.mesh.read_target module.

Covers:
- read_target_mesh_file: file extension dispatch (PLY, STL, OBJ, OFF, GLB, GLTF, 3MF)
- Scene concatenation for multi-mesh formats (GLB, GLTF)
- Error handling for unsupported formats

Mesh loading is mocked — no real mesh files needed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import trimesh

from bites.mesh.read_target import read_target_mesh_file


# ============================================================================
# TestReadTargetMeshFile
# ============================================================================

class TestReadTargetMeshFile:
    """Tests for read_target_mesh_file extension-based dispatch."""

    @patch("bites.mesh.read_target.trimesh")
    def test_ply_file_uses_trimesh_load(self, mock_trimesh, tmp_path):
        """PLY files should be loaded via trimesh.load."""
        mock_trimesh.Scene = trimesh.Scene
        ply = tmp_path / "target.ply"
        ply.write_text("ply\n")
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_trimesh.load.return_value = mock_mesh

        result = read_target_mesh_file(str(ply))
        mock_trimesh.load.assert_called_once_with(str(ply), process=False)
        assert result is mock_mesh

    @patch("bites.mesh.read_target.read_coff")
    def test_off_file_uses_read_coff(self, mock_read_coff, tmp_path):
        """OFF files should be loaded via read_coff.read_coff_mesh."""
        off = tmp_path / "target.off"
        off.write_text("OFF\n")
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_read_coff.read_coff_mesh.return_value = mock_mesh

        result = read_target_mesh_file(str(off))
        mock_read_coff.read_coff_mesh.assert_called_once_with(str(off))
        assert result is mock_mesh

    @patch("bites.mesh.read_target.trimesh")
    def test_stl_file_uses_trimesh_load(self, mock_trimesh, tmp_path):
        """STL files should be loaded via trimesh.load."""
        mock_trimesh.Scene = trimesh.Scene
        stl = tmp_path / "target.stl"
        stl.write_text("solid\n")
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_trimesh.load.return_value = mock_mesh

        result = read_target_mesh_file(str(stl))
        mock_trimesh.load.assert_called_once_with(str(stl), process=False)
        assert result is mock_mesh

    @patch("bites.mesh.read_target.trimesh")
    def test_obj_file_uses_trimesh_load(self, mock_trimesh, tmp_path):
        """OBJ files should be loaded via trimesh.load."""
        mock_trimesh.Scene = trimesh.Scene
        obj = tmp_path / "target.obj"
        obj.write_text("v 0 0 0\n")
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_trimesh.load.return_value = mock_mesh

        result = read_target_mesh_file(str(obj))
        mock_trimesh.load.assert_called_once_with(str(obj), process=False)
        assert result is mock_mesh

    @patch("bites.mesh.read_target.trimesh")
    def test_glb_file_uses_trimesh_load(self, mock_trimesh, tmp_path):
        """GLB files should be loaded via trimesh.load."""
        mock_trimesh.Scene = trimesh.Scene
        glb = tmp_path / "target.glb"
        glb.write_bytes(b"\x00")
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_trimesh.load.return_value = mock_mesh

        result = read_target_mesh_file(str(glb))
        mock_trimesh.load.assert_called_once_with(str(glb), process=False)
        assert result is mock_mesh

    @patch("bites.mesh.read_target.trimesh")
    def test_gltf_file_uses_trimesh_load(self, mock_trimesh, tmp_path):
        """GLTF files should be loaded via trimesh.load."""
        mock_trimesh.Scene = trimesh.Scene
        gltf = tmp_path / "target.gltf"
        gltf.write_text("{}\n")
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_trimesh.load.return_value = mock_mesh

        result = read_target_mesh_file(str(gltf))
        mock_trimesh.load.assert_called_once_with(str(gltf), process=False)
        assert result is mock_mesh

    @patch("bites.mesh.read_target.trimesh")
    def test_3mf_file_uses_trimesh_load(self, mock_trimesh, tmp_path):
        """3MF files should be loaded via trimesh.load."""
        mock_trimesh.Scene = trimesh.Scene
        mf = tmp_path / "target.3mf"
        mf.write_bytes(b"\x00")
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_trimesh.load.return_value = mock_mesh

        result = read_target_mesh_file(str(mf))
        mock_trimesh.load.assert_called_once_with(str(mf), process=False)
        assert result is mock_mesh

    @patch("bites.mesh.read_target.trimesh")
    def test_scene_result_is_concatenated(self, mock_trimesh, tmp_path):
        """When trimesh.load returns a Scene, it should be concatenated."""
        mock_trimesh.Scene = trimesh.Scene
        glb = tmp_path / "multi.glb"
        glb.write_bytes(b"\x00")
        mock_scene = MagicMock(spec=trimesh.Scene)
        mock_concatenated = MagicMock(spec=trimesh.Trimesh)
        mock_scene.dump.return_value = mock_concatenated
        mock_trimesh.load.return_value = mock_scene

        result = read_target_mesh_file(str(glb))
        mock_scene.dump.assert_called_once_with(concatenate=True)
        assert result is mock_concatenated

    @patch("bites.mesh.read_target.trimesh")
    def test_unsupported_extension_trimesh_fails_raises(self, mock_trimesh, tmp_path):
        """If trimesh.load fails for unsupported format, should raise ValueError."""
        mock_trimesh.Scene = trimesh.Scene
        txt = tmp_path / "target.txt"
        txt.write_text("not a mesh\n")
        mock_trimesh.load.side_effect = Exception("Unknown format")

        with pytest.raises(ValueError, match="Could not load"):
            read_target_mesh_file(str(txt))

    @patch("bites.mesh.read_target.trimesh")
    def test_accepts_path_object(self, mock_trimesh, tmp_path):
        """Should accept pathlib.Path objects, not just strings."""
        mock_trimesh.Scene = trimesh.Scene
        ply = tmp_path / "target.ply"
        ply.write_text("ply\n")
        mock_trimesh.load.return_value = MagicMock()

        result = read_target_mesh_file(ply)
        assert result is not None

    @patch("bites.mesh.read_target.trimesh")
    def test_case_insensitive_extension(self, mock_trimesh, tmp_path):
        """Extension matching should be case-insensitive."""
        mock_trimesh.Scene = trimesh.Scene
        ply = tmp_path / "target.PLY"
        ply.write_text("ply\n")
        mock_trimesh.load.return_value = MagicMock()

        result = read_target_mesh_file(str(ply))
        mock_trimesh.load.assert_called_once()
