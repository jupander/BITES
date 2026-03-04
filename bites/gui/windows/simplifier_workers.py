#!/usr/bin/env python

"""
Mesh simplifier workers - Background QThread workers for simplification and benchmarking.
"""

from __future__ import annotations
import logging
import time
import trimesh
import numpy as np
from typing import List, Callable
from PySide6.QtCore import QThread, Signal

from bites.gui.windows.simplifier_core import simplify_mesh

logger = logging.getLogger(__name__)


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
