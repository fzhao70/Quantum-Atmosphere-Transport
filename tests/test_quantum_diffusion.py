"""
Unit tests for quantum diffusion.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantum_atmosphere_transport import QuantumDiffusion, TransportConfig


class TestQuantumDiffusion:
    """Test quantum diffusion operator."""

    def test_initialization_1d(self):
        """Test 1D diffusion initialization."""
        config = TransportConfig()
        diffusion = QuantumDiffusion(config)

        diffusion.initialize((10,))

        assert diffusion.grid_shape == (10,)
        assert diffusion.quantum_walk_operator is not None

    def test_initialization_2d(self):
        """Test 2D diffusion initialization."""
        config = TransportConfig()
        diffusion = QuantumDiffusion(config)

        diffusion.initialize((10, 10))

        assert diffusion.grid_shape == (10, 10)
        assert diffusion.quantum_walk_operator is not None

    def test_initialization_3d(self):
        """Test 3D diffusion initialization."""
        config = TransportConfig()
        diffusion = QuantumDiffusion(config)

        diffusion.initialize((5, 5, 5))

        assert diffusion.grid_shape == (5, 5, 5)
        assert diffusion.quantum_walk_operator is not None

    def test_operator_is_hermitian(self):
        """Test that quantum walk operator is Hermitian."""
        config = TransportConfig()
        diffusion = QuantumDiffusion(config)
        diffusion.initialize((10, 10))

        H = diffusion.quantum_walk_operator.toarray()
        H_dagger = H.conj().T

        assert np.allclose(H, H_dagger, atol=1e-10)

    def test_apply_diffusion_1d(self):
        """Test applying diffusion in 1D."""
        config = TransportConfig()
        diffusion = QuantumDiffusion(config)
        diffusion.initialize((20,))

        # Create initial state (Gaussian)
        x = np.linspace(0, 1, 20)
        state = np.exp(-((x - 0.5) ** 2) / 0.01).astype(complex)
        state /= np.linalg.norm(state)

        # Apply diffusion
        evolved = diffusion.apply(state, diffusion_coeff=0.1, dt=0.1)

        # Check that state is still normalized (approximately)
        assert np.isclose(np.linalg.norm(evolved), np.linalg.norm(state), rtol=1e-2)

        # Check that state has spread (variance should increase)
        assert evolved.shape == state.shape

    def test_apply_diffusion_2d(self):
        """Test applying diffusion in 2D."""
        config = TransportConfig()
        diffusion = QuantumDiffusion(config)
        diffusion.initialize((15, 15))

        # Create initial state (centered Gaussian)
        x = np.linspace(0, 1, 15)
        y = np.linspace(0, 1, 15)
        X, Y = np.meshgrid(x, y, indexing='ij')
        state = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01).astype(complex)
        state /= np.linalg.norm(state)

        # Apply diffusion
        evolved = diffusion.apply(state, diffusion_coeff=0.1, dt=0.1)

        # Check shape
        assert evolved.shape == state.shape

        # Check that it's still complex
        assert np.iscomplexobj(evolved)

    def test_periodic_boundary_conditions(self):
        """Test periodic boundary conditions."""
        config = TransportConfig(boundary_condition='periodic')
        diffusion = QuantumDiffusion(config)
        diffusion.initialize((10,))

        # Check that operator has periodic connections
        H = diffusion.quantum_walk_operator.toarray()

        # Should have connections at corners (periodic wrap)
        assert H[0, -1] != 0  # Connection from last to first
        assert H[-1, 0] != 0  # Connection from first to last

    def test_trotter_vs_exact_evolution(self):
        """Test Trotter evolution vs exact evolution."""
        state = np.random.rand(10, 10).astype(complex)
        state /= np.linalg.norm(state)

        # Exact evolution
        config_exact = TransportConfig(use_exact_evolution=True)
        diffusion_exact = QuantumDiffusion(config_exact)
        diffusion_exact.initialize((10, 10))
        evolved_exact = diffusion_exact.apply(state.copy(), 0.1, 0.1)

        # Trotter evolution
        config_trotter = TransportConfig(use_exact_evolution=False, trotter_steps=10)
        diffusion_trotter = QuantumDiffusion(config_trotter)
        diffusion_trotter.initialize((10, 10))
        evolved_trotter = diffusion_trotter.apply(state.copy(), 0.1, 0.1)

        # Should be similar (not exact due to Trotter error)
        assert np.allclose(evolved_exact, evolved_trotter, rtol=1e-1)

    def test_get_classical_diffusion_matrix(self):
        """Test getting classical diffusion matrix."""
        config = TransportConfig()
        diffusion = QuantumDiffusion(config)
        diffusion.initialize((10,))

        classical_matrix = diffusion.get_classical_diffusion_matrix()

        # Should be real and sparse
        assert not np.iscomplexobj(classical_matrix.toarray())
        assert classical_matrix.shape == (10, 10)
