"""
Tests for all transport methods.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantum_atmosphere_transport import (
    QuantumTracerTransport,
    ClassicalEulerianTransport,
    ClassicalSemiLagrangianTransport,
    ClassicalSpectralTransport,
    MonteCarloParticleTransport,
    FiniteVolumeTransport,
    HybridQuantumClassicalTransport,
    TransportConfig,
)


class TestAllMethods:
    """Test all transport methods with the same test case."""

    @pytest.fixture
    def config(self):
        """Create default configuration."""
        return TransportConfig(
            grid_spacing=1.0,
            boundary_condition='periodic',
            interpolation_order=2,
        )

    @pytest.fixture
    def test_problem(self):
        """Create simple 2D test problem."""
        nx, ny = 20, 20
        initial = np.zeros((nx, ny))
        initial[10, 10] = 1.0

        velocity = np.zeros((nx, ny, 2))
        velocity[..., 0] = 0.5

        return {
            'initial': initial,
            'velocity': velocity,
            'grid_shape': (nx, ny),
        }

    def test_quantum_method(self, config, test_problem):
        """Test quantum transport method."""
        transport = QuantumTracerTransport(config)
        transport.initialize(test_problem['initial'], test_problem['grid_shape'])

        result = transport.step(0.1, test_problem['velocity'], 0.1)

        assert result.shape == test_problem['grid_shape']
        assert np.sum(result) > 0

    def test_classical_eulerian(self, config, test_problem):
        """Test classical Eulerian method."""
        transport = ClassicalEulerianTransport(config)
        transport.initialize(test_problem['initial'], test_problem['grid_shape'])

        result = transport.step(0.1, test_problem['velocity'], 0.1)

        assert result.shape == test_problem['grid_shape']
        assert np.all(result >= 0)

    def test_classical_semi_lagrangian(self, config, test_problem):
        """Test classical semi-Lagrangian method."""
        transport = ClassicalSemiLagrangianTransport(config)
        transport.initialize(test_problem['initial'], test_problem['grid_shape'])

        result = transport.step(0.1, test_problem['velocity'], 0.1)

        assert result.shape == test_problem['grid_shape']
        assert np.all(result >= 0)

    def test_classical_spectral(self, config, test_problem):
        """Test classical spectral method."""
        transport = ClassicalSpectralTransport(config)
        transport.initialize(test_problem['initial'], test_problem['grid_shape'])

        result = transport.step(0.1, test_problem['velocity'], 0.1)

        assert result.shape == test_problem['grid_shape']
        assert np.all(result >= 0)

    def test_monte_carlo_particles(self, config, test_problem):
        """Test Monte Carlo particle method."""
        transport = MonteCarloParticleTransport(config, n_particles=1000)
        transport.initialize(test_problem['initial'], test_problem['grid_shape'])

        result = transport.step(0.1, test_problem['velocity'], 0.1)

        assert result.shape == test_problem['grid_shape']
        assert np.sum(result) > 0

    def test_finite_volume(self, config, test_problem):
        """Test finite volume method."""
        transport = FiniteVolumeTransport(config, flux_limiter='minmod')
        transport.initialize(test_problem['initial'], test_problem['grid_shape'])

        result = transport.step(0.1, test_problem['velocity'], 0.1)

        assert result.shape == test_problem['grid_shape']
        assert np.all(result >= 0)

    def test_hybrid_quantum_classical(self, config, test_problem):
        """Test hybrid quantum-classical method."""
        transport = HybridQuantumClassicalTransport(config, quantum_ratio=0.5)
        transport.initialize(test_problem['initial'], test_problem['grid_shape'])

        result = transport.step(0.1, test_problem['velocity'], 0.1)

        assert result.shape == test_problem['grid_shape']
        assert np.sum(result) > 0

    def test_mass_conservation_all_methods(self, config, test_problem):
        """Test that all methods conserve mass reasonably well."""
        methods = [
            QuantumTracerTransport(config),
            ClassicalSemiLagrangianTransport(config),
            ClassicalSpectralTransport(config),
            FiniteVolumeTransport(config),
            HybridQuantumClassicalTransport(config),
        ]

        initial_mass = np.sum(test_problem['initial'])

        for method in methods:
            method.initialize(test_problem['initial'].copy(), test_problem['grid_shape'])

            # Run several steps
            for _ in range(5):
                result = method.step(0.1, test_problem['velocity'], 0.1)

            final_mass = np.sum(result)
            relative_error = abs(final_mass - initial_mass) / initial_mass

            # Allow up to 20% error (some methods are approximate)
            assert relative_error < 0.2, f"{method.get_method_name()} failed mass conservation"

    def test_get_statistics(self, config, test_problem):
        """Test that all methods provide statistics."""
        methods = [
            QuantumTracerTransport(config),
            ClassicalSemiLagrangianTransport(config),
            MonteCarloParticleTransport(config, n_particles=500),
            HybridQuantumClassicalTransport(config),
        ]

        for method in methods:
            method.initialize(test_problem['initial'], test_problem['grid_shape'])
            method.step(0.1, test_problem['velocity'], 0.1)

            stats = method.get_statistics()

            assert isinstance(stats, dict)
            assert 'total_mass' in stats
            assert stats['total_mass'] > 0
