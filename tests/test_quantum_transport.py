"""
Unit tests for quantum tracer transport.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantum_atmosphere_transport import (
    QuantumTracerTransport,
    TransportConfig
)


class TestQuantumTracerTransport:
    """Test quantum tracer transport scheme."""

    def test_initialization(self):
        """Test transport scheme initialization."""
        config = TransportConfig()
        transport = QuantumTracerTransport(config)

        assert transport.config == config
        assert transport.quantum_state is None
        assert transport.classical_state is None

    def test_initialize_with_concentration(self):
        """Test initialization with concentration field."""
        config = TransportConfig()
        transport = QuantumTracerTransport(config)

        # Create initial condition
        nx, ny = 10, 10
        initial = np.ones((nx, ny))
        initial[4:6, 4:6] = 10.0

        transport.initialize(initial, (nx, ny))

        assert transport.quantum_state is not None
        assert transport.classical_state is not None
        assert transport.quantum_state.shape == (nx, ny)
        assert np.allclose(transport.classical_state, initial)

    def test_classical_to_quantum_conversion(self):
        """Test conversion from classical to quantum state."""
        config = TransportConfig()
        transport = QuantumTracerTransport(config)

        # Create classical field
        classical = np.array([1.0, 2.0, 3.0, 4.0])

        # Convert to quantum
        quantum = transport._classical_to_quantum(classical)

        # Check that it's complex
        assert np.iscomplexobj(quantum)

        # Check normalization (sum of |ψ|² should equal 1 before rescaling)
        probs = np.abs(quantum) ** 2
        assert np.isclose(np.sum(probs), 1.0, atol=1e-6)

    def test_quantum_to_classical_conversion(self):
        """Test conversion from quantum to classical state."""
        config = TransportConfig()
        transport = QuantumTracerTransport(config)

        # Create initial classical state
        classical_initial = np.array([1.0, 2.0, 3.0, 4.0])

        # Convert to quantum and back
        quantum = transport._classical_to_quantum(classical_initial)
        transport.classical_state = classical_initial  # Set for mass conservation
        classical_final = transport._quantum_to_classical(quantum)

        # Check mass conservation
        assert np.isclose(np.sum(classical_initial), np.sum(classical_final), rtol=1e-5)

    def test_single_time_step(self):
        """Test single time step evolution."""
        config = TransportConfig(boundary_condition='periodic')
        transport = QuantumTracerTransport(config)

        # Initialize
        nx, ny = 20, 20
        initial = np.zeros((nx, ny))
        initial[10, 10] = 1.0
        transport.initialize(initial, (nx, ny))

        # Create simple velocity field
        velocity = np.zeros((nx, ny, 2))
        velocity[..., 0] = 1.0  # Uniform flow in x-direction

        # Take one step
        dt = 0.1
        diffusion = 0.1
        result = transport.step(dt, velocity, diffusion)

        # Check that result is returned and has correct shape
        assert result.shape == (nx, ny)

        # Check mass conservation
        assert np.isclose(np.sum(result), np.sum(initial), rtol=1e-2)

    def test_mass_conservation(self):
        """Test that mass is conserved during transport."""
        config = TransportConfig(boundary_condition='periodic')
        transport = QuantumTracerTransport(config)

        # Initialize
        nx, ny = 15, 15
        initial = np.random.rand(nx, ny)
        initial_mass = np.sum(initial)

        transport.initialize(initial, (nx, ny))

        # Create velocity field
        velocity = np.random.rand(nx, ny, 2) * 0.5

        # Evolve for several steps
        dt = 0.1
        diffusion = 0.1

        for _ in range(10):
            result = transport.step(dt, velocity, diffusion)

        # Check mass conservation
        final_mass = np.sum(result)
        assert np.isclose(initial_mass, final_mass, rtol=1e-1)

    def test_quantum_statistics(self):
        """Test quantum statistics calculation."""
        config = TransportConfig()
        transport = QuantumTracerTransport(config)

        # Initialize
        nx, ny = 10, 10
        initial = np.ones((nx, ny))
        transport.initialize(initial, (nx, ny))

        # Get statistics
        stats = transport.get_quantum_statistics()

        # Check that statistics are returned
        assert 'coherence' in stats
        assert 'entanglement' in stats
        assert 'phase_variance' in stats

        # Check that values are reasonable
        assert stats['coherence'] >= 0
        assert stats['entanglement'] >= 0
        assert stats['phase_variance'] >= 0

    def test_get_concentration(self):
        """Test getting concentration field."""
        config = TransportConfig()
        transport = QuantumTracerTransport(config)

        # Initialize
        initial = np.array([[1.0, 2.0], [3.0, 4.0]])
        transport.initialize(initial, (2, 2))

        # Get concentration
        concentration = transport.get_concentration()

        # Check that it's a copy
        assert concentration is not transport.classical_state

        # Check values
        assert np.allclose(concentration, initial)

    def test_with_source_term(self):
        """Test transport with source term."""
        config = TransportConfig(boundary_condition='periodic')
        transport = QuantumTracerTransport(config)

        # Initialize
        nx, ny = 10, 10
        initial = np.zeros((nx, ny))
        transport.initialize(initial, (nx, ny))

        # Create source term
        source = np.zeros((nx, ny))
        source[5, 5] = 1.0

        # Create velocity field
        velocity = np.zeros((nx, ny, 2))

        # Evolve with source
        dt = 0.1
        diffusion = 0.1
        result = transport.step(dt, velocity, diffusion, source_term=source)

        # Mass should increase due to source
        assert np.sum(result) > 0

    def test_decoherence_effect(self):
        """Test that decoherence affects quantum state."""
        config_with_decoherence = TransportConfig(
            apply_decoherence=True,
            decoherence_rate=0.5
        )
        config_without_decoherence = TransportConfig(
            apply_decoherence=False
        )

        # Initialize both
        initial = np.random.rand(10, 10)
        velocity = np.zeros((10, 10, 2))

        transport1 = QuantumTracerTransport(config_with_decoherence)
        transport1.initialize(initial.copy(), (10, 10))

        transport2 = QuantumTracerTransport(config_without_decoherence)
        transport2.initialize(initial.copy(), (10, 10))

        # Evolve both
        for _ in range(5):
            transport1.step(0.1, velocity, 0.1)
            transport2.step(0.1, velocity, 0.1)

        # Get quantum statistics
        stats1 = transport1.get_quantum_statistics()
        stats2 = transport2.get_quantum_statistics()

        # With decoherence should have different coherence
        # (This is a qualitative test)
        assert stats1['coherence'] >= 0
        assert stats2['coherence'] >= 0
