"""
Hybrid Quantum-Classical Transport Methods

Combines quantum and classical approaches for enhanced performance.
"""

import numpy as np
from typing import Optional, Tuple
from .base_transport import BaseTransport
from .quantum_transport import QuantumTracerTransport
from .classical_transport import ClassicalSemiLagrangianTransport


class HybridQuantumClassicalTransport(BaseTransport):
    """
    Hybrid quantum-classical transport method.

    Uses quantum methods for diffusion (better spreading behavior)
    and classical semi-Lagrangian for advection (computational efficiency).
    """

    def __init__(self, config, quantum_ratio: float = 0.5):
        """
        Initialize hybrid transport.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        quantum_ratio : float
            Ratio of quantum to classical contribution (0.0-1.0)
            0.0 = fully classical, 1.0 = fully quantum
        """
        super().__init__(config)
        self.quantum_ratio = np.clip(quantum_ratio, 0.0, 1.0)

        # Initialize both quantum and classical schemes
        self.quantum_transport = QuantumTracerTransport(config)
        self.classical_transport = ClassicalSemiLagrangianTransport(config)

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize both quantum and classical components."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()

        # Initialize both methods
        self.quantum_transport.initialize(initial_concentration.copy(), grid_shape)
        self.classical_transport.initialize(initial_concentration.copy(), grid_shape)

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance by one time step using hybrid approach.

        Combines quantum and classical predictions.
        """
        # Get quantum prediction
        c_quantum = self.quantum_transport.step(
            dt, velocity_field, diffusion_coeff, source_term
        )

        # Get classical prediction
        c_classical = self.classical_transport.step(
            dt, velocity_field, diffusion_coeff, source_term
        )

        # Hybrid combination
        self.concentration = (
            self.quantum_ratio * c_quantum +
            (1 - self.quantum_ratio) * c_classical
        )

        return self.concentration.copy()

    def get_concentration(self) -> np.ndarray:
        """Get current concentration field."""
        return self.concentration.copy()

    def get_statistics(self) -> dict:
        """Get statistics from both methods."""
        base_stats = super().get_statistics()

        # Add quantum statistics
        quantum_stats = self.quantum_transport.get_quantum_statistics()

        return {
            **base_stats,
            'quantum_coherence': quantum_stats.get('coherence', 0),
            'quantum_entanglement': quantum_stats.get('entanglement', 0),
            'quantum_ratio': self.quantum_ratio,
        }


class AdaptiveQuantumClassicalTransport(BaseTransport):
    """
    Adaptive hybrid transport that switches between quantum and classical
    based on local flow conditions.

    Uses quantum methods in turbulent regions and classical in smooth regions.
    """

    def __init__(self, config, turbulence_threshold: float = 1.0):
        """
        Initialize adaptive hybrid transport.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        turbulence_threshold : float
            Threshold for switching to quantum method
        """
        super().__init__(config)
        self.turbulence_threshold = turbulence_threshold

        # Initialize both methods
        self.quantum_transport = QuantumTracerTransport(config)
        self.classical_transport = ClassicalSemiLagrangianTransport(config)

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize both components."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()

        self.quantum_transport.initialize(initial_concentration.copy(), grid_shape)
        self.classical_transport.initialize(initial_concentration.copy(), grid_shape)

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Adaptive step based on local turbulence indicators.
        """
        # Calculate turbulence indicator (velocity gradient magnitude)
        turbulence_indicator = self._calculate_turbulence(velocity_field)

        # Get both predictions
        c_quantum = self.quantum_transport.step(
            dt, velocity_field, diffusion_coeff, source_term
        )

        c_classical = self.classical_transport.step(
            dt, velocity_field, diffusion_coeff, source_term
        )

        # Adaptive blending based on turbulence
        # High turbulence → more quantum, Low turbulence → more classical
        quantum_weight = np.minimum(turbulence_indicator / self.turbulence_threshold, 1.0)
        classical_weight = 1.0 - quantum_weight

        self.concentration = quantum_weight * c_quantum + classical_weight * c_classical

        return self.concentration.copy()

    def _calculate_turbulence(self, velocity_field: np.ndarray) -> np.ndarray:
        """
        Calculate turbulence indicator from velocity field.

        Uses velocity gradient magnitude as indicator.
        """
        ndim = len(self.grid_shape)
        turbulence = np.zeros(self.grid_shape)

        for dim in range(ndim):
            for component in range(ndim):
                # Calculate gradient
                gradient = np.gradient(velocity_field[..., component], axis=dim)
                turbulence += gradient ** 2

        turbulence = np.sqrt(turbulence)

        return turbulence

    def get_concentration(self) -> np.ndarray:
        """Get current concentration field."""
        return self.concentration.copy()
