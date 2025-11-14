"""
Quantum Tracer Transport Module

Main module for quantum algorithm-based atmospheric tracer transport.
Uses quantum walks and quantum superposition principles for transport modeling.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from .quantum_diffusion import QuantumDiffusion
from .quantum_advection import QuantumAdvection
from .config import TransportConfig


class QuantumTracerTransport:
    """
    Quantum-based atmospheric tracer transport scheme.

    This scheme uses quantum algorithms to solve the advection-diffusion equation:
    ∂C/∂t + u·∇C = ∇·(K∇C) + S

    Where:
    - C is tracer concentration
    - u is velocity field (wind)
    - K is diffusion coefficient
    - S is source/sink term

    The quantum approach uses:
    1. Quantum walks for diffusion modeling
    2. Quantum superposition for multiple scenarios
    3. Quantum-inspired optimization for numerical stability
    """

    def __init__(self, config: Optional[TransportConfig] = None):
        """
        Initialize the quantum tracer transport scheme.

        Parameters
        ----------
        config : TransportConfig, optional
            Configuration object for transport scheme
        """
        self.config = config or TransportConfig()
        self.quantum_diffusion = QuantumDiffusion(self.config)
        self.quantum_advection = QuantumAdvection(self.config)

        # Quantum state tracking
        self.quantum_state: Optional[np.ndarray] = None
        self.classical_state: Optional[np.ndarray] = None

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """
        Initialize the transport scheme with initial conditions.

        Parameters
        ----------
        initial_concentration : np.ndarray
            Initial tracer concentration field
        grid_shape : tuple
            Shape of the computational grid (nx, ny, nz)
        """
        self.grid_shape = grid_shape
        self.classical_state = initial_concentration.copy()

        # Convert to quantum state using superposition
        self.quantum_state = self._classical_to_quantum(initial_concentration)

        # Initialize sub-modules
        self.quantum_diffusion.initialize(grid_shape)
        self.quantum_advection.initialize(grid_shape)

    def _classical_to_quantum(self, classical_field: np.ndarray) -> np.ndarray:
        """
        Convert classical concentration field to quantum state representation.

        Uses quantum superposition to represent the field as a quantum state
        where each grid point has amplitude and phase information.

        Parameters
        ----------
        classical_field : np.ndarray
            Classical concentration field

        Returns
        -------
        np.ndarray
            Quantum state representation (complex-valued)
        """
        # Normalize the field to create probability amplitudes
        normalized = classical_field / (np.sum(classical_field) + 1e-10)

        # Create quantum amplitudes (complex numbers)
        # Real part: normalized concentration
        # Imaginary part: quantum phase (initially zero)
        quantum_state = np.sqrt(normalized).astype(complex)

        # Add small quantum phase for numerical stability
        quantum_state *= np.exp(1j * self.config.quantum_phase_factor)

        return quantum_state

    def _quantum_to_classical(self, quantum_state: np.ndarray) -> np.ndarray:
        """
        Convert quantum state back to classical concentration field.

        Performs quantum measurement to collapse the superposition state
        into a classical concentration field.

        Parameters
        ----------
        quantum_state : np.ndarray
            Quantum state representation

        Returns
        -------
        np.ndarray
            Classical concentration field
        """
        # Quantum measurement: |ψ|² gives probability (concentration)
        classical_field = np.abs(quantum_state) ** 2

        # Renormalize to conserve mass
        total_mass = np.sum(self.classical_state)
        classical_field *= total_mass / (np.sum(classical_field) + 1e-10)

        return classical_field

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance the transport scheme by one time step using quantum algorithms.

        Parameters
        ----------
        dt : float
            Time step size
        velocity_field : np.ndarray
            Wind velocity field (u, v, w components)
        diffusion_coeff : float
            Diffusion coefficient K
        source_term : np.ndarray, optional
            Source/sink term S

        Returns
        -------
        np.ndarray
            Updated tracer concentration field
        """
        # Split-operator approach (Strang splitting for 2nd order accuracy)
        # 1. Half-step advection
        self.quantum_state = self.quantum_advection.apply(
            self.quantum_state, velocity_field, dt / 2
        )

        # 2. Full-step diffusion (quantum walk)
        self.quantum_state = self.quantum_diffusion.apply(
            self.quantum_state, diffusion_coeff, dt
        )

        # 3. Half-step advection
        self.quantum_state = self.quantum_advection.apply(
            self.quantum_state, velocity_field, dt / 2
        )

        # 4. Apply source/sink term if provided
        if source_term is not None:
            self.quantum_state += self._classical_to_quantum(source_term * dt)

        # 5. Quantum measurement to get classical field
        self.classical_state = self._quantum_to_classical(self.quantum_state)

        # 6. Apply quantum decoherence to maintain stability
        if self.config.apply_decoherence:
            self._apply_decoherence()

        return self.classical_state.copy()

    def _apply_decoherence(self) -> None:
        """
        Apply quantum decoherence to prevent numerical instabilities.

        Quantum decoherence gradually collapses quantum superposition states,
        which helps maintain numerical stability in the transport scheme.
        """
        decoherence_rate = self.config.decoherence_rate

        # Mix quantum and classical states
        classical_component = self._classical_to_quantum(self.classical_state)
        self.quantum_state = (
            (1 - decoherence_rate) * self.quantum_state +
            decoherence_rate * classical_component
        )

    def get_quantum_statistics(self) -> Dict[str, float]:
        """
        Calculate quantum statistics of the current state.

        Returns
        -------
        dict
            Dictionary containing quantum metrics:
            - coherence: Quantum coherence measure
            - entanglement: Spatial entanglement measure
            - phase_variance: Variance of quantum phases
        """
        if self.quantum_state is None:
            return {}

        # Calculate quantum coherence (off-diagonal density matrix elements)
        density_matrix = np.outer(self.quantum_state.flatten(),
                                 np.conj(self.quantum_state.flatten()))
        coherence = np.sum(np.abs(density_matrix)) - np.trace(np.abs(density_matrix))
        coherence /= density_matrix.size

        # Calculate phase variance
        phases = np.angle(self.quantum_state.flatten())
        phase_variance = np.var(phases)

        # Calculate spatial entanglement (von Neumann entropy)
        probs = np.abs(self.quantum_state.flatten()) ** 2
        probs = probs[probs > 1e-10]  # Remove zeros
        entanglement = -np.sum(probs * np.log(probs))

        return {
            "coherence": float(coherence),
            "entanglement": float(entanglement),
            "phase_variance": float(phase_variance),
        }

    def get_concentration(self) -> np.ndarray:
        """
        Get the current tracer concentration field.

        Returns
        -------
        np.ndarray
            Current concentration field
        """
        return self.classical_state.copy()
