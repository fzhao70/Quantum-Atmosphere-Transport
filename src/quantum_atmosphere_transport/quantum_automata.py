"""
Quantum Cellular Automata and Quantum Annealing

Implements quantum automata rules and optimization-based transport methods.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from .base_transport import BaseTransport


class QuantumCellularAutomataTransport(BaseTransport):
    """
    Quantum Cellular Automata (QCA) Transport.

    Uses quantum lattice gas automaton rules for transport simulation.
    Each cell evolves according to unitary quantum rules.
    """

    def __init__(self, config, rule_type: str = 'partitioned'):
        """
        Initialize QCA transport.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        rule_type : str
            Type of QCA rule: 'partitioned', 'one_dimensional', 'block'
        """
        super().__init__(config)
        self.rule_type = rule_type
        self.quantum_state: Optional[np.ndarray] = None

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize QCA quantum state."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()

        # Each cell has multiple quantum states (for different velocities)
        ndim = len(grid_shape)
        n_velocity_states = 2 * ndim + 1  # Rest + 2*dim directions

        # Initialize quantum state
        self.quantum_state = np.zeros(grid_shape + (n_velocity_states,), dtype=complex)

        # Distribute initial concentration among velocity states
        for i in range(n_velocity_states):
            self.quantum_state[..., i] = np.sqrt(
                initial_concentration / n_velocity_states
            )

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance using QCA rules."""
        n_steps = max(1, int(dt * 10))

        for _ in range(n_steps):
            # 1. Collision step (local unitary evolution)
            self.quantum_state = self._collision_step(self.quantum_state)

            # 2. Streaming step (propagation)
            self.quantum_state = self._streaming_step(self.quantum_state)

        # Measure total concentration
        self.concentration = np.sum(np.abs(self.quantum_state) ** 2, axis=-1)

        # Apply source term
        if source_term is not None:
            self.concentration += source_term * dt

        return self.concentration.copy()

    def _collision_step(self, state: np.ndarray) -> np.ndarray:
        """
        Apply collision operator (local unitary rotation).

        Mixes velocity states at each spatial location.
        """
        new_state = state.copy()
        n_velocity = state.shape[-1]

        # Apply unitary collision matrix at each point
        collision_matrix = self._get_collision_matrix(n_velocity)

        # Apply to each spatial location
        for idx in np.ndindex(state.shape[:-1]):
            velocity_vector = state[idx]
            new_state[idx] = collision_matrix @ velocity_vector

        return new_state

    def _get_collision_matrix(self, n_velocity: int) -> np.ndarray:
        """
        Get quantum collision matrix.

        Uses quantum Fourier transform as collision operator.
        """
        if self.rule_type == 'partitioned':
            # Partitioned QCA: block diagonal structure
            U = self._quantum_fourier_matrix(n_velocity)

        elif self.rule_type == 'one_dimensional':
            # 1D QCA: simpler structure
            U = self._hadamard_like_matrix(n_velocity)

        elif self.rule_type == 'block':
            # Block QCA: structured collision
            U = self._block_collision_matrix(n_velocity)

        else:
            # Default: Quantum Fourier Transform
            U = self._quantum_fourier_matrix(n_velocity)

        return U

    def _quantum_fourier_matrix(self, n: int) -> np.ndarray:
        """Create quantum Fourier transform matrix."""
        omega = np.exp(2j * np.pi / n)
        F = np.array([[omega ** (i * j) for j in range(n)] for i in range(n)])
        return F / np.sqrt(n)

    def _hadamard_like_matrix(self, n: int) -> np.ndarray:
        """Create Hadamard-like matrix."""
        H = np.ones((n, n), dtype=complex) / np.sqrt(n)
        # Add phase variations
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 1:
                    H[i, j] *= -1
        return H

    def _block_collision_matrix(self, n: int) -> np.ndarray:
        """Create block-structured collision matrix."""
        U = np.eye(n, dtype=complex)

        # Create small blocks
        block_size = min(2, n)

        for i in range(0, n - 1, block_size):
            if i + 1 < n:
                # 2x2 rotation in this block
                theta = np.pi / 4
                U[i:i + 2, i:i + 2] = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])

        return U

    def _streaming_step(self, state: np.ndarray) -> np.ndarray:
        """
        Apply streaming operator (propagation).

        Each velocity state propagates in its corresponding direction.
        """
        ndim = len(self.grid_shape)
        new_state = np.zeros_like(state)

        # Velocity state 0: rest (no movement)
        new_state[..., 0] = state[..., 0]

        # Other velocity states: propagate
        for dim in range(ndim):
            # Positive direction
            vel_idx = 1 + dim * 2
            new_state[..., vel_idx] = np.roll(state[..., vel_idx], -1, axis=dim)

            # Negative direction
            vel_idx = 1 + dim * 2 + 1
            new_state[..., vel_idx] = np.roll(state[..., vel_idx], 1, axis=dim)

        return new_state

    def get_concentration(self) -> np.ndarray:
        """Get current concentration."""
        return self.concentration.copy()

    def get_velocity_distributions(self) -> np.ndarray:
        """
        Get velocity-resolved distributions.

        Returns
        -------
        np.ndarray
            Probability distribution for each velocity state
        """
        return np.abs(self.quantum_state) ** 2


class QuantumAnnealingTransport(BaseTransport):
    """
    Quantum Annealing Transport.

    Uses quantum annealing to optimize transport pathways.
    Minimizes energy functional representing transport equation.
    """

    def __init__(self, config, n_anneal_steps: int = 50):
        """
        Initialize quantum annealing transport.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        n_anneal_steps : int
            Number of annealing steps
        """
        super().__init__(config)
        self.n_anneal_steps = n_anneal_steps
        self.quantum_state: Optional[np.ndarray] = None
        self.temperature_schedule: np.ndarray = np.linspace(1.0, 0.01, n_anneal_steps)

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize quantum annealing."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()
        self.quantum_state = np.sqrt(initial_concentration).astype(complex)

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance using quantum annealing."""
        # Define energy functional
        def energy_functional(state):
            return self._transport_energy(
                state, velocity_field, diffusion_coeff, dt
            )

        # Quantum annealing process
        for step, temperature in enumerate(self.temperature_schedule):
            # Quantum tunneling with decreasing strength
            tunneling_strength = temperature

            # 1. Apply transverse field (quantum fluctuations)
            self.quantum_state = self._apply_transverse_field(
                self.quantum_state, tunneling_strength
            )

            # 2. Apply problem Hamiltonian (energy minimization)
            self.quantum_state = self._apply_problem_hamiltonian(
                self.quantum_state, energy_functional, 1.0 - temperature
            )

            # 3. Thermal fluctuations
            if temperature > 0.01:
                self.quantum_state = self._add_thermal_noise(
                    self.quantum_state, temperature * 0.1
                )

        # Final measurement
        self.concentration = np.abs(self.quantum_state) ** 2

        # Apply source term
        if source_term is not None:
            self.concentration += source_term * dt

        return self.concentration.copy()

    def _transport_energy(self,
                           state: np.ndarray,
                           velocity_field: np.ndarray,
                           diffusion_coeff: float,
                           dt: float) -> float:
        """
        Calculate transport energy functional.

        Lower energy = better transport solution.
        """
        concentration = np.abs(state) ** 2

        # Gradient energy (penalizes sharp gradients)
        gradient_energy = 0.0
        ndim = len(self.grid_shape)

        for dim in range(ndim):
            grad = np.gradient(concentration, axis=dim)
            gradient_energy += np.sum(grad ** 2)

        # Diffusion energy
        diffusion_energy = diffusion_coeff * gradient_energy

        # Advection energy (misalignment with velocity field)
        advection_energy = 0.0
        for dim in range(ndim):
            if velocity_field.shape[-1] > dim:
                flow = velocity_field[..., dim] * concentration
                advection_energy += np.sum(flow ** 2)

        # Total energy
        total_energy = diffusion_energy + advection_energy * 0.1

        return float(total_energy)

    def _apply_transverse_field(self,
                                 state: np.ndarray,
                                 strength: float) -> np.ndarray:
        """
        Apply transverse field (quantum tunneling).

        Creates quantum superposition across different configurations.
        """
        # Transverse field: mix current state with neighboring states
        new_state = state.copy()
        ndim = len(self.grid_shape)

        for dim in range(ndim):
            neighbor_plus = np.roll(state, 1, axis=dim)
            neighbor_minus = np.roll(state, -1, axis=dim)

            # Quantum tunneling to neighbors
            new_state += strength * 0.1 * (neighbor_plus + neighbor_minus) * 1j

        # Normalize
        new_state /= (np.linalg.norm(new_state) + 1e-10)

        return new_state

    def _apply_problem_hamiltonian(self,
                                    state: np.ndarray,
                                    energy_func: Callable,
                                    strength: float) -> np.ndarray:
        """
        Apply problem Hamiltonian (guides toward solution).

        Uses energy gradient to evolve state.
        """
        # Calculate energy gradient (simplified)
        current_energy = energy_func(state)

        # Small perturbations to estimate gradient
        epsilon = 0.01
        gradient = np.zeros_like(state)

        for idx in np.ndindex(state.shape):
            perturbed_state = state.copy()
            perturbed_state[idx] += epsilon
            perturbed_energy = energy_func(perturbed_state)

            gradient[idx] = (perturbed_energy - current_energy) / epsilon

        # Move opposite to gradient
        new_state = state - strength * 0.01 * gradient

        # Normalize
        new_state /= (np.linalg.norm(new_state) + 1e-10)

        return new_state

    def _add_thermal_noise(self, state: np.ndarray, temperature: float) -> np.ndarray:
        """Add thermal fluctuations."""
        noise = np.random.randn(*state.shape) * temperature
        new_state = state + noise.astype(complex) * 1j

        # Normalize
        new_state /= (np.linalg.norm(new_state) + 1e-10)

        return new_state

    def get_concentration(self) -> np.ndarray:
        """Get current concentration."""
        return self.concentration.copy()

    def get_annealing_schedule(self) -> np.ndarray:
        """
        Get the temperature schedule used in annealing.

        Returns
        -------
        np.ndarray
            Temperature values over annealing steps
        """
        return self.temperature_schedule.copy()
