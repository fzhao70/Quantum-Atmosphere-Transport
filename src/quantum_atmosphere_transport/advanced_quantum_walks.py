"""
Advanced Quantum Walk Methods

Implements various quantum walk algorithms for atmospheric transport.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from .base_transport import BaseTransport


class DiscreteTimeQuantumWalkTransport(BaseTransport):
    """
    Discrete-Time Quantum Walk (DTQW) Transport.

    Uses coin operator and shift operator for discrete-time evolution.
    Different from continuous-time quantum walk - has additional coin space.
    """

    def __init__(self, config, coin_type: str = 'hadamard'):
        """
        Initialize discrete-time quantum walk transport.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        coin_type : str
            Type of coin operator: 'hadamard', 'grover', 'fourier'
        """
        super().__init__(config)
        self.coin_type = coin_type
        self.quantum_state: Optional[np.ndarray] = None  # Includes coin space
        self.coin_dim = 4  # For 2D: 4 directions

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize DTQW with coin space."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()

        # Create quantum state with coin degree of freedom
        # Shape: (coin_dim, *grid_shape)
        ndim = len(grid_shape)
        self.coin_dim = 2 ** ndim  # 2^d directions in d dimensions

        # Initialize quantum state
        quantum_shape = (self.coin_dim,) + grid_shape
        self.quantum_state = np.zeros(quantum_shape, dtype=complex)

        # Distribute initial concentration across coin space
        for i in range(self.coin_dim):
            self.quantum_state[i] = np.sqrt(initial_concentration / self.coin_dim)

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance using discrete-time quantum walk."""
        # Number of DTQW steps based on time
        n_steps = max(1, int(dt * 10))  # 10 steps per time unit

        for _ in range(n_steps):
            # 1. Apply coin operator
            self.quantum_state = self._apply_coin_operator(self.quantum_state)

            # 2. Apply shift operator (with velocity modulation)
            self.quantum_state = self._apply_shift_operator(
                self.quantum_state, velocity_field, dt / n_steps
            )

            # 3. Apply diffusion (decoherence)
            if diffusion_coeff > 0:
                self.quantum_state = self._apply_quantum_diffusion(
                    self.quantum_state, diffusion_coeff, dt / n_steps
                )

        # Measure to get classical concentration
        self.concentration = self._measure_quantum_state()

        # Apply source term
        if source_term is not None:
            self.concentration += source_term * dt

        return self.concentration.copy()

    def _apply_coin_operator(self, state: np.ndarray) -> np.ndarray:
        """
        Apply coin operator to quantum state.

        The coin operator mixes the coin degrees of freedom.
        """
        if self.coin_type == 'hadamard':
            coin = self._hadamard_coin()
        elif self.coin_type == 'grover':
            coin = self._grover_coin()
        elif self.coin_type == 'fourier':
            coin = self._fourier_coin()
        else:
            coin = self._hadamard_coin()

        # Apply coin to each position
        new_state = np.zeros_like(state)

        # Reshape for matrix multiplication
        for idx in np.ndindex(state.shape[1:]):
            coin_state = state[(slice(None),) + idx]
            new_state[(slice(None),) + idx] = coin @ coin_state

        return new_state

    def _hadamard_coin(self) -> np.ndarray:
        """Create Hadamard coin operator."""
        if self.coin_dim == 2:
            # 1D Hadamard
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif self.coin_dim == 4:
            # 2D Grover coin (generalized Hadamard)
            H = np.ones((4, 4)) / 2
            H -= np.eye(4) / 2
            return H
        else:
            # General Hadamard
            return np.ones((self.coin_dim, self.coin_dim)) / np.sqrt(self.coin_dim)

    def _grover_coin(self) -> np.ndarray:
        """Create Grover coin operator."""
        G = 2 * np.ones((self.coin_dim, self.coin_dim)) / self.coin_dim
        G -= np.eye(self.coin_dim)
        return G

    def _fourier_coin(self) -> np.ndarray:
        """Create Quantum Fourier Transform coin."""
        n = self.coin_dim
        omega = np.exp(2j * np.pi / n)
        F = np.array([[omega ** (i * j) for j in range(n)] for i in range(n)])
        return F / np.sqrt(n)

    def _apply_shift_operator(self,
                               state: np.ndarray,
                               velocity_field: np.ndarray,
                               dt: float) -> np.ndarray:
        """
        Apply shift operator - moves walker based on coin state.

        Each coin state corresponds to a direction.
        """
        ndim = len(self.grid_shape)
        new_state = np.zeros_like(state)

        if ndim == 1:
            # Coin 0: left, Coin 1: right
            new_state[0] = np.roll(state[0], 1, axis=0)
            new_state[1] = np.roll(state[1], -1, axis=0)

        elif ndim == 2:
            # Coin 0: left, 1: right, 2: down, 3: up
            new_state[0] = np.roll(state[0], 1, axis=0)   # left
            new_state[1] = np.roll(state[1], -1, axis=0)  # right
            new_state[2] = np.roll(state[2], 1, axis=1)   # down
            new_state[3] = np.roll(state[3], -1, axis=1)  # up

        return new_state

    def _apply_quantum_diffusion(self,
                                  state: np.ndarray,
                                  diffusion_coeff: float,
                                  dt: float) -> np.ndarray:
        """Apply diffusion as decoherence."""
        # Simple decoherence model
        decoherence = np.exp(-diffusion_coeff * dt)
        return state * decoherence

    def _measure_quantum_state(self) -> np.ndarray:
        """Measure quantum state to get classical concentration."""
        # Sum over coin space and take absolute value squared
        concentration = np.sum(np.abs(self.quantum_state) ** 2, axis=0)
        return concentration

    def get_concentration(self) -> np.ndarray:
        """Get current concentration."""
        return self.concentration.copy()


class StaggeredQuantumWalkTransport(BaseTransport):
    """
    Staggered Quantum Walk Transport.

    Uses tessellation cover and polygon operators.
    More general than standard quantum walks.
    """

    def __init__(self, config):
        """Initialize staggered quantum walk."""
        super().__init__(config)
        self.quantum_state: Optional[np.ndarray] = None

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize staggered quantum walk."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()

        # Create quantum state
        self.quantum_state = np.sqrt(initial_concentration).astype(complex)

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance using staggered quantum walk."""
        n_steps = max(1, int(dt * 10))

        for _ in range(n_steps):
            # Apply tessellation operators alternately
            self.quantum_state = self._apply_tessellation_a(self.quantum_state)
            self.quantum_state = self._apply_tessellation_b(self.quantum_state)

        # Measure
        self.concentration = np.abs(self.quantum_state) ** 2

        # Apply source
        if source_term is not None:
            self.concentration += source_term * dt

        return self.concentration.copy()

    def _apply_tessellation_a(self, state: np.ndarray) -> np.ndarray:
        """Apply first tessellation cover."""
        # Horizontal edges
        new_state = state.copy()
        ndim = len(self.grid_shape)

        if ndim >= 1:
            # Reflect along first dimension
            new_state = -1j * new_state + 1j * np.roll(state, 1, axis=0)
            new_state /= np.sqrt(2)

        return new_state

    def _apply_tessellation_b(self, state: np.ndarray) -> np.ndarray:
        """Apply second tessellation cover."""
        # Vertical edges
        new_state = state.copy()
        ndim = len(self.grid_shape)

        if ndim >= 2:
            # Reflect along second dimension
            new_state = -1j * new_state + 1j * np.roll(state, 1, axis=1)
            new_state /= np.sqrt(2)

        return new_state

    def get_concentration(self) -> np.ndarray:
        """Get current concentration."""
        return self.concentration.copy()


class QuantumAmplificationTransport(BaseTransport):
    """
    Quantum Amplitude Amplification Transport.

    Uses Grover-like amplitude amplification to enhance
    probability at target regions.
    """

    def __init__(self, config, target_amplification: float = 1.5):
        """
        Initialize quantum amplitude amplification transport.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        target_amplification : float
            Amplification factor for target regions
        """
        super().__init__(config)
        self.target_amplification = target_amplification
        self.quantum_state: Optional[np.ndarray] = None

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize quantum state."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()
        self.quantum_state = np.sqrt(initial_concentration).astype(complex)

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance with amplitude amplification."""
        # Standard transport
        self.quantum_state = self._transport_step(
            self.quantum_state, velocity_field, diffusion_coeff, dt
        )

        # Amplitude amplification
        self.quantum_state = self._amplify_amplitude(self.quantum_state)

        # Measure
        self.concentration = np.abs(self.quantum_state) ** 2

        # Normalize to conserve mass
        total_mass = np.sum(self.concentration)
        if total_mass > 0:
            self.concentration *= np.sum(initial_concentration) / total_mass

        # Apply source
        if source_term is not None:
            self.concentration += source_term * dt

        return self.concentration.copy()

    def _transport_step(self,
                        state: np.ndarray,
                        velocity_field: np.ndarray,
                        diffusion_coeff: float,
                        dt: float) -> np.ndarray:
        """Basic transport step."""
        # Simple diffusion using quantum walk
        ndim = len(self.grid_shape)

        for dim in range(ndim):
            state_forward = np.roll(state, -1, axis=dim)
            state_backward = np.roll(state, 1, axis=dim)

            # Quantum superposition of forward and backward
            state = (state + state_forward * np.exp(-1j * dt) +
                    state_backward * np.exp(1j * dt)) / np.sqrt(3)

        return state

    def _amplify_amplitude(self, state: np.ndarray) -> np.ndarray:
        """
        Apply Grover-like amplitude amplification.

        Amplifies high-amplitude regions.
        """
        # Calculate mean amplitude
        mean_amplitude = np.mean(np.abs(state))

        # Identify target states (above-average amplitude)
        target_mask = np.abs(state) > mean_amplitude

        # Oracle: mark target states
        marked_state = state.copy()
        marked_state[target_mask] *= -1

        # Diffusion operator: inversion about average
        avg_state = np.mean(marked_state)
        amplified_state = 2 * avg_state - marked_state

        # Blend with original for stability
        blend = 0.1
        return (1 - blend) * state + blend * amplified_state

    def get_concentration(self) -> np.ndarray:
        """Get current concentration."""
        return self.concentration.copy()
