"""
Quantum Advection Module

Implements quantum-inspired advection scheme for atmospheric tracer transport.
Uses quantum phase shifting and unitary transport operators.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import map_coordinates


class QuantumAdvection:
    """
    Quantum-inspired advection operator.

    Uses quantum phase shifts to represent transport in the momentum space.
    The advection is implemented as a unitary operator that preserves
    quantum coherence during transport.
    """

    def __init__(self, config):
        """
        Initialize quantum advection operator.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        """
        self.config = config
        self.grid_shape: Optional[Tuple[int, ...]] = None
        self.grid_spacing: float = 1.0

    def initialize(self, grid_shape: Tuple[int, ...]) -> None:
        """
        Initialize the quantum advection operator.

        Parameters
        ----------
        grid_shape : tuple
            Shape of the computational grid (nx, ny, nz)
        """
        self.grid_shape = grid_shape
        self.grid_spacing = self.config.grid_spacing

    def apply(self,
              quantum_state: np.ndarray,
              velocity_field: np.ndarray,
              dt: float) -> np.ndarray:
        """
        Apply quantum advection operator to the quantum state.

        Uses a quantum-inspired semi-Lagrangian scheme where particles
        are transported backwards in time while maintaining quantum coherence.

        Parameters
        ----------
        quantum_state : np.ndarray
            Current quantum state
        velocity_field : np.ndarray
            Wind velocity field (last dimension contains velocity components)
        dt : float
            Time step

        Returns
        -------
        np.ndarray
            Updated quantum state after advection
        """
        if self.grid_shape is None:
            raise RuntimeError("Quantum advection not initialized. Call initialize() first.")

        if self.config.advection_scheme == 'quantum_semi_lagrangian':
            return self._quantum_semi_lagrangian(quantum_state, velocity_field, dt)
        elif self.config.advection_scheme == 'quantum_spectral':
            return self._quantum_spectral(quantum_state, velocity_field, dt)
        elif self.config.advection_scheme == 'quantum_upwind':
            return self._quantum_upwind(quantum_state, velocity_field, dt)
        else:
            raise ValueError(f"Unknown advection scheme: {self.config.advection_scheme}")

    def _quantum_semi_lagrangian(self,
                                 quantum_state: np.ndarray,
                                 velocity_field: np.ndarray,
                                 dt: float) -> np.ndarray:
        """
        Quantum semi-Lagrangian advection scheme.

        Traces characteristics backward in time and interpolates quantum
        amplitudes while preserving phase information.

        Parameters
        ----------
        quantum_state : np.ndarray
            Current quantum state
        velocity_field : np.ndarray
            Velocity field
        dt : float
            Time step

        Returns
        -------
        np.ndarray
            Advected quantum state
        """
        ndim = len(self.grid_shape)

        # Create coordinate grids
        if ndim == 1:
            coords = [np.arange(self.grid_shape[0])]
        elif ndim == 2:
            coords = np.meshgrid(
                np.arange(self.grid_shape[0]),
                np.arange(self.grid_shape[1]),
                indexing='ij'
            )
        elif ndim == 3:
            coords = np.meshgrid(
                np.arange(self.grid_shape[0]),
                np.arange(self.grid_shape[1]),
                np.arange(self.grid_shape[2]),
                indexing='ij'
            )
        else:
            raise ValueError("Only 1D, 2D, and 3D supported")

        # Calculate departure points (backward trajectories)
        departure_coords = []
        for i in range(ndim):
            # Backward trajectory: x_d = x - u * dt
            departure = coords[i] - velocity_field[..., i] * dt / self.grid_spacing

            # Apply boundary conditions
            if self.config.boundary_condition == 'periodic':
                departure = departure % self.grid_shape[i]
            else:
                departure = np.clip(departure, 0, self.grid_shape[i] - 1)

            departure_coords.append(departure)

        # Interpolate both real and imaginary parts separately
        # This preserves quantum phase information
        real_part = map_coordinates(
            quantum_state.real,
            departure_coords,
            order=self.config.interpolation_order,
            mode='wrap' if self.config.boundary_condition == 'periodic' else 'nearest'
        )

        imag_part = map_coordinates(
            quantum_state.imag,
            departure_coords,
            order=self.config.interpolation_order,
            mode='wrap' if self.config.boundary_condition == 'periodic' else 'nearest'
        )

        advected_state = real_part + 1j * imag_part

        # Apply quantum phase correction to maintain unitarity
        if self.config.apply_phase_correction:
            advected_state = self._apply_phase_correction(
                advected_state, velocity_field, dt
            )

        return advected_state

    def _quantum_spectral(self,
                         quantum_state: np.ndarray,
                         velocity_field: np.ndarray,
                         dt: float) -> np.ndarray:
        """
        Quantum spectral advection using FFT.

        Uses quantum Fourier transform to perform advection in spectral space,
        which naturally preserves quantum properties.

        Parameters
        ----------
        quantum_state : np.ndarray
            Current quantum state
        velocity_field : np.ndarray
            Velocity field
        dt : float
            Time step

        Returns
        -------
        np.ndarray
            Advected quantum state
        """
        ndim = len(self.grid_shape)

        # Transform to spectral space (quantum Fourier transform)
        quantum_spectral = np.fft.fftn(quantum_state)

        # Get wave numbers
        wave_numbers = []
        for i in range(ndim):
            k = np.fft.fftfreq(self.grid_shape[i], d=self.grid_spacing)
            wave_numbers.append(k)

        # Create meshgrid of wave numbers
        if ndim == 1:
            k_grid = [wave_numbers[0]]
        else:
            k_grid = np.meshgrid(*wave_numbers, indexing='ij')

        # Calculate average velocity for each mode
        velocity_spectral = np.fft.fftn(velocity_field, axes=tuple(range(ndim)))

        # Apply quantum phase shift in spectral space
        # exp(i k · u dt) represents translation operator
        phase_shift = np.zeros(self.grid_shape, dtype=complex)
        for i in range(ndim):
            # Get velocity component in spectral space
            u_component = np.fft.ifftn(velocity_spectral[..., i]).real
            u_avg = np.mean(u_component)

            # Add phase shift: i k_i u_i dt
            phase_shift += 1j * k_grid[i] * u_avg * dt

        # Apply phase shift operator
        quantum_spectral *= np.exp(phase_shift)

        # Transform back to physical space
        advected_state = np.fft.ifftn(quantum_spectral)

        return advected_state

    def _quantum_upwind(self,
                       quantum_state: np.ndarray,
                       velocity_field: np.ndarray,
                       dt: float) -> np.ndarray:
        """
        Quantum upwind scheme.

        Uses directional finite differences with quantum phase preservation.

        Parameters
        ----------
        quantum_state : np.ndarray
            Current quantum state
        velocity_field : np.ndarray
            Velocity field
        dt : float
            Time step

        Returns
        -------
        np.ndarray
            Advected quantum state
        """
        ndim = len(self.grid_shape)
        advected = quantum_state.copy()

        # Apply upwind scheme for each dimension
        for dim in range(ndim):
            u = velocity_field[..., dim]
            dx = self.grid_spacing
            cfl = u * dt / dx

            # Roll operations for finite differences
            forward_roll = np.roll(advected, -1, axis=dim)
            backward_roll = np.roll(advected, 1, axis=dim)

            # Upwind: use backward difference when u > 0, forward when u < 0
            positive_u = (u >= 0)
            negative_u = (u < 0)

            # Expand dimensions for broadcasting
            shape = [1] * ndim
            shape[dim] = self.grid_shape[dim]

            # Compute gradient with upwind direction
            gradient = np.where(
                positive_u,
                advected - backward_roll,
                forward_roll - advected
            )

            # Update state
            advected = advected - cfl * gradient

        return advected

    def _apply_phase_correction(self,
                                quantum_state: np.ndarray,
                                velocity_field: np.ndarray,
                                dt: float) -> np.ndarray:
        """
        Apply quantum phase correction to maintain unitarity.

        Adds geometric phase (Berry phase) acquired during transport.

        Parameters
        ----------
        quantum_state : np.ndarray
            Quantum state after advection
        velocity_field : np.ndarray
            Velocity field
        dt : float
            Time step

        Returns
        -------
        np.ndarray
            Phase-corrected quantum state
        """
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(
            np.sum(velocity_field ** 2, axis=-1)
        )

        # Geometric phase acquired: φ = v²t / (2ℏ)
        # Using natural units where ℏ = 1
        geometric_phase = velocity_magnitude ** 2 * dt / 2

        # Apply phase rotation
        phase_factor = np.exp(1j * geometric_phase)
        corrected_state = quantum_state * phase_factor

        return corrected_state

    def get_cfl_number(self, velocity_field: np.ndarray, dt: float) -> float:
        """
        Calculate the CFL (Courant-Friedrichs-Lewy) number.

        CFL = |u| * dt / dx

        For stability, CFL should typically be < 1.

        Parameters
        ----------
        velocity_field : np.ndarray
            Velocity field
        dt : float
            Time step

        Returns
        -------
        float
            Maximum CFL number
        """
        velocity_magnitude = np.sqrt(
            np.sum(velocity_field ** 2, axis=-1)
        )
        max_velocity = np.max(velocity_magnitude)
        cfl = max_velocity * dt / self.grid_spacing

        return float(cfl)
