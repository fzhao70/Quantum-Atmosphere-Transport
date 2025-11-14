"""
Classical Transport Methods

Implements traditional Eulerian and semi-Lagrangian transport schemes
for comparison with quantum methods.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import map_coordinates
from .base_transport import BaseTransport


class ClassicalEulerianTransport(BaseTransport):
    """
    Classical Eulerian transport scheme using finite differences.

    Uses explicit time stepping with upwind advection and
    centered differences for diffusion.
    """

    def __init__(self, config):
        """Initialize classical Eulerian transport."""
        super().__init__(config)

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize with initial concentration field."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance by one time step using Eulerian scheme."""
        # Split operator: advection + diffusion
        # 1. Advection step
        self.concentration = self._advect_upwind(
            self.concentration, velocity_field, dt
        )

        # 2. Diffusion step
        self.concentration = self._diffuse_centered(
            self.concentration, diffusion_coeff, dt
        )

        # 3. Apply source term
        if source_term is not None:
            self.concentration += source_term * dt

        # 4. Ensure non-negativity
        self.concentration = np.maximum(self.concentration, 0.0)

        return self.concentration.copy()

    def _advect_upwind(self,
                       concentration: np.ndarray,
                       velocity_field: np.ndarray,
                       dt: float) -> np.ndarray:
        """
        Upwind advection scheme.

        Uses first-order upwind differences for stability.
        """
        ndim = len(self.grid_shape)
        c = concentration.copy()
        dx = self.config.grid_spacing

        for dim in range(ndim):
            u = velocity_field[..., dim]
            cfl = u * dt / dx

            # Forward and backward differences
            c_forward = np.roll(c, -1, axis=dim)
            c_backward = np.roll(c, 1, axis=dim)

            # Upwind scheme
            positive_u = (u >= 0)
            dc = np.where(
                positive_u,
                c - c_backward,  # Backward difference when u > 0
                c_forward - c    # Forward difference when u < 0
            )

            c = c - cfl * dc

        return c

    def _diffuse_centered(self,
                          concentration: np.ndarray,
                          diffusion_coeff: float,
                          dt: float) -> np.ndarray:
        """
        Centered difference diffusion.

        Uses second-order centered differences for the Laplacian.
        """
        ndim = len(self.grid_shape)
        c = concentration.copy()
        dx = self.config.grid_spacing
        alpha = diffusion_coeff * dt / (dx ** 2)

        laplacian = np.zeros_like(c)

        for dim in range(ndim):
            c_forward = np.roll(c, -1, axis=dim)
            c_backward = np.roll(c, 1, axis=dim)

            # Second derivative: (c[i+1] - 2*c[i] + c[i-1]) / dx^2
            laplacian += c_forward - 2 * c + c_backward

        c = c + alpha * laplacian

        return c

    def get_concentration(self) -> np.ndarray:
        """Get current concentration field."""
        return self.concentration.copy()


class ClassicalSemiLagrangianTransport(BaseTransport):
    """
    Classical semi-Lagrangian transport scheme.

    Traces characteristics backward in time and interpolates
    concentration values. Second-order accurate and unconditionally stable.
    """

    def __init__(self, config):
        """Initialize classical semi-Lagrangian transport."""
        super().__init__(config)

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize with initial concentration field."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance by one time step using semi-Lagrangian scheme."""
        # 1. Advection via semi-Lagrangian
        self.concentration = self._semi_lagrangian_advect(
            self.concentration, velocity_field, dt
        )

        # 2. Diffusion via implicit scheme
        self.concentration = self._implicit_diffusion(
            self.concentration, diffusion_coeff, dt
        )

        # 3. Apply source term
        if source_term is not None:
            self.concentration += source_term * dt

        # 4. Ensure non-negativity
        self.concentration = np.maximum(self.concentration, 0.0)

        return self.concentration.copy()

    def _semi_lagrangian_advect(self,
                                 concentration: np.ndarray,
                                 velocity_field: np.ndarray,
                                 dt: float) -> np.ndarray:
        """
        Semi-Lagrangian advection.

        Traces backward along characteristics and interpolates.
        """
        ndim = len(self.grid_shape)
        dx = self.config.grid_spacing

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

        # Calculate departure points
        departure_coords = []
        for i in range(ndim):
            departure = coords[i] - velocity_field[..., i] * dt / dx

            # Apply boundary conditions
            if self.config.boundary_condition == 'periodic':
                departure = departure % self.grid_shape[i]
            else:
                departure = np.clip(departure, 0, self.grid_shape[i] - 1)

            departure_coords.append(departure)

        # Interpolate
        advected = map_coordinates(
            concentration,
            departure_coords,
            order=self.config.interpolation_order,
            mode='wrap' if self.config.boundary_condition == 'periodic' else 'nearest'
        )

        return advected

    def _implicit_diffusion(self,
                            concentration: np.ndarray,
                            diffusion_coeff: float,
                            dt: float) -> np.ndarray:
        """
        Implicit diffusion using explicit approximation.

        For simplicity, uses explicit scheme with small time steps.
        A full implementation would use sparse linear solvers.
        """
        # For now, use explicit diffusion with subcycling
        n_subcycles = max(1, int(diffusion_coeff * dt / (self.config.grid_spacing ** 2) * 10))
        dt_sub = dt / n_subcycles

        c = concentration.copy()
        for _ in range(n_subcycles):
            c = self._diffuse_step(c, diffusion_coeff, dt_sub)

        return c

    def _diffuse_step(self,
                      concentration: np.ndarray,
                      diffusion_coeff: float,
                      dt: float) -> np.ndarray:
        """Single diffusion step."""
        ndim = len(self.grid_shape)
        c = concentration.copy()
        dx = self.config.grid_spacing
        alpha = diffusion_coeff * dt / (dx ** 2)

        laplacian = np.zeros_like(c)

        for dim in range(ndim):
            c_forward = np.roll(c, -1, axis=dim)
            c_backward = np.roll(c, 1, axis=dim)
            laplacian += c_forward - 2 * c + c_backward

        c = c + alpha * laplacian

        return c

    def get_concentration(self) -> np.ndarray:
        """Get current concentration field."""
        return self.concentration.copy()


class ClassicalSpectralTransport(BaseTransport):
    """
    Classical spectral transport method using FFT.

    Solves transport in spectral space for high accuracy
    with periodic boundary conditions.
    """

    def __init__(self, config):
        """Initialize classical spectral transport."""
        super().__init__(config)

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize with initial concentration field."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance by one time step using spectral method."""
        # Transform to spectral space
        c_hat = np.fft.fftn(self.concentration)

        # Get wave numbers
        ndim = len(self.grid_shape)
        wave_numbers = []
        for i in range(ndim):
            k = 2 * np.pi * np.fft.fftfreq(self.grid_shape[i],
                                            d=self.config.grid_spacing)
            wave_numbers.append(k)

        # Create meshgrid
        if ndim == 1:
            k_grid = [wave_numbers[0]]
        else:
            k_grid = np.meshgrid(*wave_numbers, indexing='ij')

        # Diffusion term: exp(-K * k^2 * dt)
        k_squared = sum(k**2 for k in k_grid)
        diffusion_factor = np.exp(-diffusion_coeff * k_squared * dt)

        # Apply diffusion in spectral space
        c_hat *= diffusion_factor

        # Advection: simple spectral advection for constant velocity
        # For spatially varying velocity, this is approximate
        u_mean = np.mean(velocity_field, axis=tuple(range(ndim)))
        for i in range(ndim):
            phase_shift = -1j * k_grid[i] * u_mean[i] * dt
            c_hat *= np.exp(phase_shift)

        # Transform back to physical space
        self.concentration = np.real(np.fft.ifftn(c_hat))

        # Add source term
        if source_term is not None:
            self.concentration += source_term * dt

        # Ensure non-negativity
        self.concentration = np.maximum(self.concentration, 0.0)

        return self.concentration.copy()

    def get_concentration(self) -> np.ndarray:
        """Get current concentration field."""
        return self.concentration.copy()
