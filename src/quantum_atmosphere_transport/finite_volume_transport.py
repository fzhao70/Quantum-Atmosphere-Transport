"""
Finite Volume Transport Method

Implements finite volume method with flux limiters
for conservative and monotonic transport.
"""

import numpy as np
from typing import Optional, Tuple
from .base_transport import BaseTransport


class FiniteVolumeTransport(BaseTransport):
    """
    Finite volume transport method.

    Uses flux-based formulation with limiters for stability and monotonicity.
    Provides conservative transport suitable for atmospheric applications.
    """

    def __init__(self, config, flux_limiter: str = 'minmod'):
        """
        Initialize finite volume transport.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        flux_limiter : str
            Flux limiter type: 'minmod', 'superbee', 'van_leer', or 'mc'
        """
        super().__init__(config)
        self.flux_limiter = flux_limiter

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
        """Advance by one time step using finite volume method."""
        # Split operator: advection + diffusion
        # 1. Advection with flux limiters
        self.concentration = self._advect_finite_volume(
            self.concentration, velocity_field, dt
        )

        # 2. Diffusion
        self.concentration = self._diffuse_finite_volume(
            self.concentration, diffusion_coeff, dt
        )

        # 3. Apply source term
        if source_term is not None:
            self.concentration += source_term * dt

        # 4. Ensure non-negativity
        self.concentration = np.maximum(self.concentration, 0.0)

        return self.concentration.copy()

    def _advect_finite_volume(self,
                               concentration: np.ndarray,
                               velocity_field: np.ndarray,
                               dt: float) -> np.ndarray:
        """
        Finite volume advection with flux limiters.

        Uses MUSCL-Hancock scheme for second-order accuracy.
        """
        ndim = len(self.grid_shape)
        c = concentration.copy()
        dx = self.config.grid_spacing

        for dim in range(ndim):
            u = velocity_field[..., dim]

            # Calculate fluxes at cell faces
            flux = self._calculate_flux(c, u, dim)

            # Update using flux difference
            flux_forward = np.roll(flux, -1, axis=dim)
            flux_backward = flux

            c = c - (dt / dx) * (flux_forward - flux_backward)

        return c

    def _calculate_flux(self,
                        concentration: np.ndarray,
                        velocity: np.ndarray,
                        dim: int) -> np.ndarray:
        """
        Calculate flux at cell faces using flux limiters.

        Parameters
        ----------
        concentration : np.ndarray
            Concentration field
        velocity : np.ndarray
            Velocity component
        dim : int
            Dimension index

        Returns
        -------
        np.ndarray
            Flux at cell faces
        """
        # Get values at neighboring cells
        c_center = concentration
        c_forward = np.roll(concentration, -1, axis=dim)
        c_backward = np.roll(concentration, 1, axis=dim)
        c_forward2 = np.roll(concentration, -2, axis=dim)

        # Calculate gradients
        delta_plus = c_forward - c_center
        delta_minus = c_center - c_backward

        # Avoid division by zero
        epsilon = 1e-10
        r = np.where(
            np.abs(delta_plus) > epsilon,
            delta_minus / (delta_plus + epsilon),
            0.0
        )

        # Apply flux limiter
        phi = self._flux_limiter_function(r)

        # Calculate limited gradient
        limited_gradient = phi * delta_plus

        # Upwind flux with limiter correction
        flux = np.where(
            velocity >= 0,
            velocity * (c_center + 0.5 * limited_gradient),
            velocity * (c_forward - 0.5 * self._flux_limiter_function(1/r) * delta_plus)
        )

        return flux

    def _flux_limiter_function(self, r: np.ndarray) -> np.ndarray:
        """
        Apply flux limiter function.

        Parameters
        ----------
        r : np.ndarray
            Gradient ratio

        Returns
        -------
        np.ndarray
            Limiter function value
        """
        if self.flux_limiter == 'minmod':
            # Minmod: phi(r) = max(0, min(1, r))
            return np.maximum(0.0, np.minimum(1.0, r))

        elif self.flux_limiter == 'superbee':
            # Superbee: phi(r) = max(0, min(2r, 1), min(r, 2))
            return np.maximum(
                np.maximum(0.0, np.minimum(2*r, 1.0)),
                np.minimum(r, 2.0)
            )

        elif self.flux_limiter == 'van_leer':
            # Van Leer: phi(r) = (r + |r|) / (1 + |r|)
            return (r + np.abs(r)) / (1.0 + np.abs(r))

        elif self.flux_limiter == 'mc':
            # MC (Monotonized Central): phi(r) = max(0, min((1+r)/2, 2, 2r))
            return np.maximum(
                0.0,
                np.minimum(np.minimum((1.0 + r) / 2.0, 2.0), 2.0 * r)
            )

        else:
            # Default to minmod
            return np.maximum(0.0, np.minimum(1.0, r))

    def _diffuse_finite_volume(self,
                                concentration: np.ndarray,
                                diffusion_coeff: float,
                                dt: float) -> np.ndarray:
        """
        Finite volume diffusion.

        Uses centered differences for diffusion fluxes.
        """
        ndim = len(self.grid_shape)
        c = concentration.copy()
        dx = self.config.grid_spacing

        for dim in range(ndim):
            # Calculate diffusion fluxes at faces
            c_forward = np.roll(c, -1, axis=dim)
            c_backward = np.roll(c, 1, axis=dim)

            # Flux at right face: -D * (c[i+1] - c[i]) / dx
            flux_right = -diffusion_coeff * (c_forward - c) / dx

            # Flux at left face: -D * (c[i] - c[i-1]) / dx
            flux_left = -diffusion_coeff * (c - c_backward) / dx

            # Update: dc/dt = -(flux_right - flux_left) / dx
            c = c - (dt / dx) * (flux_right - flux_left)

        return c

    def get_concentration(self) -> np.ndarray:
        """Get current concentration field."""
        return self.concentration.copy()
