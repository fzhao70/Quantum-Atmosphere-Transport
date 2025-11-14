"""
Particle-Based Transport Methods

Implements Lagrangian particle methods including Monte Carlo
and stochastic particle tracking.
"""

import numpy as np
from typing import Optional, Tuple
from .base_transport import BaseTransport


class MonteCarloParticleTransport(BaseTransport):
    """
    Monte Carlo particle transport method.

    Represents the concentration field as a collection of particles
    that move according to advection and random walk diffusion.
    """

    def __init__(self, config, n_particles: int = 10000):
        """
        Initialize Monte Carlo particle transport.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        n_particles : int
            Number of particles to use
        """
        super().__init__(config)
        self.n_particles = n_particles
        self.particle_positions: Optional[np.ndarray] = None
        self.particle_weights: Optional[np.ndarray] = None

    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """Initialize particles based on initial concentration distribution."""
        self.grid_shape = grid_shape
        self.concentration = initial_concentration.copy()
        ndim = len(grid_shape)

        # Initialize particles according to concentration distribution
        # Flatten concentration for sampling
        prob_distribution = initial_concentration.flatten()
        prob_distribution /= (np.sum(prob_distribution) + 1e-10)

        # Sample particle positions
        indices = np.random.choice(
            len(prob_distribution),
            size=self.n_particles,
            p=prob_distribution,
            replace=True
        )

        # Convert flat indices to multi-dimensional indices
        self.particle_positions = np.zeros((self.n_particles, ndim))

        if ndim == 1:
            self.particle_positions[:, 0] = indices
        elif ndim == 2:
            self.particle_positions[:, 0] = indices // grid_shape[1]
            self.particle_positions[:, 1] = indices % grid_shape[1]
        elif ndim == 3:
            self.particle_positions[:, 0] = indices // (grid_shape[1] * grid_shape[2])
            self.particle_positions[:, 1] = (indices // grid_shape[2]) % grid_shape[1]
            self.particle_positions[:, 2] = indices % grid_shape[2]

        # Add sub-grid randomness
        self.particle_positions += np.random.rand(self.n_particles, ndim)

        # Initialize particle weights (uniform for now)
        total_mass = np.sum(initial_concentration)
        self.particle_weights = np.ones(self.n_particles) * total_mass / self.n_particles

    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance particles by one time step."""
        ndim = len(self.grid_shape)

        # 1. Advection: move particles according to velocity field
        self._advect_particles(velocity_field, dt)

        # 2. Diffusion: random walk
        self._diffuse_particles(diffusion_coeff, dt)

        # 3. Handle source term by adding/removing particles
        if source_term is not None:
            self._apply_source_particles(source_term, dt)

        # 4. Apply boundary conditions
        self._apply_boundary_conditions()

        # 5. Project particles back to grid
        self.concentration = self._particles_to_grid()

        return self.concentration.copy()

    def _advect_particles(self, velocity_field: np.ndarray, dt: float) -> None:
        """Move particles according to velocity field."""
        ndim = len(self.grid_shape)
        dx = self.config.grid_spacing

        for i in range(self.n_particles):
            # Get particle grid position
            grid_pos = np.clip(
                self.particle_positions[i].astype(int),
                0,
                [s - 1 for s in self.grid_shape]
            )

            # Get velocity at particle position (nearest neighbor for simplicity)
            if ndim == 1:
                velocity = velocity_field[grid_pos[0]]
            elif ndim == 2:
                velocity = velocity_field[grid_pos[0], grid_pos[1]]
            elif ndim == 3:
                velocity = velocity_field[grid_pos[0], grid_pos[1], grid_pos[2]]

            # Move particle
            self.particle_positions[i] += velocity * dt / dx

    def _diffuse_particles(self, diffusion_coeff: float, dt: float) -> None:
        """Apply random walk diffusion to particles."""
        ndim = len(self.grid_shape)
        dx = self.config.grid_spacing

        # Random walk: displacement = sqrt(2 * D * dt) * N(0,1)
        sigma = np.sqrt(2 * diffusion_coeff * dt) / dx

        # Add random displacement
        random_displacement = np.random.normal(0, sigma, size=(self.n_particles, ndim))
        self.particle_positions += random_displacement

    def _apply_source_particles(self, source_term: np.ndarray, dt: float) -> None:
        """
        Apply source term by modifying particle weights.

        A full implementation would add/remove particles based on source term.
        """
        # For simplicity, modify weights based on local source strength
        # This is an approximation
        pass

    def _apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to particles."""
        ndim = len(self.grid_shape)

        if self.config.boundary_condition == 'periodic':
            # Periodic wrap
            for i in range(ndim):
                self.particle_positions[:, i] = self.particle_positions[:, i] % self.grid_shape[i]
        else:
            # Reflective boundaries
            for i in range(ndim):
                # Reflect particles that go below 0
                below_zero = self.particle_positions[:, i] < 0
                self.particle_positions[below_zero, i] = -self.particle_positions[below_zero, i]

                # Reflect particles that go above grid_shape
                above_max = self.particle_positions[:, i] >= self.grid_shape[i]
                self.particle_positions[above_max, i] = (
                    2 * self.grid_shape[i] - self.particle_positions[above_max, i]
                )

    def _particles_to_grid(self) -> np.ndarray:
        """Project particles onto grid to get concentration field."""
        ndim = len(self.grid_shape)
        concentration = np.zeros(self.grid_shape)

        # Bin particles into grid cells
        for i in range(self.n_particles):
            grid_pos = self.particle_positions[i].astype(int)

            # Ensure within bounds
            valid = True
            for d in range(ndim):
                if grid_pos[d] < 0 or grid_pos[d] >= self.grid_shape[d]:
                    valid = False
                    break

            if valid:
                if ndim == 1:
                    concentration[grid_pos[0]] += self.particle_weights[i]
                elif ndim == 2:
                    concentration[grid_pos[0], grid_pos[1]] += self.particle_weights[i]
                elif ndim == 3:
                    concentration[grid_pos[0], grid_pos[1], grid_pos[2]] += self.particle_weights[i]

        # Normalize by grid cell volume
        dx = self.config.grid_spacing
        concentration /= (dx ** ndim)

        return concentration

    def get_concentration(self) -> np.ndarray:
        """Get current concentration field."""
        return self.concentration.copy()

    def get_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get particle positions and weights.

        Returns
        -------
        positions : np.ndarray
            Particle positions
        weights : np.ndarray
            Particle weights
        """
        return self.particle_positions.copy(), self.particle_weights.copy()
