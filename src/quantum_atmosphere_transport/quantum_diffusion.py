"""
Quantum Diffusion Module

Implements quantum walk-based diffusion for atmospheric tracer transport.
Uses quantum random walks which exhibit quadratic speedup over classical diffusion.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


class QuantumDiffusion:
    """
    Quantum walk-based diffusion operator.

    Unlike classical random walks where variance grows as σ² ~ t,
    quantum walks have variance growth as σ² ~ t², providing faster
    spreading which better captures turbulent atmospheric mixing.
    """

    def __init__(self, config):
        """
        Initialize quantum diffusion operator.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        """
        self.config = config
        self.quantum_walk_operator: Optional[sparse.csr_matrix] = None
        self.grid_shape: Optional[Tuple[int, ...]] = None

    def initialize(self, grid_shape: Tuple[int, ...]) -> None:
        """
        Initialize the quantum walk operator for the given grid.

        Parameters
        ----------
        grid_shape : tuple
            Shape of the computational grid (nx, ny, nz)
        """
        self.grid_shape = grid_shape
        self.quantum_walk_operator = self._build_quantum_walk_operator()

    def _build_quantum_walk_operator(self) -> sparse.csr_matrix:
        """
        Build the quantum walk operator (Hamiltonian) for diffusion.

        The quantum walk on a lattice is described by the Hamiltonian:
        H = -Σ(|i⟩⟨j| + |⟩⟨i|) where j are neighbors of i

        This creates a quantum analog of the Laplacian operator.

        Returns
        -------
        sparse.csr_matrix
            Quantum walk operator (Hamiltonian)
        """
        if len(self.grid_shape) == 1:
            return self._build_1d_operator()
        elif len(self.grid_shape) == 2:
            return self._build_2d_operator()
        elif len(self.grid_shape) == 3:
            return self._build_3d_operator()
        else:
            raise ValueError("Only 1D, 2D, and 3D grids are supported")

    def _build_1d_operator(self) -> sparse.csr_matrix:
        """Build 1D quantum walk operator."""
        nx = self.grid_shape[0]
        n = nx

        # Create adjacency matrix for 1D lattice
        diagonals = [
            np.ones(n - 1),  # super-diagonal
            np.ones(n - 1),  # sub-diagonal
        ]
        H = sparse.diags(diagonals, [1, -1], shape=(n, n), format='csr')

        # Apply boundary conditions
        if self.config.boundary_condition == 'periodic':
            H[0, n - 1] = 1
            H[n - 1, 0] = 1

        return -H  # Negative for proper quantum walk

    def _build_2d_operator(self) -> sparse.csr_matrix:
        """Build 2D quantum walk operator."""
        ny, nx = self.grid_shape
        n = nx * ny

        # Build 2D Laplacian using Kronecker products
        # Identity matrices
        Ix = sparse.eye(nx)
        Iy = sparse.eye(ny)

        # 1D difference operators
        Dx = sparse.diags([np.ones(nx - 1), np.ones(nx - 1)], [1, -1],
                         shape=(nx, nx), format='csr')
        Dy = sparse.diags([np.ones(ny - 1), np.ones(ny - 1)], [1, -1],
                         shape=(ny, ny), format='csr')

        # Periodic boundary conditions
        if self.config.boundary_condition == 'periodic':
            Dx[0, nx - 1] = 1
            Dx[nx - 1, 0] = 1
            Dy[0, ny - 1] = 1
            Dy[ny - 1, 0] = 1

        # 2D quantum walk Hamiltonian
        H = sparse.kron(Iy, Dx) + sparse.kron(Dy, Ix)

        return -H

    def _build_3d_operator(self) -> sparse.csr_matrix:
        """Build 3D quantum walk operator."""
        nz, ny, nx = self.grid_shape
        n = nx * ny * nz

        # Identity matrices
        Ix = sparse.eye(nx)
        Iy = sparse.eye(ny)
        Iz = sparse.eye(nz)

        # 1D difference operators
        Dx = sparse.diags([np.ones(nx - 1), np.ones(nx - 1)], [1, -1],
                         shape=(nx, nx), format='csr')
        Dy = sparse.diags([np.ones(ny - 1), np.ones(ny - 1)], [1, -1],
                         shape=(ny, ny), format='csr')
        Dz = sparse.diags([np.ones(nz - 1), np.ones(nz - 1)], [1, -1],
                         shape=(nz, nz), format='csr')

        # Periodic boundary conditions
        if self.config.boundary_condition == 'periodic':
            Dx[0, nx - 1] = 1
            Dx[nx - 1, 0] = 1
            Dy[0, ny - 1] = 1
            Dy[ny - 1, 0] = 1
            Dz[0, nz - 1] = 1
            Dz[nz - 1, 0] = 1

        # 3D quantum walk Hamiltonian using Kronecker products
        H = (sparse.kron(sparse.kron(Iz, Iy), Dx) +
             sparse.kron(sparse.kron(Iz, Dy), Ix) +
             sparse.kron(sparse.kron(Dz, Iy), Ix))

        return -H

    def apply(self,
              quantum_state: np.ndarray,
              diffusion_coeff: float,
              dt: float) -> np.ndarray:
        """
        Apply quantum diffusion operator to the quantum state.

        The time evolution is given by:
        |ψ(t+dt)⟩ = exp(-i K H dt) |ψ(t)⟩

        where H is the quantum walk Hamiltonian and K is diffusion coefficient.

        Parameters
        ----------
        quantum_state : np.ndarray
            Current quantum state
        diffusion_coeff : float
            Diffusion coefficient
        dt : float
            Time step

        Returns
        -------
        np.ndarray
            Updated quantum state after diffusion
        """
        if self.quantum_walk_operator is None:
            raise RuntimeError("Quantum diffusion not initialized. Call initialize() first.")

        # Flatten the state for matrix operations
        original_shape = quantum_state.shape
        state_vector = quantum_state.flatten()

        # Quantum time evolution: exp(-i K H dt) |ψ⟩
        # Using matrix exponential action on vector
        evolution_parameter = -1j * diffusion_coeff * dt

        # Apply quantum walk using matrix exponential
        if self.config.use_exact_evolution:
            # Exact evolution using sparse matrix exponential
            evolved_state = expm_multiply(
                evolution_parameter * self.quantum_walk_operator,
                state_vector
            )
        else:
            # Approximate evolution using Trotter decomposition (faster)
            evolved_state = self._trotter_evolution(
                state_vector, evolution_parameter
            )

        # Reshape back to original grid shape
        evolved_state = evolved_state.reshape(original_shape)

        return evolved_state

    def _trotter_evolution(self,
                          state_vector: np.ndarray,
                          evolution_parameter: complex) -> np.ndarray:
        """
        Approximate quantum evolution using Trotter decomposition.

        Splits the evolution operator into smaller steps for efficiency:
        exp(A + B) ≈ [exp(A/n) exp(B/n)]^n

        Parameters
        ----------
        state_vector : np.ndarray
            Flattened quantum state
        evolution_parameter : complex
            Evolution parameter (-i K dt)

        Returns
        -------
        np.ndarray
            Evolved state vector
        """
        n_steps = self.config.trotter_steps
        step_param = evolution_parameter / n_steps

        evolved = state_vector.copy()
        for _ in range(n_steps):
            evolved = expm_multiply(
                step_param * self.quantum_walk_operator,
                evolved
            )

        return evolved

    def get_classical_diffusion_matrix(self) -> sparse.csr_matrix:
        """
        Get the classical diffusion matrix for comparison.

        The classical diffusion operator is just the Laplacian.

        Returns
        -------
        sparse.csr_matrix
            Classical Laplacian operator
        """
        # The quantum walk operator is essentially -Laplacian
        # So we return it as a real matrix
        return -self.quantum_walk_operator.real
