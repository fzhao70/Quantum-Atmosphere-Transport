"""
Configuration Module

Configuration settings for the quantum atmospheric transport scheme.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TransportConfig:
    """
    Configuration for quantum atmospheric transport scheme.

    Attributes
    ----------
    grid_spacing : float
        Grid spacing in meters (default: 1000.0 m = 1 km)
    boundary_condition : str
        Boundary condition type: 'periodic' or 'fixed'
    advection_scheme : str
        Advection scheme: 'quantum_semi_lagrangian', 'quantum_spectral', or 'quantum_upwind'
    interpolation_order : int
        Order of interpolation for semi-Lagrangian scheme (1-5)
    quantum_phase_factor : float
        Initial quantum phase factor for state initialization
    apply_decoherence : bool
        Whether to apply quantum decoherence for stability
    decoherence_rate : float
        Rate of quantum decoherence (0.0 - 1.0)
    apply_phase_correction : bool
        Whether to apply geometric phase correction in advection
    use_exact_evolution : bool
        Use exact quantum evolution (slower but more accurate)
    trotter_steps : int
        Number of Trotter steps for approximate evolution
    """

    # Grid parameters
    grid_spacing: float = 1000.0  # meters

    # Boundary conditions
    boundary_condition: Literal['periodic', 'fixed'] = 'periodic'

    # Advection parameters
    advection_scheme: Literal[
        'quantum_semi_lagrangian',
        'quantum_spectral',
        'quantum_upwind'
    ] = 'quantum_semi_lagrangian'
    interpolation_order: int = 3

    # Quantum parameters
    quantum_phase_factor: float = 0.0
    apply_decoherence: bool = True
    decoherence_rate: float = 0.01

    # Phase correction
    apply_phase_correction: bool = True

    # Evolution parameters
    use_exact_evolution: bool = False
    trotter_steps: int = 4

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if self.grid_spacing <= 0:
            raise ValueError("grid_spacing must be positive")

        if self.interpolation_order < 0 or self.interpolation_order > 5:
            raise ValueError("interpolation_order must be between 0 and 5")

        if not (0.0 <= self.decoherence_rate <= 1.0):
            raise ValueError("decoherence_rate must be between 0.0 and 1.0")

        if self.trotter_steps < 1:
            raise ValueError("trotter_steps must be at least 1")

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            'grid_spacing': self.grid_spacing,
            'boundary_condition': self.boundary_condition,
            'advection_scheme': self.advection_scheme,
            'interpolation_order': self.interpolation_order,
            'quantum_phase_factor': self.quantum_phase_factor,
            'apply_decoherence': self.apply_decoherence,
            'decoherence_rate': self.decoherence_rate,
            'apply_phase_correction': self.apply_phase_correction,
            'use_exact_evolution': self.use_exact_evolution,
            'trotter_steps': self.trotter_steps,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TransportConfig':
        """
        Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary

        Returns
        -------
        TransportConfig
            Configuration object
        """
        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"TransportConfig(\n"
            f"  grid_spacing={self.grid_spacing},\n"
            f"  boundary_condition='{self.boundary_condition}',\n"
            f"  advection_scheme='{self.advection_scheme}',\n"
            f"  interpolation_order={self.interpolation_order},\n"
            f"  quantum_phase_factor={self.quantum_phase_factor},\n"
            f"  apply_decoherence={self.apply_decoherence},\n"
            f"  decoherence_rate={self.decoherence_rate},\n"
            f"  apply_phase_correction={self.apply_phase_correction},\n"
            f"  use_exact_evolution={self.use_exact_evolution},\n"
            f"  trotter_steps={self.trotter_steps}\n"
            f")"
        )
