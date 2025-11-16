"""
Quantum Atmospheric Transport Scheme

A comprehensive atmospheric tracer transport package with multiple solution methods:
- Quantum algorithms (15+ different quantum methods)
- Classical methods (Eulerian, semi-Lagrangian, spectral)
- Particle-based methods (Monte Carlo)
- Finite volume methods (with flux limiters)
- Hybrid quantum-classical methods
"""

# Core quantum methods
from .quantum_transport import QuantumTracerTransport
from .quantum_diffusion import QuantumDiffusion
from .quantum_advection import QuantumAdvection

# Advanced quantum walk methods
from .advanced_quantum_walks import (
    DiscreteTimeQuantumWalkTransport,
    StaggeredQuantumWalkTransport,
    QuantumAmplificationTransport,
)

# Variational quantum methods
from .variational_quantum import (
    VariationalQuantumTransport,
    QuantumNeuralTransport,
)

# Quantum linear solver methods
from .quantum_linear_solver import (
    HHLQuantumTransport,
    QuantumMatrixInversionTransport,
)

# Tensor network methods
from .tensor_network import MPSQuantumTransport

# Quantum automata and annealing
from .quantum_automata import (
    QuantumCellularAutomataTransport,
    QuantumAnnealingTransport,
)

# Classical methods
from .classical_transport import (
    ClassicalEulerianTransport,
    ClassicalSemiLagrangianTransport,
    ClassicalSpectralTransport,
)

# Particle methods
from .particle_transport import MonteCarloParticleTransport

# Finite volume methods
from .finite_volume_transport import FiniteVolumeTransport

# Hybrid methods
from .hybrid_transport import (
    HybridQuantumClassicalTransport,
    AdaptiveQuantumClassicalTransport,
)

# Base interface and configuration
from .base_transport import BaseTransport
from .config import TransportConfig

__version__ = "0.1.0"
__author__ = "Fanghe Zhao"

__all__ = [
    # Core quantum methods
    "QuantumTracerTransport",
    "QuantumDiffusion",
    "QuantumAdvection",
    # Advanced quantum walk methods
    "DiscreteTimeQuantumWalkTransport",
    "StaggeredQuantumWalkTransport",
    "QuantumAmplificationTransport",
    # Variational quantum methods
    "VariationalQuantumTransport",
    "QuantumNeuralTransport",
    # Quantum linear solver methods
    "HHLQuantumTransport",
    "QuantumMatrixInversionTransport",
    # Tensor network methods
    "MPSQuantumTransport",
    # Quantum automata and annealing
    "QuantumCellularAutomataTransport",
    "QuantumAnnealingTransport",
    # Classical methods
    "ClassicalEulerianTransport",
    "ClassicalSemiLagrangianTransport",
    "ClassicalSpectralTransport",
    # Particle methods
    "MonteCarloParticleTransport",
    # Finite volume methods
    "FiniteVolumeTransport",
    # Hybrid methods
    "HybridQuantumClassicalTransport",
    "AdaptiveQuantumClassicalTransport",
    # Base and config
    "BaseTransport",
    "TransportConfig",
]
