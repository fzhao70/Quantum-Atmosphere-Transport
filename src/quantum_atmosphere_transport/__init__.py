"""
Quantum Atmospheric Transport Scheme

A comprehensive atmospheric tracer transport package with multiple solution methods:
- Quantum algorithms (quantum walks, superposition)
- Classical methods (Eulerian, semi-Lagrangian, spectral)
- Particle-based methods (Monte Carlo)
- Finite volume methods (with flux limiters)
- Hybrid quantum-classical methods
"""

# Quantum methods
from .quantum_transport import QuantumTracerTransport
from .quantum_diffusion import QuantumDiffusion
from .quantum_advection import QuantumAdvection

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
    # Quantum methods
    "QuantumTracerTransport",
    "QuantumDiffusion",
    "QuantumAdvection",
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
