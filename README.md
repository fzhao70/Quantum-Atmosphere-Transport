# Quantum-Atmosphere-Transport

A quantum algorithm-based atmospheric tracer transport scheme for numerical weather prediction and climate modeling.

## Overview

This package implements a novel quantum-inspired approach to solving atmospheric tracer transport equations. Unlike classical transport schemes, this implementation uses:

- **Quantum Walk Algorithms** for diffusion modeling, providing quadratic speedup in spreading behavior
- **Quantum Superposition** to represent and evolve concentration fields as quantum states
- **Quantum Phase Shifts** for unitary advection operators that preserve numerical properties
- **Quantum Decoherence** mechanisms for numerical stability

The scheme solves the advection-diffusion equation:

```
∂C/∂t + u·∇C = ∇·(K∇C) + S
```

where:
- C is tracer concentration
- u is velocity field (wind)
- K is diffusion coefficient
- S is source/sink term

## Features

- **Multiple advection schemes**: Semi-Lagrangian, spectral, and upwind methods with quantum enhancements
- **Quantum diffusion**: Based on quantum random walks on lattices
- **Mass conservation**: Maintained through quantum measurement and normalization
- **Flexible configuration**: Easy-to-use configuration system for all parameters
- **1D, 2D, and 3D support**: Works with any dimensionality
- **Boundary conditions**: Periodic and fixed boundary conditions
- **Quantum statistics**: Track coherence, entanglement, and phase variance

## Installation

### From source

```bash
git clone https://github.com/fzhao70/Quantum-Atmosphere-Transport.git
cd Quantum-Atmosphere-Transport
pip install -e .
```

### Requirements

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.3 (for visualization)

## Quick Start

```python
import numpy as np
from quantum_atmosphere_transport import QuantumTracerTransport, TransportConfig

# Configure the transport scheme
config = TransportConfig(
    grid_spacing=1000.0,  # 1 km grid spacing
    boundary_condition='periodic',
    advection_scheme='quantum_semi_lagrangian',
    apply_decoherence=True,
    decoherence_rate=0.01
)

# Initialize transport scheme
transport = QuantumTracerTransport(config)

# Set up initial condition (e.g., Gaussian plume)
nx, ny = 50, 50
initial_concentration = np.zeros((nx, ny))
initial_concentration[25, 25] = 1.0

transport.initialize(initial_concentration, (nx, ny))

# Create velocity field (uniform wind)
velocity_field = np.zeros((nx, ny, 2))
velocity_field[..., 0] = 5.0  # 5 m/s eastward

# Time step parameters
dt = 60.0  # 60 seconds
diffusion_coeff = 100.0  # m²/s

# Advance one time step
new_concentration = transport.step(dt, velocity_field, diffusion_coeff)

# Get quantum statistics
stats = transport.get_quantum_statistics()
print(f"Quantum coherence: {stats['coherence']:.6f}")
print(f"Entanglement: {stats['entanglement']:.6f}")
```

## Configuration Options

The `TransportConfig` class provides extensive configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_spacing` | 1000.0 | Grid spacing in meters |
| `boundary_condition` | 'periodic' | Boundary condition type ('periodic' or 'fixed') |
| `advection_scheme` | 'quantum_semi_lagrangian' | Advection method |
| `interpolation_order` | 3 | Interpolation order (1-5) |
| `quantum_phase_factor` | 0.0 | Initial quantum phase |
| `apply_decoherence` | True | Apply quantum decoherence |
| `decoherence_rate` | 0.01 | Decoherence rate (0.0-1.0) |
| `apply_phase_correction` | True | Apply geometric phase correction |
| `use_exact_evolution` | False | Use exact vs. Trotter evolution |
| `trotter_steps` | 4 | Number of Trotter steps |

### Advection Schemes

1. **quantum_semi_lagrangian**: Backward trajectory method with quantum phase preservation
2. **quantum_spectral**: FFT-based method using quantum Fourier transforms
3. **quantum_upwind**: Directional finite difference with quantum corrections

## Examples

See the `examples/` directory for complete examples:

```bash
# Run simple transport example
python examples/simple_transport.py
```

This will create a visualization showing:
- Tracer concentration evolution over time
- Quantum statistics (coherence, entanglement)
- Transport in a uniform wind field

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_quantum_transport.py -v
pytest tests/test_quantum_diffusion.py -v
pytest tests/test_config.py -v
```

## Scientific Background

### Quantum Walks for Diffusion

Classical diffusion follows a random walk where variance grows linearly with time (σ² ∝ t). Quantum walks exhibit quadratic spreading (σ² ∝ t²), better capturing turbulent atmospheric mixing processes.

### Quantum Superposition

The concentration field is represented as a quantum state |ψ⟩ where:
- Amplitude: √(concentration)
- Phase: Quantum phase information

This allows simultaneous evolution of multiple transport scenarios.

### Quantum Evolution

Time evolution is governed by the Schrödinger equation:

```
iℏ ∂|ψ⟩/∂t = H|ψ⟩
```

where H is the quantum walk Hamiltonian (related to the Laplacian).

### Measurement and Decoherence

Quantum measurement (|ψ|²) collapses the superposition to classical concentration. Decoherence gradually mixes quantum and classical states for numerical stability.

## Performance Considerations

- **Grid Size**: Scales as O(N³) for 3D problems
- **Time Evolution**: Exact evolution is slower but more accurate; Trotter decomposition provides good speed-accuracy tradeoff
- **Spectral Methods**: Most efficient for smooth flows with periodic boundaries
- **Semi-Lagrangian**: Best for complex velocity fields

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_atmosphere_transport,
  author = {Zhao, Fanghe},
  title = {Quantum-Atmosphere-Transport: A Quantum Algorithm-Based Atmospheric Tracer Transport Scheme},
  year = {2025},
  url = {https://github.com/fzhao70/Quantum-Atmosphere-Transport}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds on concepts from:
- Quantum computing and quantum walks
- Numerical weather prediction
- Atmospheric chemistry transport modeling

## Contact

Fanghe Zhao - fzhao70@github

Project Link: [https://github.com/fzhao70/Quantum-Atmosphere-Transport](https://github.com/fzhao70/Quantum-Atmosphere-Transport)
