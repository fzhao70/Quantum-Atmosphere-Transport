# Quantum-Atmosphere-Transport

A comprehensive atmospheric tracer transport package with **multiple solution methods** for numerical weather prediction and climate modeling.

## Overview

This package provides **7 different transport methods** for solving atmospheric tracer transport equations, ranging from quantum algorithms to classical numerical methods. This allows users to:

- **Compare** different approaches on the same problem
- **Choose** the best method for their specific application
- **Benchmark** performance and accuracy
- **Research** novel quantum-inspired algorithms

### Available Methods

1. **Quantum Walk Transport** - Quantum algorithms with superposition and quantum walks
2. **Classical Eulerian** - Traditional finite difference method
3. **Classical Semi-Lagrangian** - Backward trajectory method
4. **Classical Spectral** - FFT-based spectral method
5. **Monte Carlo Particles** - Lagrangian particle tracking
6. **Finite Volume** - Conservative flux-based method with limiters
7. **Hybrid Quantum-Classical** - Best of both worlds

All methods solve the advection-diffusion equation:

```
∂C/∂t + u·∇C = ∇·(K∇C) + S
```

where:
- C is tracer concentration
- u is velocity field (wind)
- K is diffusion coefficient
- S is source/sink term

## Features

### Quantum Methods
- **Quantum Walk Algorithms** for diffusion modeling (quadratic spreading)
- **Quantum Superposition** for concentration field representation
- **Quantum Phase Shifts** for unitary advection operators
- **Quantum Decoherence** for numerical stability
- **Quantum Statistics** tracking (coherence, entanglement)

### Classical Methods
- **Eulerian**: Explicit finite differences with upwind advection
- **Semi-Lagrangian**: Unconditionally stable backward trajectories
- **Spectral**: High-accuracy FFT-based method
- **Finite Volume**: Conservative with flux limiters (minmod, superbee, van Leer, MC)

### Particle Methods
- **Monte Carlo**: Stochastic particle tracking with random walk diffusion
- Lagrangian framework with automatic grid projection

### Hybrid Methods
- **Fixed Hybrid**: Blend quantum and classical with adjustable ratio
- **Adaptive Hybrid**: Automatically switches based on turbulence indicators

### General Features
- **Unified Interface**: All methods inherit from common base class
- **Benchmarking Framework**: Built-in comparison and performance tools
- **1D, 2D, and 3D support**: Works with any dimensionality
- **Boundary conditions**: Periodic and fixed boundary conditions
- **Mass conservation**: Tracked for all methods
- **Flexible configuration**: Easy-to-use configuration system

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

## Comparing Multiple Methods

The package includes a comprehensive benchmarking framework:

```python
from quantum_atmosphere_transport import (
    QuantumTracerTransport,
    ClassicalSemiLagrangianTransport,
    MonteCarloParticleTransport,
    FiniteVolumeTransport,
    TransportConfig,
)
from quantum_atmosphere_transport.benchmark import TransportBenchmark, create_test_problem

# Create test problem
config = TransportConfig()
problem = create_test_problem(grid_shape=(50, 50), problem_type='gaussian')

# Initialize methods to compare
methods = [
    QuantumTracerTransport(config),
    ClassicalSemiLagrangianTransport(config),
    MonteCarloParticleTransport(config, n_particles=10000),
    FiniteVolumeTransport(config, flux_limiter='minmod'),
]

# Run benchmark
benchmark = TransportBenchmark(methods,
    method_names=['Quantum', 'Semi-Lagrangian', 'Monte Carlo', 'Finite Volume'])

results = benchmark.run(
    initial_concentration=problem['initial_concentration'],
    grid_shape=problem['grid_shape'],
    velocity_field=problem['velocity_field'],
    diffusion_coeff=100.0,
    dt=0.1,
    n_steps=100,
)

# Print comparison
benchmark.print_summary()
print(f"Fastest method: {benchmark.get_fastest_method()}")
print(f"Best mass conservation: {benchmark.get_best_mass_conservation()}")

# Export results
benchmark.export_results('benchmark_results.npz')
```

See `examples/compare_methods.py` for a complete multi-method comparison with visualization.

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

The `examples/` directory contains complete working examples:

### Simple Quantum Transport

```bash
python examples/simple_transport.py
```

Demonstrates quantum transport with visualization showing:
- Tracer concentration evolution over time
- Quantum statistics (coherence, entanglement, phase variance)
- Transport in a uniform wind field

### Multi-Method Comparison

```bash
python examples/compare_methods.py
```

Comprehensive comparison of all 7 transport methods:
- Side-by-side concentration field visualizations
- Performance benchmarks (execution time, mass conservation)
- Statistical comparison charts
- Exports numerical results for further analysis

This example automatically:
- Runs all methods on the same problem
- Generates comparison plots
- Reports fastest method and best conservation
- Saves results to files

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

## Method Selection Guide

Choose the best method for your application:

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Quantum Walk** | Research, turbulent flows | Novel approach, enhanced spreading | Computationally intensive |
| **Semi-Lagrangian** | General purpose, large CFL | Unconditionally stable, accurate | Requires interpolation |
| **Spectral** | Smooth flows, periodic domains | High accuracy, fast for large grids | Requires periodic BCs |
| **Finite Volume** | Conservation-critical applications | Strictly conservative, monotonic | More complex implementation |
| **Monte Carlo** | Lagrangian tracking, sparse plumes | Natural for particles, local adaptivity | Statistical noise |
| **Eulerian** | Simple applications, teaching | Easy to understand | Stability restrictions |
| **Hybrid** | Varied flow regimes | Combines strengths | Parameter tuning needed |

### Performance Considerations

- **Grid Size**:
  - Spectral methods: O(N log N) with FFT
  - Other methods: O(N) to O(N²) depending on scheme
  - Quantum methods: Additional overhead for state management

- **Time Step**:
  - Semi-Lagrangian, Spectral: Unconditionally stable (large dt possible)
  - Eulerian, Finite Volume: CFL condition applies
  - Monte Carlo: Adaptive (based on diffusion length scale)

- **Accuracy**:
  - Spectral: Highest for smooth solutions
  - Semi-Lagrangian: Second-order accurate
  - Finite Volume: First to second-order (limiter dependent)
  - Quantum: Research ongoing

- **Memory**:
  - Quantum: ~2x (complex-valued states)
  - Monte Carlo: ~N_particles × dim
  - Others: ~1x concentration field

### Recommended Workflow

1. **Start**: Use `ClassicalSemiLagrangianTransport` for reliable baseline
2. **Compare**: Run `examples/compare_methods.py` on your problem
3. **Optimize**: Select fastest method that meets accuracy requirements
4. **Research**: Experiment with quantum methods for novel insights

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
