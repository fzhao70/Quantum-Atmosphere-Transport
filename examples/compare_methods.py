"""
Multi-Method Comparison Example

Compares all available transport methods on the same problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantum_atmosphere_transport import (
    QuantumTracerTransport,
    ClassicalEulerianTransport,
    ClassicalSemiLagrangianTransport,
    ClassicalSpectralTransport,
    MonteCarloParticleTransport,
    FiniteVolumeTransport,
    HybridQuantumClassicalTransport,
    TransportConfig,
)
from quantum_atmosphere_transport.benchmark import (
    TransportBenchmark,
    create_test_problem,
)


def main():
    """Run multi-method comparison."""

    print("=" * 80)
    print("ATMOSPHERIC TRACER TRANSPORT: MULTI-METHOD COMPARISON")
    print("=" * 80)

    # Problem setup
    print("\n[1/5] Setting up test problem...")
    grid_shape = (40, 40)
    problem = create_test_problem(grid_shape, problem_type='gaussian')

    initial_concentration = problem['initial_concentration']
    velocity_field = problem['velocity_field']

    # Simulation parameters
    dt = 0.01
    n_steps = 50
    diffusion_coeff = 0.05

    # Configuration
    config = TransportConfig(
        grid_spacing=1.0,
        boundary_condition='periodic',
        advection_scheme='quantum_semi_lagrangian',
        interpolation_order=3,
        apply_decoherence=True,
        decoherence_rate=0.01,
    )

    # Initialize all methods
    print("[2/5] Initializing transport methods...")

    methods = [
        QuantumTracerTransport(config),
        ClassicalSemiLagrangianTransport(config),
        ClassicalSpectralTransport(config),
        FiniteVolumeTransport(config, flux_limiter='minmod'),
        MonteCarloParticleTransport(config, n_particles=5000),
        HybridQuantumClassicalTransport(config, quantum_ratio=0.5),
    ]

    method_names = [
        "Quantum Walk",
        "Classical Semi-Lagrangian",
        "Classical Spectral (FFT)",
        "Finite Volume (Minmod)",
        "Monte Carlo Particles",
        "Hybrid Quantum-Classical",
    ]

    # Run benchmark
    print("[3/5] Running benchmark...")
    benchmark = TransportBenchmark(methods, method_names)

    results = benchmark.run(
        initial_concentration=initial_concentration,
        grid_shape=grid_shape,
        velocity_field=velocity_field,
        diffusion_coeff=diffusion_coeff,
        dt=dt,
        n_steps=n_steps,
    )

    # Print summary
    print("\n[4/5] Benchmark results:")
    benchmark.print_summary()

    # Additional analysis
    print(f"\nFastest method: {benchmark.get_fastest_method()}")
    print(f"Best mass conservation: {benchmark.get_best_mass_conservation()}")

    # Create comparison visualization
    print("\n[5/5] Creating visualization...")

    fig = plt.figure(figsize=(18, 10))

    # Plot initial condition
    ax = plt.subplot(2, 4, 1)
    im = ax.imshow(initial_concentration.T, origin='lower', cmap='hot')
    ax.set_title('Initial Condition')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='Concentration')

    # Plot each method's result
    for idx, (name, result) in enumerate(results.items(), start=2):
        if idx > 7:
            break

        ax = plt.subplot(2, 4, idx)
        concentration = result['final_concentration']
        im = ax.imshow(concentration.T, origin='lower', cmap='hot',
                      vmin=0, vmax=np.max(initial_concentration))
        ax.set_title(f'{name}\n(t={dt*n_steps:.3f}s)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='Concentration')

        # Add performance info
        textstr = (f"Time: {result['elapsed_time']:.3f}s\n"
                  f"Mass err: {result['mass_conservation_error']:.2e}")
        ax.text(0.02, 0.98, textstr,
               transform=ax.transAxes,
               fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent / 'method_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")

    # Create performance comparison chart
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Execution time comparison
    names = list(results.keys())
    times = [results[name]['elapsed_time'] for name in names]

    ax1.barh(names, times, color='steelblue')
    ax1.set_xlabel('Execution Time (s)')
    ax1.set_title('Computational Performance')
    ax1.grid(axis='x', alpha=0.3)

    # Mass conservation comparison
    mass_errors = [results[name]['mass_conservation_error'] for name in names]

    ax2.barh(names, mass_errors, color='coral')
    ax2.set_xlabel('Relative Mass Error')
    ax2.set_xscale('log')
    ax2.set_title('Mass Conservation')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save performance chart
    perf_file = Path(__file__).parent / 'performance_comparison.png'
    plt.savefig(perf_file, dpi=150, bbox_inches='tight')
    print(f"Performance chart saved to: {perf_file}")

    # Export numerical results
    results_file = Path(__file__).parent / 'benchmark_results.npz'
    benchmark.export_results(str(results_file))

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  • Fastest method: {benchmark.get_fastest_method()}")
    print(f"  • Best conservation: {benchmark.get_best_mass_conservation()}")
    print(f"\nFiles generated:")
    print(f"  • {output_file}")
    print(f"  • {perf_file}")
    print(f"  • {results_file}")


if __name__ == '__main__':
    main()
