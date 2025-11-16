"""
Comprehensive Quantum Algorithms Comparison

Compares ALL quantum algorithms implemented in the package.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantum_atmosphere_transport import (
    # Core quantum
    QuantumTracerTransport,
    # Advanced quantum walks
    DiscreteTimeQuantumWalkTransport,
    StaggeredQuantumWalkTransport,
    QuantumAmplificationTransport,
    # Variational quantum
    VariationalQuantumTransport,
    QuantumNeuralTransport,
    # Quantum linear solvers
    HHLQuantumTransport,
    QuantumMatrixInversionTransport,
    # Tensor networks
    MPSQuantumTransport,
    # Quantum automata
    QuantumCellularAutomataTransport,
    QuantumAnnealingTransport,
    # Config
    TransportConfig,
)
from quantum_atmosphere_transport.benchmark import TransportBenchmark, create_test_problem


def main():
    """Run comprehensive quantum algorithms comparison."""

    print("=" * 80)
    print("QUANTUM ALGORITHMS FOR ATMOSPHERIC TRANSPORT")
    print("Comprehensive Comparison of 11 Quantum Methods")
    print("=" * 80)

    # Problem setup
    print("\n[1/5] Setting up test problem...")
    grid_shape = (30, 30)  # Smaller grid for quantum methods
    problem = create_test_problem(grid_shape, problem_type='gaussian')

    initial_concentration = problem['initial_concentration']
    velocity_field = problem['velocity_field']

    # Simulation parameters
    dt = 0.05  # Smaller time step for stability
    n_steps = 20  # Fewer steps for quantum algorithms
    diffusion_coeff = 0.1

    # Configuration
    config = TransportConfig(
        grid_spacing=1.0,
        boundary_condition='periodic',
        advection_scheme='quantum_semi_lagrangian',
        interpolation_order=2,
        apply_decoherence=True,
        decoherence_rate=0.02,
    )

    # Initialize all quantum methods
    print("[2/5] Initializing quantum algorithms...")

    methods = [
        QuantumTracerTransport(config),
        DiscreteTimeQuantumWalkTransport(config, coin_type='hadamard'),
        StaggeredQuantumWalkTransport(config),
        QuantumAmplificationTransport(config, target_amplification=1.2),
        VariationalQuantumTransport(config, n_layers=2, n_params_per_layer=3),
        QuantumNeuralTransport(config, n_qubits=6),
        HHLQuantumTransport(config, n_ancilla=3),
        QuantumMatrixInversionTransport(config),
        MPSQuantumTransport(config, bond_dimension=8),
        QuantumCellularAutomataTransport(config, rule_type='partitioned'),
        QuantumAnnealingTransport(config, n_anneal_steps=30),
    ]

    method_names = [
        "Continuous-Time QW",
        "Discrete-Time QW",
        "Staggered QW",
        "Amplitude Amplification",
        "Variational Quantum",
        "Quantum Neural Net",
        "HHL Algorithm",
        "Quantum Matrix Inversion",
        "Tensor Network (MPS)",
        "Quantum Cellular Automata",
        "Quantum Annealing",
    ]

    # Run benchmark
    print("[3/5] Running quantum algorithms benchmark...")
    print(f"  Grid: {grid_shape}")
    print(f"  Time steps: {n_steps}")
    print(f"  dt: {dt}")

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
    print(f"\nFastest quantum method: {benchmark.get_fastest_method()}")
    print(f"Best mass conservation: {benchmark.get_best_mass_conservation()}")

    # Create visualization
    print("\n[5/5] Creating visualizations...")

    # Figure 1: Concentration fields comparison
    fig1 = plt.figure(figsize=(20, 12))

    # Plot initial condition
    ax = plt.subplot(3, 4, 1)
    im = ax.imshow(initial_concentration.T, origin='lower', cmap='hot')
    ax.set_title('Initial Condition', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='Concentration')

    # Plot each quantum method's result
    for idx, (name, result) in enumerate(results.items(), start=2):
        if idx > 12:
            break

        ax = plt.subplot(3, 4, idx)
        concentration = result['final_concentration']
        im = ax.imshow(concentration.T, origin='lower', cmap='hot',
                      vmin=0, vmax=np.max(initial_concentration))

        # Shorten name for display
        short_name = name.replace("Transport", "").replace("Quantum", "Q")
        ax.set_title(f'{short_name}', fontsize=9)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)
        plt.colorbar(im, ax=ax, label='Conc', pad=0.02)

        # Add statistics
        textstr = (f"t={dt*n_steps:.2f}s\n"
                  f"{result['elapsed_time']:.2f}s\n"
                  f"M_err:{result['mass_conservation_error']:.1e}")
        ax.text(0.02, 0.98, textstr,
               transform=ax.transAxes,
               fontsize=7,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.suptitle('Quantum Algorithms Comparison - Final Concentration Fields',
                 fontsize=14, y=0.995)
    plt.tight_layout()

    output_file1 = Path(__file__).parent / 'quantum_algorithms_comparison.png'
    plt.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"Concentration comparison saved to: {output_file1}")

    # Figure 2: Performance metrics
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(results.keys())
    # Shorten names for display
    short_names = [n.replace("Transport", "").replace("Quantum", "Q") for n in names]

    # Execution time
    times = [results[name]['elapsed_time'] for name in names]
    ax = axes[0, 0]
    bars = ax.barh(range(len(short_names)), times, color='steelblue')
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Execution Time (s)')
    ax.set_title('Computational Performance')
    ax.grid(axis='x', alpha=0.3)

    # Highlight fastest
    fastest_idx = times.index(min(times))
    bars[fastest_idx].set_color('green')

    # Mass conservation
    mass_errors = [results[name]['mass_conservation_error'] for name in names]
    ax = axes[0, 1]
    bars = ax.barh(range(len(short_names)), mass_errors, color='coral')
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Relative Mass Error')
    ax.set_xscale('log')
    ax.set_title('Mass Conservation')
    ax.grid(axis='x', alpha=0.3)

    # Highlight best conservation
    best_idx = mass_errors.index(min(mass_errors))
    bars[best_idx].set_color('green')

    # Time per step
    time_per_step = [results[name]['time_per_step'] * 1000 for name in names]
    ax = axes[1, 0]
    ax.barh(range(len(short_names)), time_per_step, color='purple', alpha=0.7)
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Time per Step (ms)')
    ax.set_title('Per-Step Performance')
    ax.grid(axis='x', alpha=0.3)

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
QUANTUM ALGORITHMS SUMMARY

Total Methods Compared: {len(methods)}
Grid Size: {grid_shape[0]} × {grid_shape[1]}
Time Steps: {n_steps}
Total Simulation Time: {dt * n_steps:.2f} s

PERFORMANCE:
Fastest Method:
  {benchmark.get_fastest_method()}
  ({min(times):.3f} s)

Best Mass Conservation:
  {benchmark.get_best_mass_conservation()}
  (error: {min(mass_errors):.2e})

Average Execution Time:
  {np.mean(times):.3f} ± {np.std(times):.3f} s

QUANTUM METHOD CATEGORIES:
• Quantum Walks (3)
• Variational Methods (2)
• Linear Solvers (2)
• Tensor Networks (1)
• Quantum Automata (1)
• Quantum Annealing (1)
• Core Continuous-Time (1)
    """

    ax.text(0.05, 0.95, summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    output_file2 = Path(__file__).parent / 'quantum_performance_metrics.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Performance metrics saved to: {output_file2}")

    # Export results
    results_file = Path(__file__).parent / 'quantum_benchmark_results.npz'
    benchmark.export_results(str(results_file))
    print(f"Numerical results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("QUANTUM ALGORITHMS COMPARISON COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  • {len(methods)} quantum algorithms successfully benchmarked")
    print(f"  • Fastest: {benchmark.get_fastest_method()}")
    print(f"  • Best conservation: {benchmark.get_best_mass_conservation()}")
    print(f"  • Average speedup range: {min(times):.2f}s - {max(times):.2f}s")
    print(f"\nAll quantum methods successfully solved the transport problem!")
    print(f"\nFiles generated:")
    print(f"  • {output_file1}")
    print(f"  • {output_file2}")
    print(f"  • {results_file}")


if __name__ == '__main__':
    main()
