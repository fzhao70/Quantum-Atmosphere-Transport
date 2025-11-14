"""
Simple Quantum Atmospheric Transport Example

Demonstrates basic usage of the quantum tracer transport scheme
for a 2D atmospheric domain.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantum_atmosphere_transport import (
    QuantumTracerTransport,
    TransportConfig
)


def create_initial_condition(nx: int, ny: int) -> np.ndarray:
    """
    Create a Gaussian plume as initial condition.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions

    Returns
    -------
    np.ndarray
        Initial concentration field
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Gaussian plume centered at (0.2, 0.5)
    x0, y0 = 0.2, 0.5
    sigma = 0.05

    concentration = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    concentration /= np.sum(concentration)  # Normalize

    return concentration


def create_velocity_field(nx: int, ny: int) -> np.ndarray:
    """
    Create a simple velocity field (uniform wind).

    Parameters
    ----------
    nx, ny : int
        Grid dimensions

    Returns
    -------
    np.ndarray
        Velocity field with shape (nx, ny, 2)
    """
    velocity = np.zeros((nx, ny, 2))

    # Uniform wind: 5 m/s eastward, 2 m/s northward
    velocity[..., 0] = 5.0  # u component (eastward)
    velocity[..., 1] = 2.0  # v component (northward)

    return velocity


def run_simulation():
    """Run the quantum transport simulation."""

    print("=" * 60)
    print("Quantum Atmospheric Transport Simulation")
    print("=" * 60)

    # Grid parameters
    nx, ny = 50, 50
    grid_shape = (nx, ny)

    # Create configuration
    config = TransportConfig(
        grid_spacing=1000.0,  # 1 km
        boundary_condition='periodic',
        advection_scheme='quantum_semi_lagrangian',
        apply_decoherence=True,
        decoherence_rate=0.01,
    )

    print("\nConfiguration:")
    print(config)

    # Initialize transport scheme
    print("\n[1/6] Initializing quantum transport scheme...")
    transport = QuantumTracerTransport(config)

    # Create initial condition
    print("[2/6] Creating initial condition (Gaussian plume)...")
    initial_concentration = create_initial_condition(nx, ny)
    transport.initialize(initial_concentration, grid_shape)

    # Create velocity field
    print("[3/6] Setting up velocity field (uniform wind)...")
    velocity_field = create_velocity_field(nx, ny)

    # Simulation parameters
    dt = 60.0  # 60 seconds
    n_steps = 100
    save_interval = 20

    # Diffusion coefficient
    diffusion_coeff = 100.0  # m²/s (typical for atmospheric turbulence)

    print(f"\n[4/6] Running simulation...")
    print(f"  Time step: {dt} s")
    print(f"  Number of steps: {n_steps}")
    print(f"  Total time: {dt * n_steps / 60:.1f} minutes")
    print(f"  Diffusion coefficient: {diffusion_coeff} m²/s")

    # Storage for results
    saved_times = []
    saved_concentrations = []
    saved_quantum_stats = []

    # Time integration
    for step in range(n_steps + 1):
        if step % save_interval == 0:
            # Save current state
            concentration = transport.get_concentration()
            quantum_stats = transport.get_quantum_statistics()

            saved_times.append(step * dt / 60)  # Convert to minutes
            saved_concentrations.append(concentration.copy())
            saved_quantum_stats.append(quantum_stats)

            print(f"  Step {step:4d} / {n_steps} "
                  f"(t = {step * dt / 60:6.1f} min) - "
                  f"Coherence: {quantum_stats.get('coherence', 0):.6f}, "
                  f"Entanglement: {quantum_stats.get('entanglement', 0):.6f}")

        if step < n_steps:
            # Advance one time step
            transport.step(dt, velocity_field, diffusion_coeff)

    print("\n[5/6] Generating visualization...")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Quantum Atmospheric Transport - Tracer Evolution', fontsize=16)

    # Plot concentration at different times
    for idx, ax in enumerate(axes.flat):
        if idx < len(saved_concentrations):
            im = ax.imshow(
                saved_concentrations[idx].T,
                origin='lower',
                extent=[0, nx, 0, ny],
                cmap='hot',
                vmin=0,
                vmax=np.max(saved_concentrations[0])
            )
            ax.set_title(f't = {saved_times[idx]:.1f} min')
            ax.set_xlabel('x (grid points)')
            ax.set_ylabel('y (grid points)')
            plt.colorbar(im, ax=ax, label='Concentration')

            # Add quantum statistics
            stats = saved_quantum_stats[idx]
            textstr = (f"Coherence: {stats.get('coherence', 0):.4f}\n"
                      f"Entanglement: {stats.get('entanglement', 0):.2f}")
            ax.text(0.02, 0.98, textstr,
                   transform=ax.transAxes,
                   fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent / 'transport_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[6/6] Visualization saved to: {output_file}")

    # Print quantum statistics evolution
    print("\n" + "=" * 60)
    print("Quantum Statistics Evolution")
    print("=" * 60)
    print(f"{'Time (min)':>12} {'Coherence':>12} {'Entanglement':>15} {'Phase Var':>12}")
    print("-" * 60)
    for time, stats in zip(saved_times, saved_quantum_stats):
        print(f"{time:12.1f} "
              f"{stats.get('coherence', 0):12.6f} "
              f"{stats.get('entanglement', 0):15.6f} "
              f"{stats.get('phase_variance', 0):12.6f}")

    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    run_simulation()
