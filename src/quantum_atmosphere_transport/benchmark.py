"""
Benchmarking and Comparison Framework

Tools for comparing different transport methods.
"""

import numpy as np
import time
from typing import List, Dict, Optional
from .base_transport import BaseTransport


class TransportBenchmark:
    """
    Benchmark multiple transport methods on the same problem.

    Compares accuracy, conservation, and performance.
    """

    def __init__(self,
                 methods: List[BaseTransport],
                 method_names: Optional[List[str]] = None):
        """
        Initialize benchmark.

        Parameters
        ----------
        methods : list of BaseTransport
            List of transport methods to compare
        method_names : list of str, optional
            Names for each method (auto-generated if not provided)
        """
        self.methods = methods
        if method_names is None:
            self.method_names = [m.get_method_name() for m in methods]
        else:
            self.method_names = method_names

        self.results: Dict = {}

    def run(self,
            initial_concentration: np.ndarray,
            grid_shape: tuple,
            velocity_field: np.ndarray,
            diffusion_coeff: float,
            dt: float,
            n_steps: int,
            reference_solution: Optional[np.ndarray] = None,
            source_term: Optional[np.ndarray] = None) -> Dict:
        """
        Run benchmark for all methods.

        Parameters
        ----------
        initial_concentration : np.ndarray
            Initial concentration field
        grid_shape : tuple
            Grid shape
        velocity_field : np.ndarray
            Velocity field
        diffusion_coeff : float
            Diffusion coefficient
        dt : float
            Time step
        n_steps : int
            Number of time steps
        reference_solution : np.ndarray, optional
            Reference solution for error calculation
        source_term : np.ndarray, optional
            Source/sink term

        Returns
        -------
        dict
            Benchmark results
        """
        initial_mass = np.sum(initial_concentration)

        for i, (method, name) in enumerate(zip(self.methods, self.method_names)):
            print(f"Running method {i+1}/{len(self.methods)}: {name}")

            # Initialize
            method.initialize(initial_concentration.copy(), grid_shape)

            # Time the execution
            start_time = time.time()

            # Run simulation
            for step in range(n_steps):
                method.step(dt, velocity_field, diffusion_coeff, source_term)

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Get final concentration
            final_concentration = method.get_concentration()
            final_mass = np.sum(final_concentration)

            # Calculate metrics
            mass_conservation_error = abs(final_mass - initial_mass) / initial_mass

            # Calculate error if reference solution provided
            if reference_solution is not None:
                l1_error = np.mean(np.abs(final_concentration - reference_solution))
                l2_error = np.sqrt(np.mean((final_concentration - reference_solution) ** 2))
                max_error = np.max(np.abs(final_concentration - reference_solution))
            else:
                l1_error = None
                l2_error = None
                max_error = None

            # Get method-specific statistics
            method_stats = method.get_statistics()

            # Store results
            self.results[name] = {
                'elapsed_time': elapsed_time,
                'time_per_step': elapsed_time / n_steps,
                'final_concentration': final_concentration,
                'final_mass': final_mass,
                'mass_conservation_error': mass_conservation_error,
                'l1_error': l1_error,
                'l2_error': l2_error,
                'max_error': max_error,
                'method_statistics': method_stats,
            }

        return self.results

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        print(f"\n{'Method':<40} {'Time (s)':<12} {'Time/Step (ms)':<15} {'Mass Error':<12}")
        print("-" * 80)

        for name, result in self.results.items():
            print(f"{name:<40} "
                  f"{result['elapsed_time']:>11.4f} "
                  f"{result['time_per_step']*1000:>14.4f} "
                  f"{result['mass_conservation_error']:>11.2e}")

        if any(r['l2_error'] is not None for r in self.results.values()):
            print(f"\n{'Method':<40} {'L1 Error':<12} {'L2 Error':<12} {'Max Error':<12}")
            print("-" * 80)

            for name, result in self.results.items():
                if result['l2_error'] is not None:
                    print(f"{name:<40} "
                          f"{result['l1_error']:>11.2e} "
                          f"{result['l2_error']:>11.2e} "
                          f"{result['max_error']:>11.2e}")

        print("=" * 80)

    def get_fastest_method(self) -> str:
        """Get the name of the fastest method."""
        return min(self.results.items(), key=lambda x: x[1]['elapsed_time'])[0]

    def get_most_accurate_method(self) -> Optional[str]:
        """Get the name of the most accurate method (lowest L2 error)."""
        valid_results = {name: res for name, res in self.results.items()
                        if res['l2_error'] is not None}

        if not valid_results:
            return None

        return min(valid_results.items(), key=lambda x: x[1]['l2_error'])[0]

    def get_best_mass_conservation(self) -> str:
        """Get the method with best mass conservation."""
        return min(self.results.items(),
                  key=lambda x: x[1]['mass_conservation_error'])[0]

    def compare_methods(self, metric: str = 'elapsed_time') -> Dict[str, float]:
        """
        Compare methods by a specific metric.

        Parameters
        ----------
        metric : str
            Metric to compare ('elapsed_time', 'mass_conservation_error', 'l2_error')

        Returns
        -------
        dict
            Dictionary mapping method names to metric values
        """
        return {name: result[metric] for name, result in self.results.items()
                if metric in result and result[metric] is not None}

    def export_results(self, filename: str) -> None:
        """
        Export results to a file.

        Parameters
        ----------
        filename : str
            Output filename (supports .npz format)
        """
        if filename.endswith('.npz'):
            # Export as numpy archive
            data = {}
            for name, result in self.results.items():
                safe_name = name.replace(' ', '_')
                data[f'{safe_name}_concentration'] = result['final_concentration']
                data[f'{safe_name}_time'] = result['elapsed_time']
                data[f'{safe_name}_mass_error'] = result['mass_conservation_error']

            np.savez(filename, **data)
            print(f"Results exported to {filename}")
        else:
            print(f"Unsupported format. Use .npz extension.")


def create_test_problem(grid_shape: tuple,
                       problem_type: str = 'gaussian') -> dict:
    """
    Create a standard test problem for benchmarking.

    Parameters
    ----------
    grid_shape : tuple
        Shape of computational grid
    problem_type : str
        Type of test problem: 'gaussian', 'step', 'rotating'

    Returns
    -------
    dict
        Dictionary containing initial condition and parameters
    """
    ndim = len(grid_shape)

    if problem_type == 'gaussian':
        # Gaussian plume
        if ndim == 1:
            x = np.linspace(0, 1, grid_shape[0])
            initial = np.exp(-((x - 0.2) ** 2) / 0.01)
        elif ndim == 2:
            x = np.linspace(0, 1, grid_shape[0])
            y = np.linspace(0, 1, grid_shape[1])
            X, Y = np.meshgrid(x, y, indexing='ij')
            initial = np.exp(-((X - 0.2) ** 2 + (Y - 0.5) ** 2) / 0.01)
        elif ndim == 3:
            x = np.linspace(0, 1, grid_shape[0])
            y = np.linspace(0, 1, grid_shape[1])
            z = np.linspace(0, 1, grid_shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            initial = np.exp(-((X - 0.2) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) / 0.01)

        # Uniform velocity field
        velocity = np.zeros(grid_shape + (ndim,))
        velocity[..., 0] = 1.0  # Unit velocity in first direction

    elif problem_type == 'step':
        # Step function
        initial = np.zeros(grid_shape)
        if ndim == 1:
            initial[:grid_shape[0]//4] = 1.0
        elif ndim == 2:
            initial[:grid_shape[0]//4, :] = 1.0
        elif ndim == 3:
            initial[:grid_shape[0]//4, :, :] = 1.0

        # Uniform velocity
        velocity = np.zeros(grid_shape + (ndim,))
        velocity[..., 0] = 1.0

    elif problem_type == 'rotating':
        # Rotating flow (2D only)
        if ndim != 2:
            raise ValueError("Rotating flow only available for 2D")

        x = np.linspace(0, 1, grid_shape[0])
        y = np.linspace(0, 1, grid_shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Gaussian initial condition
        initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.7) ** 2) / 0.01)

        # Circular velocity field
        velocity = np.zeros(grid_shape + (2,))
        velocity[..., 0] = -(Y - 0.5)  # u = -y
        velocity[..., 1] = (X - 0.5)   # v = x

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    return {
        'initial_concentration': initial,
        'velocity_field': velocity,
        'grid_shape': grid_shape,
    }
