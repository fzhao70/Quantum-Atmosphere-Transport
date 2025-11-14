"""
Base Transport Interface

Abstract base class defining the interface for all transport methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Dict


class BaseTransport(ABC):
    """
    Abstract base class for all atmospheric tracer transport methods.

    All transport schemes (quantum, classical, hybrid, etc.) should
    inherit from this class and implement the required methods.
    """

    def __init__(self, config):
        """
        Initialize the transport scheme.

        Parameters
        ----------
        config : TransportConfig
            Configuration object
        """
        self.config = config
        self.grid_shape: Optional[Tuple[int, ...]] = None
        self.concentration: Optional[np.ndarray] = None

    @abstractmethod
    def initialize(self,
                   initial_concentration: np.ndarray,
                   grid_shape: Tuple[int, ...]) -> None:
        """
        Initialize the transport scheme with initial conditions.

        Parameters
        ----------
        initial_concentration : np.ndarray
            Initial tracer concentration field
        grid_shape : tuple
            Shape of the computational grid
        """
        pass

    @abstractmethod
    def step(self,
             dt: float,
             velocity_field: np.ndarray,
             diffusion_coeff: float,
             source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance the transport scheme by one time step.

        Parameters
        ----------
        dt : float
            Time step size
        velocity_field : np.ndarray
            Wind velocity field
        diffusion_coeff : float
            Diffusion coefficient
        source_term : np.ndarray, optional
            Source/sink term

        Returns
        -------
        np.ndarray
            Updated tracer concentration field
        """
        pass

    @abstractmethod
    def get_concentration(self) -> np.ndarray:
        """
        Get the current tracer concentration field.

        Returns
        -------
        np.ndarray
            Current concentration field
        """
        pass

    def get_method_name(self) -> str:
        """
        Get the name of the transport method.

        Returns
        -------
        str
            Method name
        """
        return self.__class__.__name__

    def get_statistics(self) -> Dict[str, float]:
        """
        Get method-specific statistics.

        Returns
        -------
        dict
            Statistics dictionary
        """
        if self.concentration is None:
            return {}

        return {
            'total_mass': float(np.sum(self.concentration)),
            'max_concentration': float(np.max(self.concentration)),
            'min_concentration': float(np.min(self.concentration)),
            'mean_concentration': float(np.mean(self.concentration)),
            'std_concentration': float(np.std(self.concentration)),
        }
