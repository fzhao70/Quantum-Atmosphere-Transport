"""
Unit tests for configuration.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantum_atmosphere_transport import TransportConfig


class TestTransportConfig:
    """Test transport configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TransportConfig()

        assert config.grid_spacing == 1000.0
        assert config.boundary_condition == 'periodic'
        assert config.advection_scheme == 'quantum_semi_lagrangian'
        assert config.interpolation_order == 3
        assert config.apply_decoherence is True
        assert config.decoherence_rate == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = TransportConfig(
            grid_spacing=500.0,
            boundary_condition='fixed',
            advection_scheme='quantum_spectral',
            decoherence_rate=0.05
        )

        assert config.grid_spacing == 500.0
        assert config.boundary_condition == 'fixed'
        assert config.advection_scheme == 'quantum_spectral'
        assert config.decoherence_rate == 0.05

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = TransportConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_grid_spacing(self):
        """Test validation with invalid grid spacing."""
        config = TransportConfig(grid_spacing=-1.0)

        with pytest.raises(ValueError, match="grid_spacing must be positive"):
            config.validate()

    def test_validate_invalid_interpolation_order(self):
        """Test validation with invalid interpolation order."""
        config = TransportConfig(interpolation_order=10)

        with pytest.raises(ValueError, match="interpolation_order must be between"):
            config.validate()

    def test_validate_invalid_decoherence_rate(self):
        """Test validation with invalid decoherence rate."""
        config = TransportConfig(decoherence_rate=1.5)

        with pytest.raises(ValueError, match="decoherence_rate must be between"):
            config.validate()

    def test_validate_invalid_trotter_steps(self):
        """Test validation with invalid Trotter steps."""
        config = TransportConfig(trotter_steps=0)

        with pytest.raises(ValueError, match="trotter_steps must be at least 1"):
            config.validate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TransportConfig(grid_spacing=500.0)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['grid_spacing'] == 500.0
        assert 'boundary_condition' in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'grid_spacing': 500.0,
            'boundary_condition': 'fixed',
            'advection_scheme': 'quantum_spectral',
        }

        config = TransportConfig.from_dict(config_dict)

        assert config.grid_spacing == 500.0
        assert config.boundary_condition == 'fixed'
        assert config.advection_scheme == 'quantum_spectral'

    def test_repr(self):
        """Test string representation."""
        config = TransportConfig()
        repr_str = repr(config)

        assert 'TransportConfig' in repr_str
        assert 'grid_spacing' in repr_str
