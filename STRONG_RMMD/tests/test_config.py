"""
Unit Tests for STRONG-RMMD Components

Minimal test suite to validate each module before integration.
Run before proceeding to next phase.

pytest usage:
    pytest tests/ -v  # Run all
    pytest tests/test_*.py -v  # Run specific module
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strong_rmmd import config

# ============================================================================
# TESTS: config.py
# ============================================================================

class TestMachineRegistry:
    """Test machine configuration loading."""
    
    def test_all_machines_loadable(self):
        """Verify all 6 machines load without error."""
        for machine_name in config.get_all_machines():
            spec = config.get_machine(machine_name)
            assert spec.name == machine_name
            assert spec.major_radius > 0
            assert spec.minor_radius > 0
            assert spec.aspect_ratio > 0
    
    def test_machine_names_correct(self):
        """Check machine names match expected list."""
        expected = {'NSTX', 'HL2A', 'D3D', 'KSTR', 'EAST', 'ITER'}
        actual = set(config.get_all_machines())
        assert actual == expected, f"Expected {expected}, got {actual}"
    
    def test_machine_dimensions(self):
        """Verify physical dimensions are reasonable."""
        nstx = config.get_machine('NSTX')
        assert 0.8 < nstx.major_radius < 0.9
        assert 0.6 < nstx.minor_radius < 0.7
        
        iter_spec = config.get_machine('ITER')
        assert 6.0 < iter_spec.major_radius < 6.5
        assert 1.9 < iter_spec.minor_radius < 2.1
        
        # Check size range
        assert iter_spec.major_radius > 5 * nstx.major_radius  # ITER much bigger
    
    def test_machine_pair_similarity(self):
        """Test cross-machine similarity metrics."""
        similarity = config.get_machine_pair_similarity('NSTX', 'HL2A')
        
        # Both should be in same general range but different machines
        assert isinstance(similarity, dict)
        assert all(k in similarity for k in ['aspect_ratio_diff', 'rho_star_ratio', 'radius_ratio', 'field_ratio'])
        assert similarity['radius_ratio'] > 0.0
    
    def test_unknown_machine_raises_error(self):
        """Verify error on unknown machine name."""
        with pytest.raises(ValueError, match="Unknown machine"):
            config.get_machine('INVALID_MACHINE')


class TestGyroBoehmUnits:
    """Test gyro-Bohm normalization utilities."""
    
    def test_ion_thermal_velocity(self):
        """Test v_ti computation."""
        T_i = torch.as_tensor([1.0, 10.0])  # keV
        Z_eff = torch.as_tensor([1.0, 1.0])
        
        v_ti = config.GyroBoehmUnits.compute_v_ti(T_i, Z_eff)
        
        # v_ti should increase with temperature
        assert v_ti[1] > v_ti[0]
        # compute_v_ti returns m/s: 1-10 keV deuterium -> ~4.4e5-1.4e6 m/s (~440-1380 km/s)
        assert torch.all(v_ti > 1e5) and torch.all(v_ti < 3e6)
    
    def test_frequency_normalization(self):
        """Test frequency normalization to GB units."""
        omega = torch.as_tensor([1e4])  # rad/s
        a = 1.0  # m
        v_ti = torch.as_tensor([1e6])  # m/s
        
        omega_normalized = config.GyroBoehmUnits.normalize_frequency(omega, a, v_ti)
        
        # Normalized frequency should be dimensionless and <1 for reasonable plasma
        assert omega_normalized.dim() == 1
        assert omega_normalized[0] > 0


class TestTrainingConfig:
    """Test training hyperparameters."""
    
    def test_training_config_valid(self):
        """Verify training config is sensible."""
        cfg = config.TRAINING_CONFIG
        
        # Check key fields exist
        assert 'K' in cfg and cfg['K'] == 1024
        assert 'epochs' in cfg and cfg['epochs'] > 0
        assert 'batch_size' in cfg and cfg['batch_size'] > 0
        
        # Loss weights sum to ~1-2
        loss_weights = [v.get('constant', v.get('lambda_final', 0)) 
                       for v in cfg['loss_schedule'].values()]
        assert sum(loss_weights) > 0.1  # Should have meaningful losses
    
    def test_curriculum_config(self):
        """Verify curriculum schedule is valid."""
        cfg = config.TRAINING_CONFIG
        assert cfg['T_predict_min'] < cfg['T_predict_max']
        assert cfg['T_ramp_epoch'] > 0


class TestPaths:
    """Test path configuration."""
    
    def test_paths_defined(self):
        """Verify all required paths defined."""
        required_paths = ['data_root', 'checkpoint_dir', 'results_dir', 'logs_dir', 'plots_dir']
        for path_key in required_paths:
            assert path_key in config.PATHS
            assert config.PATHS[path_key].startswith('/scratch') or config.PATHS[path_key].startswith('/')


class TestValidationGates:
    """Test validation gate definitions."""
    
    def test_all_gates_defined(self):
        """Verify all 5 gates defined with proper structure."""
        gates = config.VALIDATION_GATES
        
        expected_gates = {'GATE_1_D_RES_CORRELATION', 'GATE_2_LORENZ96_GIT',
                         'GATE_3_NRMSE_IMPROVEMENT', 'GATE_4_SUT_UNIVERSALITY',
                         'GATE_5_STRONG_BOUND'}
        
        assert set(gates.keys()) == expected_gates
        
        # Check each gate has proper structure
        for gate_name, gate_config in gates.items():
            assert 'description' in gate_config
            assert 'condition' in gate_config
            assert 'action_pass' in gate_config
            assert 'action_fail' in gate_config


# ============================================================================
# RUN TESTS (if executed directly)
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
