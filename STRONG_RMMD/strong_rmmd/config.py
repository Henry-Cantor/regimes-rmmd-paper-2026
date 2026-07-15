"""
STRONG-RMMD Machine Configuration & Registry

Central repository for all 7 tokamak machine specifications, CDF variable mappings,
and physics parameters for consistent multi-machine data loading and normalization.

Machines: CMOD, NSTX, HL2A, D3D, KSTR, EAST, AUGD

This file is the SINGLE SOURCE OF TRUTH for machine specifications. All data loading,
normalization, and physics calculations reference this registry.
"""

import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch

# ============================================================================
# MACHINE SPECIFICATIONS (6 Tokamaks)
# ============================================================================

@dataclass
class MachineSpec:
    """Complete specification for a single tokamak."""
    name: str                          # Machine identifier
    major_radius: float                # R (m)
    minor_radius: float                # a (m)
    aspect_ratio: float                # A = R/a
    max_field: float                   # B_T (T)
    max_current: float                 # I_p (MA)
    rho_star_typical: float            # ρ* typical value
    n_horizontal_coils: int            # For geometry encoding
    n_vertical_coils: int              # For geometry encoding
    cdf_root_dir: str                  # Root directory for CDF files
    expected_n_shots: int              # Expected number of shots in database
    status: str                        # 'production', 'retired', 'planned'
    notes: str                         # Additional info
    
    @property
    def R_over_a(self) -> float:
        """Inverse aspect ratio."""
        return self.major_radius / self.minor_radius


# 6-MACHINE ENSEMBLE
MACHINES = {
    'NSTX': MachineSpec(
        name='NSTX',
        major_radius=0.85,
        minor_radius=0.67,
        aspect_ratio=1.27,
        max_field=0.55,
        max_current=1.0,
        rho_star_typical=1.5e-3,
        n_horizontal_coils=12,
        n_vertical_coils=12,
        cdf_root_dir='/path/to/nstx/transp',
        expected_n_shots=12000,
        status='production',
        notes='Low-aspect-ratio, strong-shaping. Baseline existing data.'
    ),
    'HL2A': MachineSpec(
        name='HL2A',
        major_radius=0.67,
        minor_radius=0.22,
        aspect_ratio=3.05,
        max_field=5.3,
        max_current=1.4,
        rho_star_typical=0.3e-3,
        n_horizontal_coils=24,
        n_vertical_coils=16,
        cdf_root_dir='/path/to/hl2a/transp',
        expected_n_shots=1025,
        status='production',
        notes='High-data compact tokamak replacement for CMOD.'
    ),
    'D3D': MachineSpec(
        name='D3D',
        major_radius=1.67,
        minor_radius=0.67,
        aspect_ratio=2.49,
        max_field=2.0,
        max_current=2.5,
        rho_star_typical=1.2e-3,
        n_horizontal_coils=24,
        n_vertical_coils=16,
        cdf_root_dir='/path/to/d3d/transp',
        expected_n_shots=15000,
        status='production',
        notes='Large mid-size divertor. High-performance H-mode data.'
    ),
    'KSTR': MachineSpec(
        name='KSTR',
        major_radius=1.8,
        minor_radius=0.5,
        aspect_ratio=3.6,
        max_field=3.5,
        max_current=0.8,
        rho_star_typical=0.8e-3,
        n_horizontal_coils=24,
        n_vertical_coils=12,
        cdf_root_dir='/path/to/kstr/transp',
        expected_n_shots=3000,
        status='production',
        notes='Superconducting K-shaped. Different magnet architecture.'
    ),
    'EAST': MachineSpec(
        name='EAST',
        major_radius=1.75,
        minor_radius=0.45,
        aspect_ratio=3.89,
        max_field=3.5,
        max_current=0.5,
        rho_star_typical=0.9e-3,
        n_horizontal_coils=16,
        n_vertical_coils=8,
        cdf_root_dir='/path/to/east/transp',
        expected_n_shots=2000,
        status='production',
        notes='Superconducting Chinese tokamak. Alternative geometry.'
    ),
    'ITER': MachineSpec(
        name='ITER',
        major_radius=6.2,
        minor_radius=2.0,
        aspect_ratio=3.1,
        max_field=5.3,
        max_current=15.0,
        rho_star_typical=0.15e-3,
        n_horizontal_coils=18,
        n_vertical_coils=6,
        cdf_root_dir='/path/to/iter/simulated',
        expected_n_shots=500,  # Simulated reference case
        status='planned',
        notes='ITER reference case (simulated TRANSP). Zero-shot prediction target.'
    ),
}


def get_machine_spec(machine_name: str) -> MachineSpec:
    """Return the machine specification for a canonical machine name."""
    return MACHINES[machine_name.upper()]

# ============================================================================
# CDF VARIABLE MAPPING (Machine-specific TRANSP output interpretation)
# ============================================================================

COMMON_CDF_VARIABLES = {
    # Profiles (radial grid)
    'T_e': 'TE',
    'T_i': 'TI',
    'n_e': 'NE',
    'q_profile': 'Q',
    'Z_eff': 'ZEFF',

    # Scalar quantities
    'P_NBI': 'PINJ',
    'P_OHMIC': 'PHEAT_IN',
    'B_T': 'BTDIA',
    'I_p': 'PCUR',
    'li': 'LI',
    'beta_N': 'BETAN',

    # Geometry / shape
    'elongation': 'ELONG',
    'triangularity': 'TRIANG',
    'R_major': 'RAXIS',
    'a_minor': 'RMAJB',
}

CDF_VARIABLES = {
    'nstx': dict(COMMON_CDF_VARIABLES),
    'hl2a': dict(COMMON_CDF_VARIABLES),
    'd3d': dict(COMMON_CDF_VARIABLES),
    'kstr': dict(COMMON_CDF_VARIABLES),
    'east': dict(COMMON_CDF_VARIABLES),
    'iter': dict(COMMON_CDF_VARIABLES),
}


def get_cdf_variable_map(machine_name: str) -> Dict[str, str]:
    """Return the CDF variable map for a machine."""
    return CDF_VARIABLES[machine_name.lower()]

# ============================================================================
# GYRO-BOHM NORMALIZATION UNIT SYSTEM
# ============================================================================

class GyroBoehmUnits:
    """
    Gyro-Bohm normalized units for consistent cross-machine comparison.
    
    Length scale: ρ_i (ion Larmor radius)
    Time scale: τ_GB = a / v_ti (gyro-Bohm time)
    Temperature scale: T_i (ion temperature, keV)
    Density scale: n_i (ion density, 10^19 m^-3)
    
    All frequencies normalized to: ω̃ = ω · τ_GB = ω · (a / v_ti)
    All wavelengths normalized to: k̃ = k · ρ_i
    """
    
    @staticmethod
    def compute_v_ti(T_i_keV: torch.Tensor, Z_eff: torch.Tensor) -> torch.Tensor:
        """
        Compute ion thermal velocity from temperature.
        
        v_ti = sqrt(2 * k_B * T_i / m_i)
        where m_i ≈ m_D (deuteron mass) ≈ 1876 m_e
        
        Args:
            T_i_keV: ion temperature in keV
            Z_eff: effective charge (used for ion mass correction if needed)
        
        Returns:
            v_ti in m/s
        """
        # Constants
        m_D_MeV = 938.3  # Deuteron rest energy in MeV/c^2

        # Convert keV to eV
        T_i_eV = T_i_keV * 1000

        # v_ti/c = sqrt(2 T_i / m_D c^2); * 299.79 (c in Mm/s) gives km/s, * 1000 -> m/s
        v_ti = torch.sqrt(2 * T_i_eV / m_D_MeV) * 299.79
        return v_ti * 1000  # m/s
    
    @staticmethod
    def compute_rho_i(B_T: float, T_i_keV: torch.Tensor) -> torch.Tensor:
        """
        Compute ion Larmor radius from field and temperature.
        
        ρ_i = m_i * v_ti / (e * B_T) in mm
        """
        m_i = 1876 * 9.109e-31  # Deuteron mass in kg
        e = 1.602e-19           # Elementary charge in C
        v_ti = torch.sqrt(2 * T_i_keV * 1e3 * 1.602e-19 / m_i)  # m/s
        rho_i = (m_i * v_ti) / (e * B_T)  # mm
        return rho_i * 1e3  # Convert to μm
    
    @staticmethod
    def normalize_frequency(omega: torch.Tensor, a: float, v_ti: torch.Tensor) -> torch.Tensor:
        """
        Normalize frequency to gyro-Bohm units.
        
        ω̃ = ω · τ_GB = ω · (a / v_ti)
        
        Args:
            omega: frequency (rad/s)
            a: minor radius (m)
            v_ti: ion thermal velocity (m/s)
        
        Returns:
            Normalized frequency (dimensionless)
        """
        tau_GB = a / v_ti  # Gyro-Bohm time (s)
        return omega * tau_GB


# ============================================================================
# TRAINING HYPERPARAMETERS (Machine-Independent)
# ============================================================================

TRAINING_CONFIG = {
    # Model architecture
    'K': 1024,                    # Koopman dimension
    'K_resonance': 256,           # Resonance kernel subspace dimension
    'n_harmonics': 4,             # Number of resonance harmonics
    'embedding_dim': 128,         # Geometry embedding dimension
    
    # Training
    'epochs': 300,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'gradient_clip_norm': 1.0,
    
    # Curriculum
    'T_predict_min': 10,          # Start horizon (steps)
    'T_predict_max': 200,         # Final horizon (steps)
    'T_ramp_epoch': 150,          # Epoch to reach T_max
    
    # Loss annealing (sigmoid ramps)
    'loss_schedule': {
        'L_energy': {'t_start': 0, 't_width': 50, 'lambda_final': 0.10},
        'L_dissip': {'t_start': 0, 't_width': 50, 'lambda_final': 0.10},
        'L_snt': {'t_start': 20, 't_width': 65, 'lambda_final': 0.05},
        'L_jarzy': {'t_start': 50, 't_width': 75, 'lambda_final': 0.05},
        'L_D_res_sparse': {'constant': 0.01},
        'L_delta_S': {'constant': 0.01},
        'L_sut_align': {'t_start': 100, 't_width': 50, 'lambda_final': 0.05},
        'L_koopman': {'constant': 0.10},
        'L_physics': {'constant': 0.10},
    },
    
    # Optimization
    'optimizer': 'Adam',
    'scheduler': 'CosineAnnealing',
    'early_stopping_patience': 100,
    'val_check_interval': 10,  # epochs
    'checkpoint_interval': 50,  # epochs
    
    # Regularization
    'noise_sigma_training': 0.1,  # Gaussian noise during training
    'teacher_forcing_ratio': 0.5,  # Curriculum on teacher forcing
    'dropout_rate': 0.1,
    
    # Mixed precision
    'use_amp': True,  # Automatic mixed precision
    'amp_dtype': 'float16',
    
    # Validation/test
    'T_eval': [1, 10, 50, 100, 200],  # Evaluate at these horizons
    'split_train_val': 0.8,
    'split_val_test': 0.5,
}

# ============================================================================
# PATHS & I/O
# ============================================================================

PATHS = {
    'data_root': '/scratch/gpfs/USER/strong_rmmd_data',
    'checkpoint_dir': '/scratch/gpfs/USER/strong_rmmd_checkpoints',
    'results_dir': '/scratch/gpfs/USER/strong_rmmd_results',
    'logs_dir': '/scratch/gpfs/USER/strong_rmmd_logs',
    'plots_dir': '/scratch/gpfs/USER/strong_rmmd_plots',
}

# ============================================================================
# VALIDATION GATES
# ============================================================================

VALIDATION_GATES = {
    'GATE_1_D_RES_CORRELATION': {
        'description': 'D_res measurement correlation with NRMSE growth',
        'condition': 'Pearson r > 0.3 on all 6 machines simultaneously',
        'action_pass': 'Proceed to model training',
        'action_fail': 'Halt. Investigate TRANSP noise floor, data quality.',
    },
    'GATE_2_LORENZ96_GIT': {
        'description': 'GIT theorem validation on Lorenz-96',
        'condition': 'KL divergence ∝ T·‖D_res‖² empirically (R² > 0.95)',
        'action_pass': 'Proceed to multi-machine training',
        'action_fail': 'Halt. Reconsider GIT formulation or proof strategy.',
    },
    'GATE_3_NRMSE_IMPROVEMENT': {
        'description': 'RMMD accuracy vs DGKNet baseline on all 6 machines',
        'condition': 'RMMD > DGKNet by ≥5% at T=100 on all machines',
        'action_pass': 'Proceed to evaluation and ablations',
        'action_fail': 'Check hyperparameters, loss balance. Investigate.',
    },
    'GATE_4_SUT_UNIVERSALITY': {
        'description': 'Spectral Universality Theorem validation',
        'condition': 'Top N₀=K/4 modes have σ/μ < 0.2 cross-machine',
        'action_pass': 'Proceed to Nature Physics manuscript preparation',
        'action_fail': 'Pivot to Nature Machine Intelligence (methods paper)',
    },
    'GATE_5_STRONG_BOUND': {
        'description': 'STRONG theorem bound fitting quality',
        'condition': 'Cross-validation R² > 0.8 on leave-one-out machines',
        'action_pass': 'Use ITER prediction as Nature Physics headline',
        'action_fail': 'Refine bound decomposition or single-machine focus.',
    },
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_machine(machine_name: str) -> MachineSpec:
    """Retrieve machine specification by name."""
    if machine_name not in MACHINES:
        available = ', '.join(MACHINES.keys())
        raise ValueError(f"Unknown machine '{machine_name}'. Available: {available}")
    return MACHINES[machine_name]

def get_all_machines() -> List[str]:
    """Return list of all available machines."""
    return list(MACHINES.keys())

def get_machine_pair_similarity(m1: str, m2: str) -> Dict[str, float]:
    """Compute similarity metrics between two machines for SUT validation."""
    spec1 = get_machine(m1)
    spec2 = get_machine(m2)
    
    return {
        'aspect_ratio_diff': abs(spec1.aspect_ratio - spec2.aspect_ratio),
        'rho_star_ratio': spec2.rho_star_typical / spec1.rho_star_typical,
        'radius_ratio': spec2.major_radius / spec1.major_radius,
        'field_ratio': spec2.max_field / spec1.max_field,
    }

def print_machine_registry():
    """Pretty-print machine registry."""
    print("\n" + "="*80)
    print("STRONG-RMMD: 6-Machine Tokamak Ensemble")
    print("="*80)
    for machine_name in get_all_machines():
        spec = get_machine(machine_name)
        print(f"\n{spec.name.upper()}")
        print(f"  R/a:     {spec.major_radius:.2f} / {spec.minor_radius:.2f} m (A={spec.aspect_ratio:.2f})")
        print(f"  B_max:   {spec.max_field:.1f} T,  I_max: {spec.max_current:.1f} MA")
        print(f"  Status:  {spec.status}  ({spec.expected_n_shots} shots)")
        print(f"  Note:    {spec.notes}")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    print_machine_registry()
