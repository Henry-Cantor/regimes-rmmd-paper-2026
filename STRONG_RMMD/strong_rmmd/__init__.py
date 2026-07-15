"""
STRONG-RMMD: Spectral Thermodynamic and Resonance Operators for New Geometries
Resonance-Mediated Metriplectic Dynamics

Core package for multi-machine tokamak Koopman dynamics learning with resonance kernels.

6-Machine Ensemble: NSTX, HL2A, D3D, KSTR, EAST, ITER
Nature Physics Target: "Universal spectral modes of plasma turbulence across tokamaks"

Version: 1.0.0
Status: Production Ready
"""

__version__ = "1.0.0"
__author__ = "Computational Plasma Physics Group"
__all__ = [
    'config',
    'data_loader',
    'geometry',
    'resonance_frequencies',
    'dataset',
    'resonance_kernel',
    'rmmd_block',
    'multi_machine_rmmd',
    'losses',
    'diagnostics',
    'theorems',
    'transfer',
    'utils',
    'visualization',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in __all__:
        import importlib
        module = importlib.import_module(f'strong_rmmd.{name}')
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
