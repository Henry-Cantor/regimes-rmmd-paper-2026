"""
Phase 0: Data Pipeline for PPPL Tokamak CDF Data

This module implements the complete data extraction and preprocessing pipeline
for the experimental plan. It reads TRANSP CDF files and constructs
the full t=0 state vector as specified in Section 2 of the experimental plan.

Key responsibilities:
1. Extract complete t=0 state vector from TRANSP CDF files
2. Construct geometry tensor from Fourier coefficients (RMC, RMS, YMC, YMS)
3. Compute derived physics quantities (η_i, Zeff, q, etc.)
4. Create PyTorch Dataset for training
"""

import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
import logging
import copy
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _infer_machine_from_path(p: str) -> str:
    #Infer machine name from a CDF path string.
    parts = Path(p).parts
    for i, part in enumerate(parts):
        if part.lower() == 'cdf' and i + 1 < len(parts):
            return parts[i + 1]
    return Path(p).parent.name


# Canonical temporal split used across all phases.
# End indices are inclusive.
TRAIN_TIMESTEPS = (0, 200)
VAL_TIMESTEPS = (201, 225)
TEST_TIMESTEPS = (226, 251)


# ==============================================================================
# SECTION 1: Variable Definitions and Constants
# ==============================================================================

@dataclass
class PlasmaVariables:
    """
    Complete specification of all variables extracted from TRANSP CDF files.
    Organized by physics category as per Section 2.2 of the experimental plan.
    """
    
    # KINETIC PROFILES (on X grid, dimension ~100 radial points)
    # These are the fundamental kinetic profiles that determine plasma transport
    KINETIC_PROFILES = {
        'NE':     'Electron density profile [m^-3]',
        'NI':     'Ion density profile [m^-3] — PRIMARY TARGET',
        'NH':     'Hydrogen ion density [m^-3]',
        'TE':     'Electron temperature [keV]',
        'TI':     'Ion temperature [keV]',
        'PPLAS':  'Total pressure profile [Pa]',
    }
    
    # STATIC MACHINE GEOMETRY
    # Use the limiter contour (RLIM, YLIM) as the pre-shot geometry input.
    LIMITER_GEOMETRY = ['RLIM', 'YLIM']

    # PLASMA BOUNDARY GEOMETRY (time-dependent Fourier coefficients)
    PLASMA_GEOMETRY_COSINE = [f'RMC{i:02d}' for i in range(17)] + [f'RMS{i:02d}' for i in range(1, 17)]
    PLASMA_GEOMETRY_Z_COSINE = [f'YMC{i:02d}' for i in range(17)] + [f'YMS{i:02d}' for i in range(1, 17)]

    # PRE-SHOT INPUTS THAT ARE SAFER THAN DERIVED STATE QUANTITIES
    # Scalar controls / initial run conditions that are known before rollout.
    PRE_SHOT_SCALARS = {
        'NSHOT': 'Shot number',
        'TINIT': 'Start time [s]',
        'FTIME': 'End time [s]',
        'TGRID1': 'Scalar input grid spacing [s]',
        'TGRID2': 'Profile input grid spacing [s]',
        'DTMINT': 'Min transport timestep [s]',
        'DTMAXT': 'Max transport timestep [s]',
        'DTINIT': 'Initial transport timestep [s]',
        'DTMINB': 'Min magnetic timestep [s]',
        'DTMAXB': 'Max magnetic timestep [s]',
        'DTINIB': 'Initial magnetic timestep [s]',
        'DTMAXG': 'Max equilibrium timestep [s]',
        'DTTOR': 'ECE source timestep [s]',
        'DTICRF': 'ICRF source timestep [s]',
        'DTLH': 'Lower hybrid timestep [s]',
        'DTBEAM': 'Neutral beam timestep [s]',
        'NZONES': 'Radial zones',
        'SEDIT': 'Profile output spacing [s]',
        'STEDIT': 'Scalar output spacing [s]',
        'SELOUT': 'Output variable selection',
        'SELAVG': 'Averaged output variable selection',
        'OUTTIM': 'ACFILE output times',
        'AVGTIM': 'ACFILE averaging time [s]',
        'NLBCCW': 'Toroidal field counter-clockwise flag',
        'NLJCCW': 'Current-drive sign convention flag',
        'NBI_PSERVE': 'Parallel NUBEAM flag',
        'NTORIC_PSERVE': 'Parallel TORIC flag',
        'NGENRAY_PSERVE': 'Parallel GENRAY flag',
        'NCQL_PSERVE': 'Parallel CQL3D flag',
        'NPTCL_PSERVE': 'Parallel PT-SOLVER flag',
        'NECOM': 'Electron density input [cm^-3]',
        'TECOM': 'Electron temperature input [eV]',
        'TICOM': 'Ion temperature input [eV]',
        'PHEAT_IN': 'Total heating power input [W]',
        'PFI0': 'Fast ion source power input [W]',
        'PSI0': 'Poloidal flux on axis input [Wb/rad]',
        'PINJ': 'Injected NBI power [W]',
        'BTDIA': 'Toroidal field [T]',
        'PCUR': 'Plasma current [A]',
        'RAXIS': 'Magnetic axis R [m]',
        'YAXIS': 'Magnetic axis Z [m]',
        'RMAJB': 'Major radius / geometric scale [m]',
        'PHEAT': 'Total heating power [W]',
        'PEDGE': 'Edge/scrape-off power [W]',
        'POHC': 'Ohmic heating [W]',
        'PRFFI': 'RF power to fast ions [W]',
        'P0FIN': 'Neutral influx power [W]',
        'P0RFL': 'Neutral reflected power [W]',
        'P0CXT': 'Neutral CX power [W]',
        'P0INZ': 'Neutral ionization power [W]',
        'P0ESC': 'Neutral escaped power [W]',
        'P0BAL': 'Neutral power balance check [W]',
        'PL2H': 'L-H power condition [W]',
        'PL2HTOT': 'Total heating power for L-H [W]',
        'PL2HREQ': 'Required L-H power threshold [W]',
        'MODEEDG': 'Electron temperature edge mode',
        'MODIEDG': 'Ion temperature edge mode',
        'MODNEDG': 'Density edge mode',
        'MODOMEDG': 'Rotation edge mode',
        'XIBOUND': 'Electron/ion boundary location',
        'XNBOUND': 'Density boundary location',
        'XPHIBOUND': 'Rotation boundary location',
        'TIEDGE': 'Edge ion temperature [eV]',
        'TEEDGE': 'Edge electron temperature [eV]',
        'EDGENE': 'Edge electron density [cm^-3]',
        'OMEEDGE': 'Edge rotation [rad/s]',
        'TI0FRC': 'Central ion temperature factor',
        'MOD0ED': 'Recycling neutral edge mode',
        'NMODEL_PED_WIDTH': 'Pedestal-width model selector',
        'NMODEL_PED_HEIGHT': 'Pedestal-height model selector',
        'TEPEDW': 'Electron pedestal width',
        'TIPEDW': 'Ion pedestal width',
        'XNEPEDW': 'Density pedestal width',
        'TEPED': 'Electron pedestal top [eV]',
        'TIPED': 'Ion pedestal top [eV]',
        'XNEPED': 'Density pedestal top [cm^-3]',
        'SCALE_TEPED': 'Electron pedestal height scale factor',
        'SCALE_TIPED': 'Ion pedestal height scale factor',
        'SCALE_NEPED': 'Density pedestal height scale factor',
        'NMODEL_L2H_TRANS': 'L-H transition model selector',
        'TIME_L2H': 'L->H transition time [s]',
        'TIME_H2L': 'H->L transition time [s]',
        'NLHMODE': 'H-mode indicator flag',
        'TAU_LH_TRANS_X2': 'Minimum factor-of-2 change time [s]',
        'NLRMPFLATTENING': 'RMP density flattening flag',
        'RMPPTOP': 'RMP rational surface location',
        'RMPICOILRATIO': 'RMP lower/upper coil current ratio',
        'RMPFACTOR': 'RMP response scale factor',
        'RMPMODE': 'RMP toroidal mode number',
        'LPED(6)': 'Pedestal model control 6',
        'LPED(8)': 'Pedestal model control 8',
        'CPED(6)': 'Pedestal control 6',
        'CPED(8)': 'Pedestal control 8',
    }

    # Profile inputs that are typically prescribed at the start of a run.
    PRE_SHOT_PROFILE_INPUTS = {
        'NER_IN': 'Electron density input profile',
        'NIM_IN': 'Impurity density input profile',
        'QPR_IN': 'Safety factor input profile',
        'TER_IN': 'Electron temperature input profile',
        'TI2_IN': 'Ion temperature input profile',
        'OMG_IN': 'Toroidal angular velocity input profile',
        'PRS_IN': 'Pressure source / profile input',
        'BOL_IN': 'Bolometer input profile',
        'ZF2_IN': 'Input Zeff profile',
    }

    PRE_SHOT_CONTEXT_DIM = 256

    SAFE_PRE_SHOT_SCALAR_KEYS = [
        # Time / resolution control
        'TGRID1',
        'TGRID2',
        'TINIT',
        'FTIME',
        'DTMINT',
        'DTMAXT',
        'DTINIT',
        'DTMINB',
        'DTMAXB',
        'DTINIB',
        'DTMAXG',
        'DTTOR',
        'DTICRF',
        'DTLH',
        'DTBEAM',
        'NZONES',
        'NZONES_NB',
        'NZONES_FP',
        'NZONES_FB',
        'SEDIT',
        'STEDIT',
        'SELOUT',
        'SELAVG',
        'OUTTIM',
        'AVGTIM',
        'MTHDAVG',
        'AVGSAMP',
        # Field orientation and parallel execution flags
        'NLBCCW',
        'NLJCCW',
        'NBI_PSERVE',
        'NTORIC_PSERVE',
        'NGENRAY_PSERVE',
        'NCQL_PSERVE',
        'NPTCL_PSERVE',
        # Predictive boundary conditions and edge settings
        'MODIEDG',
        'MOD0ED',
        'MODEEDG',
        'MODNEDG',
        'MODOMEDG',
        'XIBOUND',
        'XNBOUND',
        'XPHIBOUND',
        'TIEDGE',
        'TEEDGE',
        'EDGENE',
        'OMEEDGE',
        'TI0FRC',
        # L-H transition and pedestal controls
        'NMODEL_L2H_TRANS',
        'TIME_L2H',
        'TIME_H2L',
        'NLHMODE',
        'TAU_LH_TRANS_X2',
        'NMODEL_PED_WIDTH',
        'NMODEL_PED_HEIGHT',
        'TEPEDW',
        'TIPEDW',
        'XNEPEDW',
        'TEPED',
        'TIPED',
        'XNEPED',
        'SCALE_TEPED',
        'SCALE_TIPED',
        'SCALE_NEPED',
        # Radiation controls
        'NPRAD',
        'NLRAD_BR',
        'NLRAD_LI',
        'NLRAD_CY',
        'VREF_CY',
        # Prescribed source / machine settings
        'PHEAT_IN',
        'PFI0',
        'PSI0',
        'PINJ',
        'BTDIA',
        'RAXIS',
        'YAXIS',
        'RMAJB',
        'LPED(6)',
        'LPED(8)',
        'CPED(6)',
        'CPED(8)',
    ]
    
    # Shape parameters (derived from geometry, important for turbulence)
    SHAPE_PARAMS = ['ELONG', 'TRIANG', 'TRIANGL', 'TRIANGU', 'SQUARE_LO', 'SQUARE_UO']
    
    # GLOBAL SCALARS at t=0 (critical for initial condition specification)
    GLOBAL_SCALARS = {
        'PCUR':   'Plasma current [A]',
        'BTDIA':  'Toroidal field [T]',
        'Q0':     'q at magnetic axis',
        'Q95':    'q at 95% flux surface',
        'BETAT':  'Total beta',
        'BETAI':  'Ion beta',
        'BETAE':  'Electron beta',
        'RAXIS':  'Magnetic axis R [m]',
        'YAXIS':  'Magnetic axis Z [m]',
        'H98Y2':  'Confinement quality factor',
        'PVOL':   'Plasma volume [m^3]',
        'PINJ':   'NBI injected power [W]',
        'RMAJB':  'Minor radius [m]',
        'ZEFF0':  'Core Zeff',
    }
    
    # TRANSPORT COEFFICIENTS at t=0
    TRANSPORT = {
        'CONDE':   'Electron thermal diffusivity [m^2/s]',
        'CONDI':   'Ion thermal diffusivity [m^2/s]',
        'DIFFD':   'Electron particle diffusion [m^2/s]',
        'VELH':    'Electron velocity [m/s]',
    }
    
    # NBI SOURCE PROFILES at t=0 (Critical for ion density evolution)
    NBI_SOURCES = {
        'SBTH':     'Thermal source',
        'SBCX0_D':  'Charge exchange source',
        'SBAL_ION': 'Ion particle source',
    }
    
    # Additional important profiles for physics constraints
    OTHER_PROFILES = {
        'CUR':     'Parallel current density [A/m^2]',
        'CURBS':   'Bootstrap current density [A/m^2]',
        'Q':       'Safety factor q(ρ) on XB grid',
        'SHAT':    'Magnetic shear s(ρ)',
        # --- EXOGENOUS HEATING + FUELLING drivers (added for the time-resolved driver set) ---
        # Volume-reduced downstream into model driver channels. Graceful: only extracted where the
        # CDF has them (machines/shots without a given system simply skip it). NOT leakage: each is
        # an exogenous actuator's coupled power/rate (only mild deposition/penetration dependence).
        'PECH':    'ECH power density [W/cm^3]',          # electron-cyclotron heating
        'PRFE':    'ICRF power to electrons [W/cm^3]',    # ion-cyclotron (summed with PRFI -> ICRF)
        'PRFI':    'ICRF power to ions [W/cm^3]',
        'PLH':     'Lower-hybrid power density [W/cm^3]', # LH heating
        'SESGF':   'Gas-flow electron source [N/cm^3/s]', # gas fuelling (top DENSITY driver)
    }


class CDFVariableExtractor:
    """
    Extracts physics quantities from TRANSP CDF files with full error handling
    and validation. Implements the complete state vector extraction as specified
    in Section 2.2 of the experimental plan.
    
    The extracted state vector has shape:
    - Kinetic profiles:     15 variables × 100 radial points = 1,500 dims
    - Geometry (Fourier):   66 modes × 100 flux surfaces = 6,600 dims
    - Shape parameters:     6 variables × 100 = 600 dims
    - Global scalars:       17 scalars
    - Transport coeff:      7 variables × 100 = 700 dims
    - NBI sources:          5 variables × 100 = 500 dims
    Total raw dimension: ~9,917 (will be compressed via encoders)
    """
    
    def __init__(self, cdf_path: str, verbose: bool = True, limiter_reference_path: Optional[str] = None):
        """
        Initialize the extractor with a CDF file path.
        
        Args:
            cdf_path: Path to TRANSP CDF file
            verbose: Whether to print extraction progress
        """
        self.cdf_path = Path(cdf_path)
        self.verbose = verbose
        self.limiter_reference_path = Path(limiter_reference_path) if limiter_reference_path else None
        if self.limiter_reference_path is None:
            default_ref = Path('processed_data/limiter_reference.npz')
            if default_ref.exists():
                self.limiter_reference_path = default_ref
                self._log(f"Using stored limiter reference NPZ: {default_ref}")

        self._reference_limiter_geometry = None
        if self.limiter_reference_path is not None and self.limiter_reference_path.suffix.lower() == '.npz':
            self._log(f"Using limiter reference NPZ: {self.limiter_reference_path}")

        if not self.cdf_path.exists():
            raise FileNotFoundError(f"CDF file not found: {cdf_path}")
        
        self.cdf = nc.Dataset(str(self.cdf_path))
        self._log(f"Loaded CDF: {self.cdf_path.name}")
        self._validate_cdf()
    
    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            logger.info(message)
    
    def _validate_cdf(self):
        """Validate that CDF contains all required variables."""
        required_vars = []
        required_vars.extend(PlasmaVariables.KINETIC_PROFILES.keys())
        required_vars.extend(PlasmaVariables.LIMITER_GEOMETRY)
        
        missing = [v for v in required_vars if v not in self.cdf.variables]
        if missing:
            self._log(f"WARNING: Missing variables: {missing[:5]}...")  # Show first 5
    
    def extract_kinetic_profiles_t0(self, radial_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract kinetic profiles at t=0.
        
        Args:
            radial_index: Which radial grid to use (0 for X, 1 for XB)
        
        Returns:
            Dictionary mapping variable names to profile arrays of shape (n_radial,)
        """
        profiles = {}
        
        for var_name in PlasmaVariables.KINETIC_PROFILES.keys():
            try:
                if var_name in self.cdf.variables:
                    data = self.cdf.variables[var_name][0, :]  # First time, all radial points
                    profiles[var_name] = np.array(data)
                    self._log(f"✓ Extracted {var_name}: shape {data.shape}")
            except Exception as e:
                self._log(f"WARNING: Failed to extract {var_name}: {e}")
        
        return profiles
    
    def _build_plasma_geometry_tensor(self, time_index: int = 0, target_n_psi: int = 40) -> np.ndarray:
        """Build the plasma boundary Fourier geometry tensor from RMC/RMS/YMC/YMS."""
        coeff_names = [
            *PlasmaVariables.PLASMA_GEOMETRY_COSINE,
            *PlasmaVariables.PLASMA_GEOMETRY_Z_COSINE,
        ]

        coeff_series = []
        for var_name in coeff_names:
            if var_name not in self.cdf.variables:
                coeff_series.append(None)
                continue
            coeff_series.append(np.asarray(self.cdf.variables[var_name][time_index, :], dtype=np.float32).reshape(-1))

        raw_lengths = [series.shape[0] for series in coeff_series if series is not None]
        if not raw_lengths:
            raise ValueError("No plasma geometry Fourier coefficients found in CDF")

        raw_n_psi = raw_lengths[0]
        for series in coeff_series:
            if series is not None and series.shape[0] != raw_n_psi:
                raise ValueError(
                    f"Plasma geometry coefficient length mismatch: expected {raw_n_psi}, got {series.shape[0]}"
                )

        if target_n_psi and target_n_psi > 0 and raw_n_psi != target_n_psi:
            s_old = np.linspace(0.0, 1.0, raw_n_psi)
            s_new = np.linspace(0.0, 1.0, target_n_psi)
        else:
            s_old = None
            s_new = None

        geometry_tensor = np.zeros((target_n_psi if target_n_psi and target_n_psi > 0 else raw_n_psi, len(coeff_names)), dtype=np.float32)
        for col_idx, series in enumerate(coeff_series):
            if series is None:
                continue
            if s_old is None or s_new is None:
                geometry_tensor[:, col_idx] = series.astype(np.float32)
            else:
                geometry_tensor[:, col_idx] = np.interp(s_new, s_old, series).astype(np.float32)
        return geometry_tensor

    @staticmethod
    def _extract_limiter_contour(dataset, time_index: int = 0):
        if 'RLIM' not in dataset.variables or 'YLIM' not in dataset.variables:
            return None

        rlim_raw = np.asarray(dataset.variables['RLIM'][time_index], dtype=np.float32).reshape(-1)
        ylim_raw = np.asarray(dataset.variables['YLIM'][time_index], dtype=np.float32).reshape(-1)
        return rlim_raw, ylim_raw

    def _load_reference_limiter_contour(self, time_index: int = 0):
        if self._reference_limiter_geometry is not None:
            return self._reference_limiter_geometry

        if self.limiter_reference_path is None:
            return None
        if not self.limiter_reference_path.exists():
            self._log(f"WARNING: Limiter reference not found: {self.limiter_reference_path}")
            return None

        if self.limiter_reference_path.suffix.lower() == '.npz':
            try:
                with np.load(self.limiter_reference_path) as data:
                    rlim_raw = np.asarray(data['rlim'], dtype=np.float32)
                    ylim_raw = np.asarray(data['ylim'], dtype=np.float32)
                    contour = (rlim_raw, ylim_raw)
                    self._reference_limiter_geometry = contour
                    self._log(f"✓ Loaded limiter reference contour from {self.limiter_reference_path}")
                    return contour
            except Exception as e:
                self._log(f"WARNING: Failed to load limiter reference NPZ: {e}")
                return None

        try:
            with nc.Dataset(str(self.limiter_reference_path)) as reference_cdf:
                contour = self._extract_limiter_contour(reference_cdf, time_index=time_index)
        except Exception as e:
            self._log(f"WARNING: Failed to read limiter reference CDF: {e}")
            return None

        if contour is not None:
            self._reference_limiter_geometry = contour
        return contour

    def _build_limiter_geometry_tensor(self, time_index: int = 0, target_n_psi: int = 40) -> np.ndarray:
        """Build a static limiter geometry tensor from RLIM/YLIM or a shared reference CDF."""
        contour = self._extract_limiter_contour(self.cdf, time_index=time_index)
        if contour is None:
            contour = self._load_reference_limiter_contour(time_index=time_index)
        if contour is None:
            # FALLBACK: Create synthetic limiter geometry for NSTX tokamak
            # NSTX has major radius ~0.85m, minor radius ~0.65m
            self._log("WARNING: Using synthetic limiter geometry (NSTX approximate)")
            theta = np.linspace(0, 2*np.pi, target_n_psi, endpoint=False)
            # Approximate D-shaped limiter contour
            R0, a = 0.85, 0.65  # meters
            elongation = 1.8  # typical NSTX elongation
            triangularity = 0.3  # typical NSTX triangularity
            
            # Parametric equations for D-shape
            rlim_raw = R0 + a * np.cos(theta + triangularity * np.sin(theta))
            ylim_raw = elongation * a * np.sin(theta)
            contour = (rlim_raw.astype(np.float32), ylim_raw.astype(np.float32))

        rlim_raw, ylim_raw = contour

        if rlim_raw.shape[0] != ylim_raw.shape[0]:
            raise ValueError(
                f"Limiter contour length mismatch: RLIM={rlim_raw.shape[0]} vs YLIM={ylim_raw.shape[0]}"
            )

        if target_n_psi and target_n_psi > 0 and rlim_raw.shape[0] != target_n_psi:
            s_old = np.linspace(0.0, 1.0, rlim_raw.shape[0])
            s_new = np.linspace(0.0, 1.0, target_n_psi)
            rlim = np.interp(s_new, s_old, rlim_raw).astype(np.float32)
            ylim = np.interp(s_new, s_old, ylim_raw).astype(np.float32)
        else:
            rlim = rlim_raw.astype(np.float32)
            ylim = ylim_raw.astype(np.float32)

        geometry_tensor = np.zeros((rlim.shape[0], 66), dtype=np.float32)
        geometry_tensor[:, 0] = rlim
        geometry_tensor[:, 1] = ylim
        return geometry_tensor

    def extract_geometry_tensor_t0(self) -> np.ndarray:
        """
        Extract the plasma boundary Fourier geometry tensor at t=0.

        Returns:
            Geometry tensor of shape (n_psi, 66)
        """
        try:
            geometry_tensor = self._build_plasma_geometry_tensor(time_index=0, target_n_psi=40)
        except Exception as exc:
            # Many machine CDFs do not expose Fourier plasma coefficients consistently.
            # Fall back to the static limiter geometry so data loading remains usable.
            self._log(f"WARNING: Falling back to limiter geometry for plasma tensor: {exc}")
            geometry_tensor = self._build_limiter_geometry_tensor(time_index=0, target_n_psi=40)
        self._log(f"✓ Constructed plasma geometry tensor: shape {geometry_tensor.shape}")
        return geometry_tensor

    def extract_limiter_geometry_tensor_t0(self) -> np.ndarray:
        """
        Extract the static limiter geometry tensor at t=0.
        
        Constructs tensor of shape (n_psi, 66) where the first two columns are
        the interpolated limiter contour (RLIM, YLIM) and the remaining columns
        are zero-padded to preserve the existing model interface.
        
        Returns:
            Geometry tensor of shape (n_psi, 66)
        """
        geometry_tensor = self._build_limiter_geometry_tensor(time_index=0, target_n_psi=40)
        self._log(f"✓ Constructed static limiter geometry tensor: shape {geometry_tensor.shape}")
        return geometry_tensor
    
    def extract_global_scalars_t0(self) -> Dict[str, float]:
        """
        Extract global scalar quantities at t=0.
        
        Returns:
            Dictionary of scalar values
        """
        scalars = {}
        
        for var_name in PlasmaVariables.GLOBAL_SCALARS.keys():
            try:
                if var_name in self.cdf.variables:
                    # For time-dependent scalars
                    if 'TIME' in self.cdf.variables[var_name].dimensions:
                        value = float(self.cdf.variables[var_name][0])
                    else:
                        # Scalar variable (no time dimension)
                        value = float(self.cdf.variables[var_name][:])
                    scalars[var_name] = value
            except Exception as e:
                self._log(f"WARNING: Failed to extract global scalar {var_name}: {e}")
        
        self._log(f"✓ Extracted {len(scalars)} global scalars")
        return scalars
    
    def extract_transport_coefficients_t0(self) -> Dict[str, np.ndarray]:
        """Extract transport coefficients at t=0."""
        transport = {}
        
        for var_name in PlasmaVariables.TRANSPORT.keys():
            try:
                if var_name in self.cdf.variables:
                    data = self.cdf.variables[var_name][0, :]
                    transport[var_name] = np.array(data)
            except Exception as e:
                self._log(f"WARNING: Failed to extract transport {var_name}: {e}")
        
        return transport
    
    def extract_nbi_sources_t0(self) -> Dict[str, np.ndarray]:
        """Extract NBI source profiles at t=0."""
        sources = {}
        
        for var_name in PlasmaVariables.NBI_SOURCES.keys():
            try:
                if var_name in self.cdf.variables:
                    data = self.cdf.variables[var_name][0, :]
                    sources[var_name] = np.array(data)
            except Exception as e:
                self._log(f"WARNING: Failed to extract NBI source {var_name}: {e}")
        
        return sources
    
    def extract_other_profiles_t0(self) -> Dict[str, np.ndarray]:
        """Extract other important profiles (q, shear, current, etc.)."""
        profiles = {}
        
        for var_name in PlasmaVariables.OTHER_PROFILES.keys():
            try:
                if var_name in self.cdf.variables:
                    data = self.cdf.variables[var_name][0, :]
                    profiles[var_name] = np.array(data)
            except Exception as e:
                self._log(f"WARNING: Failed to extract {var_name}: {e}")
        
        return profiles

    def extract_pre_shot_inputs_t0(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract additional pre-shot-only inputs and their raw radial summaries."""
        inputs = {
            'scalars': {},
            'profiles': {},
            'shape_params': {},
            'control_arrays': {},
        }

        for var_name in PlasmaVariables.PRE_SHOT_SCALARS.keys():
            try:
                if var_name in self.cdf.variables:
                    var = self.cdf.variables[var_name]
                    if 'TIME' in var.dimensions:
                        inputs['scalars'][var_name] = float(var[0])
                    else:
                        inputs['scalars'][var_name] = float(var[:])
            except Exception as e:
                self._log(f"WARNING: Failed to extract pre-shot scalar {var_name}: {e}")

        for var_name in PlasmaVariables.PRE_SHOT_PROFILE_INPUTS.keys():
            try:
                if var_name in self.cdf.variables:
                    data = np.asarray(self.cdf.variables[var_name][0, :], dtype=np.float32)
                    inputs['profiles'][var_name] = data
            except Exception as e:
                self._log(f"WARNING: Failed to extract pre-shot profile {var_name}: {e}")

        for var_name in PlasmaVariables.SHAPE_PARAMS:
            try:
                if var_name in self.cdf.variables:
                    data = np.asarray(self.cdf.variables[var_name][0, :], dtype=np.float32)
                    inputs['shape_params'][var_name] = data
            except Exception as e:
                self._log(f"WARNING: Failed to extract shape parameter {var_name}: {e}")

        for var_name in ['LPED', 'CPED']:
            try:
                if var_name in self.cdf.variables:
                    data = np.asarray(self.cdf.variables[var_name][:], dtype=np.float32).reshape(-1)
                    inputs['control_arrays'][var_name] = data
            except Exception as e:
                self._log(f"WARNING: Failed to extract control array {var_name}: {e}")

        return inputs

    def extract_pre_shot_context_t0(self) -> np.ndarray:
        """Build a fixed-length pre-shot context vector from only safe pre-shot inputs.
        
        CRITICAL: Only include quantities that are known/prescribed BEFORE the shot.
        Exclude measured plasma-state proxies and any quantities that TRANSP
        infers/calculates from the full simulation.
        
        Safe pre-shot quantities:
        - timing and solver settings
        - prescribed machine controls and source settings
        - input profiles and shape summaries
        - limiter/actuator-style control arrays
        
        EXCLUDE inferred quantities:
        - NI, NE, TE, TI (transport results)
        - PCUR (measured plasma current)
        - other measured plasma-state proxies used as labels elsewhere
        - Geometry tensors (equilibrium results)
        - Any fitted/predicted quantities
        """
        pre_shot = self.extract_pre_shot_inputs_t0()

        ordered_scalars = list(PlasmaVariables.SAFE_PRE_SHOT_SCALAR_KEYS)

        ordered_profiles = list(PlasmaVariables.PRE_SHOT_PROFILE_INPUTS.keys())

        ordered_shape_params = list(PlasmaVariables.SHAPE_PARAMS)

        context_values: List[float] = []

        # Scalars
        for key in ordered_scalars:
            value = pre_shot['scalars'].get(key)
            context_values.append(float(value) if value is not None else 0.0)

        # Profile summaries (mean, std, center, edge)
        for key in ordered_profiles:
            profile = pre_shot['profiles'].get(key)
            if profile is None or profile.size == 0:
                context_values.extend([0.0, 0.0, 0.0, 0.0])
                continue

            profile = np.asarray(profile, dtype=np.float32).reshape(-1)
            mean_val = float(np.mean(profile))
            std_val = float(np.std(profile))
            center_val = float(profile[0]) if len(profile) > 0 else 0.0
            edge_val = float(profile[-1]) if len(profile) > 0 else 0.0
            context_values.extend([mean_val, std_val, center_val, edge_val])

        # Shape parameters
        for key in ordered_shape_params:
            profile = pre_shot['shape_params'].get(key)
            if profile is None or profile.size == 0:
                context_values.extend([0.0, 0.0, 0.0, 0.0])
                continue

            profile = np.asarray(profile, dtype=np.float32).reshape(-1)
            mean_val = float(np.mean(profile))
            std_val = float(np.std(profile))
            center_val = float(profile[0]) if len(profile) > 0 else 0.0
            edge_val = float(profile[-1]) if len(profile) > 0 else 0.0
            context_values.extend([mean_val, std_val, center_val, edge_val])

        for key in ['LPED', 'CPED']:
            control = pre_shot.get('control_arrays', {}).get(key)
            if control is None or control.size == 0:
                context_values.extend([0.0] * 16)
                continue
            control = np.asarray(control, dtype=np.float32).reshape(-1)
            control = control[:16]
            if control.size < 16:
                control = np.pad(control, (0, 16 - control.size), mode='constant')
            context_values.extend(control.tolist())

        # Pad to required dimension
        if len(context_values) < PlasmaVariables.PRE_SHOT_CONTEXT_DIM:
            context_values.extend([0.0] * (PlasmaVariables.PRE_SHOT_CONTEXT_DIM - len(context_values)))

        return np.asarray(context_values[:PlasmaVariables.PRE_SHOT_CONTEXT_DIM], dtype=np.float32)
    
    def extract_full_state_vector_t0(self) -> Dict[str, np.ndarray]:
        """
        Extract complete t=0 state vector as specified in Section 2.2 of plan.
        
        Returns:
            Dictionary containing all components of the plasma state
        """
        state = {}
        
        # Extract all components
        state['kinetic_profiles'] = self.extract_kinetic_profiles_t0()
        state['geometry_tensor'] = self.extract_geometry_tensor_t0()
        state['limiter_geometry_tensor'] = self.extract_limiter_geometry_tensor_t0()
        state['global_scalars'] = self.extract_global_scalars_t0()
        state['transport_coefficients'] = self.extract_transport_coefficients_t0()
        state['nbi_sources'] = self.extract_nbi_sources_t0()
        state['other_profiles'] = self.extract_other_profiles_t0()
        state['pre_shot_inputs'] = self.extract_pre_shot_inputs_t0()
        state['pre_shot_context'] = self.extract_pre_shot_context_t0()
        
        self._log("✓ Complete state vector extraction finished")
        return state
    
    def extract_ion_density_trajectory(self) -> np.ndarray:
        """
        Extract full time trajectory of ion density (primary target variable).
        
        Returns:
            Array of shape (n_time, n_radial) containing NI evolution
        """
        if 'NI' in self.cdf.variables:
            ni_trajectory = np.array(self.cdf.variables['NI'][:, :])
            self._log(f"✓ Extracted NI trajectory: shape {ni_trajectory.shape}")
            return ni_trajectory
        else:
            raise ValueError("NI variable not found in CDF")
    
    def extract_full_state_trajectory(self, max_timesteps: Optional[int] = None) -> List[Dict]:
        """
        Extract complete state vector at EVERY timestep (not just t=0).
        
        Per plan Section 7, training uses full state rollouts: model predicts
        complete state evolution, not just NI.
        
        Args:
            max_timesteps: Limit to first N timesteps (for testing)
        
        Returns:
            List of dictionaries, one per timestep. Each dict contains state_t
            with same structure as extract_full_state_vector_t0() output.
        """
        # Get total number of timesteps from any available time axis or from NI.
        if 'TIME3' in self.cdf.variables:
            n_time = len(self.cdf.variables['TIME3'][:])
        elif 'TIME' in self.cdf.variables:
            n_time = len(self.cdf.variables['TIME'][:])
        elif 'NI' in self.cdf.variables:
            n_time = int(self.cdf.variables['NI'].shape[0])
        else:
            raise ValueError("No usable time axis found in CDF")
        if max_timesteps:
            n_time = min(n_time, max_timesteps)
        
        trajectory = []
        limiter_geometry = self.extract_limiter_geometry_tensor_t0()
        pre_shot_inputs = self.extract_pre_shot_inputs_t0()
        pre_shot_context = self.extract_pre_shot_context_t0()
        
        for t_idx in range(n_time):
            state_t = {
                'kinetic_profiles': {},
                'geometry_tensor': np.zeros((40, 66), dtype=np.float32),
                'limiter_geometry_tensor': limiter_geometry,
                'global_scalars': {},
                'transport_coefficients': {},
                'nbi_sources': {},
                'other_profiles': {},
                'pre_shot_inputs': pre_shot_inputs,
                'pre_shot_context': pre_shot_context,
            }
            
            # Extract plasma geometry at time t
            try:
                state_t['geometry_tensor'] = self._build_plasma_geometry_tensor(time_index=t_idx, target_n_psi=40)
            except Exception as exc:
                self._log(f"WARNING: Falling back to limiter geometry for trajectory step {t_idx}: {exc}")
                try:
                    state_t['geometry_tensor'] = self._build_limiter_geometry_tensor(time_index=0, target_n_psi=40)
                except Exception:
                    state_t['geometry_tensor'] = np.zeros((40, 66), dtype=np.float32)

            # Preserve limiter geometry as a static machine boundary
            state_t['limiter_geometry_tensor'] = limiter_geometry

            # Extract kinetic profiles at time t
            for var_name in PlasmaVariables.KINETIC_PROFILES.keys():
                try:
                    if var_name in self.cdf.variables:
                        data = self.cdf.variables[var_name][t_idx, :]
                        state_t['kinetic_profiles'][var_name] = np.array(data)
                except Exception as e:
                    pass  # Skip if unavailable
            
            # Extract global scalars at time t
            for var_name in PlasmaVariables.GLOBAL_SCALARS.keys():
                try:
                    if var_name in self.cdf.variables:
                        if 'TIME' in self.cdf.variables[var_name].dimensions:
                            value = float(self.cdf.variables[var_name][t_idx])
                        else:
                            value = float(self.cdf.variables[var_name][:])
                        state_t['global_scalars'][var_name] = value
                except Exception as e:
                    pass

            # Extract transport coefficients at time t
            for var_name in PlasmaVariables.TRANSPORT.keys():
                try:
                    if var_name in self.cdf.variables:
                        data = self.cdf.variables[var_name][t_idx, :]
                        state_t['transport_coefficients'][var_name] = np.array(data)
                except Exception:
                    pass

            # Extract NBI sources at time t
            for var_name in PlasmaVariables.NBI_SOURCES.keys():
                try:
                    if var_name in self.cdf.variables:
                        data = self.cdf.variables[var_name][t_idx, :]
                        state_t['nbi_sources'][var_name] = np.array(data)
                except Exception:
                    pass

            # Extract other profiles at time t
            for var_name in PlasmaVariables.OTHER_PROFILES.keys():
                try:
                    if var_name in self.cdf.variables:
                        data = self.cdf.variables[var_name][t_idx, :]
                        state_t['other_profiles'][var_name] = np.array(data)
                except Exception:
                    pass
            
            trajectory.append(state_t)
        
        self._log(f"✓ Extracted full state trajectory: {len(trajectory)} timesteps")
        return trajectory
    
    def close(self):
        """Close the CDF file."""
        self.cdf.close()
        self._log("CDF file closed")


# ==============================================================================
# SECTION 2: Derived Physics Quantity Computation
# ==============================================================================

class PhysicsQuantityComputer:
    """
    Computes derived physics quantities from profiles.
    
    Examples:
    - Normalized pressure gradients (η_i, η_e) — drive turbulent transport
    - Magnetic shear s(ρ)
    - Effective charge Zeff
    - Trapped particle fraction
    - Bootstrap current consistency measures
    
    Theory: These dimensionless parameters determine transport regime and are
    more physically meaningful than raw profiles.
    """
    
    @staticmethod
    def compute_density_gradient_drive(ne: np.ndarray, ni: np.ndarray, 
                                       te: np.ndarray, ti: np.ndarray,
                                       radial_grid: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute normalized temperature/density gradient drives (η_i, η_e).
        
        These are key ITG (Ion Temperature Gradient) and ETG (Electron Temperature
        Gradient) instability drive parameters that control turbulent transport.
        
        η_i = (T_i / n_i) * (d n_i / d r) / (d T_i / d r)
        η_e = (T_e / n_e) * (d n_e / d r) / (d T_e / d r)
        
        Returns:
            Dictionary with 'eta_i' and 'eta_e' profiles
        """
        # Compute gradients using centered differences
        dne_dr = np.gradient(ne, radial_grid)
        dni_dr = np.gradient(ni, radial_grid)
        dte_dr = np.gradient(te, radial_grid)
        dti_dr = np.gradient(ti, radial_grid)
        
        # Avoid division by zero
        eps = 1e-10
        
        eta_e = (te / (ne + eps)) * dne_dr / (dte_dr + eps)
        eta_i = (ti / (ni + eps)) * dni_dr / (dti_dr + eps)
        
        return {'eta_e': eta_e, 'eta_i': eta_i}
    
    @staticmethod
    def compute_effective_charge(ne: np.ndarray, ni: np.ndarray, 
                                nimp_profiles: List[np.ndarray],
                                impurity_charges: List[int]) -> np.ndarray:
        """
        Compute effective charge Z_eff.
        
        Zeff = (n_e) / (n_i + Σ_j Z_j^2 n_j) × (n_i + Σ_j Z_j n_j)
        
        Approximation (if no impurity data): Zeff ≈ 1 (fully ionized hydrogen/deuterium)
        """
        if not nimp_profiles:
            return np.ones_like(ne)  # Default to hydrogen
        
        numerator = ne
        denominator = ni
        for nj, zj in zip(nimp_profiles, impurity_charges):
            denominator += zj * zj * nj
        
        zeff = numerator / (denominator + 1e-10)
        return np.clip(zeff, 1.0, 10.0)  # Physical bounds


# ==============================================================================
# SECTION 3: PyTorch Dataset and DataLoader
# ==============================================================================

class PPPLPlasmaDataset(Dataset):
    """
    PyTorch Dataset for PPPL tokamak CDF data.
    
    Loads multiple CDF files, extracts t=0 state vectors and target trajectories,
    and provides a unified interface for training models.
    
    Features:
    - Multi-shot support with automatic stratification
    - Data normalization (per-variable mean/std across training set)
    - Time window extraction for data augmentation
    - Physics quantity computation on-the-fly
    - GPU-ready tensor outputs
    
    Data flow:
    CDF file → Full state vector extraction → Normalization → PyTorch tensor
    """
    
    def __init__(self, 
                 cdf_paths: List[str],
                 max_timesteps: Optional[int] = None,
                 normalize: bool = True,
                 device: str = 'cpu',
                 limiter_reference_path: Optional[str] = None):
        """
        Initialize dataset from CDF files.
        
        Args:
            cdf_paths: List of paths to CDF files
            max_timesteps: Maximum number of timesteps per shot (None = all)
            normalize: Whether to normalize data to unit variance
            device: 'cpu' or 'cuda'
        """
        self.cdf_paths = cdf_paths
        self.max_timesteps = max_timesteps
        self.normalize = normalize
        self.device = device
        # Optional path to a limiter reference NPZ to pass into each extractor.
        self.limiter_reference_path = limiter_reference_path
        
        self.data = []
        self.metadata = []
        
        logger.info(f"Loading {len(cdf_paths)} CDF files...")
        self._load_all_cdf_files()
        
        logger.info(f"Dataset initialized with {len(self.data)} shot-trajectories")
    
    def _load_all_cdf_files(self):
        # IMPORTANT: Filter to well-formed CDFs only        valid_cdf_patterns = ['134020D81.CDF']
        
        filtered_cdf_paths = []
        for cdf_path in self.cdf_paths:
            # Check if CDF filename matches any valid pattern
            cdf_name = Path(cdf_path).name
            # if any(pattern in cdf_name for pattern in valid_cdf_patterns):
            filtered_cdf_paths.append(cdf_path)
            # else:
            #     logger.warning(f"⊘ Skipping non-standard CDF: {cdf_name}")
        
        if not filtered_cdf_paths:
            logger.warning("No valid CDFs found after filtering. Attempting to use all CDFs anyway...")
            filtered_cdf_paths = self.cdf_paths
        
        for i, cdf_path in enumerate(filtered_cdf_paths):
            try:
                logger.info(f"Processing shot {i+1}/{len(filtered_cdf_paths)}: {Path(cdf_path).name}")
                extractor = CDFVariableExtractor(cdf_path, verbose=False,
                                                 limiter_reference_path=self.limiter_reference_path)
                
                # Extract t=0 state vector
                state_t0 = extractor.extract_full_state_vector_t0()
                
                # Extract FULL state trajectory (all timesteps) per plan Section 7
                state_trajectory = extractor.extract_full_state_trajectory(max_timesteps=self.max_timesteps)
                
                # Extract ion density trajectory (primary target)
                ni_trajectory = extractor.extract_ion_density_trajectory()
                
                # Limit timesteps if requested
                if self.max_timesteps:
                    ni_trajectory = ni_trajectory[:self.max_timesteps, :]
                    state_trajectory = state_trajectory[:self.max_timesteps]
                
                machine = _infer_machine_from_path(cdf_path)

                self.data.append({
                    'state_t0': state_t0,
                    'state_trajectory': state_trajectory,  # NEW: full state at each timestep
                    'ni_trajectory': ni_trajectory,
                    'cdf_path': cdf_path,
                    'machine': machine,
                })

                self.metadata.append({
                    'shot': i,
                    'machine': machine,
                    'n_timesteps': ni_trajectory.shape[0],
                    'n_radial': ni_trajectory.shape[1],
                    'cdf_path': cdf_path
                })
                
                extractor.close()
                
            except Exception as e:
                logger.warning(f"Failed to process {cdf_path}: {e}")
        
        # Apply normalization if requested (per plan Section 2.4)
        if self.normalize and len(self.data) > 0:
            self._apply_normalization()
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
            - 'state_t0': Complete t=0 state vector (flattened to 1D)
            - 'ni_trajectory': Ion density trajectory (2D: time × radius)
            - 'state_trajectory': Full state trajectory (2D: time × flattened_state_dim)
            - 'shot_id': Shot identifier
        
        Per plan Section 7, training is on full state rollouts, not just NI.
        """
        sample_data = self.data[idx]
        
        # Flatten state vector components
        state_t0_flat = self._flatten_state(sample_data['state_t0'])
        
        # Build full state trajectory: apply same flattening to each timestep
        state_trajectory_flat = None
        if 'state_trajectory' in sample_data:
            state_traj_list = []
            for state_t in sample_data['state_trajectory']:
                state_traj_list.append(self._flatten_state(state_t))
            state_trajectory_flat = np.stack(state_traj_list, axis=0)  # (time, state_dim)
        
        # Convert to tensors
        state_tensor = torch.from_numpy(state_t0_flat).float().to(self.device)
        trajectory_tensor = torch.from_numpy(sample_data['ni_trajectory']).float().to(self.device)
        
        result = {
            'state_t0': state_tensor,
            'ni_trajectory': trajectory_tensor,
            'shot_id': idx
        }
        
        if state_trajectory_flat is not None:
            result['state_trajectory'] = torch.from_numpy(state_trajectory_flat).float().to(self.device)
        
        return result
    
    def _flatten_state(self, state: Dict) -> np.ndarray:
        """Flatten hierarchical state dictionary into 1D array."""
        components = []
        
        # Add kinetic profiles
        for key in sorted(state['kinetic_profiles'].keys()):
            components.append(state['kinetic_profiles'][key])
        
        # Add geometry tensor (reshape from (n_psi, 66) to 1D)
        if state.get('geometry_tensor') is not None:
            components.append(state['geometry_tensor'].flatten())
        
        # Add transport coefficients
        for key in sorted(state.get('transport_coefficients', {}).keys()):
            components.append(state['transport_coefficients'][key])
        
        # Add NBI sources
        for key in sorted(state.get('nbi_sources', {}).keys()):
            components.append(state['nbi_sources'][key])

        # Add other profiles (q, shear, current, bootstrap current)
        for key in sorted(state.get('other_profiles', {}).keys()):
            components.append(state['other_profiles'][key])
        
        # Add global scalars
        global_values = np.array(list(state.get('global_scalars', {}).values()), dtype=np.float32)
        if global_values.size > 0:
            components.append(global_values)
        
        return np.concatenate(components)
    
    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for a sample."""
        return self.metadata[idx]
    
    def _apply_normalization(self) -> None:
        """
        Normalize all data per plan Section 2.4.
        
        Strategy: Compute per-variable mean/std across all shots and all timesteps,
        then normalize each variable independently.
        
        This ensures model training is stable (no scale mismatch between tiny 
        fractions like ρ and huge densities like n_e).
        """
        logger.info("Applying normalization to dataset...")
        
        def collect(values_by_key: Dict[str, List[np.ndarray]], key: str, value):
            if value is None:
                return
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    collect(values_by_key, f"{key}.{sub_key}", sub_value)
            else:
                arr = np.asarray(value)
                values_by_key.setdefault(key, []).append(arr)

        all_values_by_key: Dict[str, List[np.ndarray]] = {}
        for sample in self.data:
            collect(all_values_by_key, 'state_t0.kinetic_profiles', sample['state_t0'].get('kinetic_profiles', {}))
            collect(all_values_by_key, 'state_t0.transport_coefficients', sample['state_t0'].get('transport_coefficients', {}))
            collect(all_values_by_key, 'state_t0.nbi_sources', sample['state_t0'].get('nbi_sources', {}))
            collect(all_values_by_key, 'state_t0.other_profiles', sample['state_t0'].get('other_profiles', {}))
            collect(all_values_by_key, 'state_t0.geometry_tensor', sample['state_t0'].get('geometry_tensor'))
            collect(all_values_by_key, 'state_t0.limiter_geometry_tensor', sample['state_t0'].get('limiter_geometry_tensor'))
            collect(all_values_by_key, 'state_t0.global_scalars', np.asarray(list(sample['state_t0'].get('global_scalars', {}).values()), dtype=np.float32))
            collect(all_values_by_key, 'state_t0.pre_shot_context', sample['state_t0'].get('pre_shot_context'))

            for state_t in sample['state_trajectory']:
                collect(all_values_by_key, 'state_trajectory.kinetic_profiles', state_t.get('kinetic_profiles', {}))
                collect(all_values_by_key, 'state_trajectory.transport_coefficients', state_t.get('transport_coefficients', {}))
                collect(all_values_by_key, 'state_trajectory.nbi_sources', state_t.get('nbi_sources', {}))
                collect(all_values_by_key, 'state_trajectory.other_profiles', state_t.get('other_profiles', {}))
                collect(all_values_by_key, 'state_trajectory.geometry_tensor', state_t.get('geometry_tensor'))
                collect(all_values_by_key, 'state_trajectory.limiter_geometry_tensor', state_t.get('limiter_geometry_tensor'))
                collect(all_values_by_key, 'state_trajectory.global_scalars', np.asarray(list(state_t.get('global_scalars', {}).values()), dtype=np.float32))
                collect(all_values_by_key, 'state_trajectory.pre_shot_context', state_t.get('pre_shot_context'))

        normalization_stats = {}
        for key, values_list in all_values_by_key.items():
            if not values_list:
                continue
            all_values = np.concatenate([np.asarray(v).reshape(-1) for v in values_list])
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            if std_val < 1e-10:
                std_val = 1.0
            normalization_stats[key] = {'mean': mean_val, 'std': std_val}
            logger.info(f"  {key}: μ={mean_val:.4e}, σ={std_val:.4e}")

        def normalize(value, key):
            if key not in normalization_stats:
                return value
            stats = normalization_stats[key]
            return (value - stats['mean']) / stats['std']

        for sample in self.data:
            for var_name in list(sample['state_t0'].get('kinetic_profiles', {}).keys()):
                sample['state_t0']['kinetic_profiles'][var_name] = normalize(
                    sample['state_t0']['kinetic_profiles'][var_name], f'state_t0.kinetic_profiles.{var_name}'
                )
            for var_name in list(sample['state_t0'].get('transport_coefficients', {}).keys()):
                sample['state_t0']['transport_coefficients'][var_name] = normalize(
                    sample['state_t0']['transport_coefficients'][var_name], f'state_t0.transport_coefficients.{var_name}'
                )
            for var_name in list(sample['state_t0'].get('nbi_sources', {}).keys()):
                sample['state_t0']['nbi_sources'][var_name] = normalize(
                    sample['state_t0']['nbi_sources'][var_name], f'state_t0.nbi_sources.{var_name}'
                )
            for var_name in list(sample['state_t0'].get('other_profiles', {}).keys()):
                sample['state_t0']['other_profiles'][var_name] = normalize(
                    sample['state_t0']['other_profiles'][var_name], f'state_t0.other_profiles.{var_name}'
                )
            if sample['state_t0'].get('geometry_tensor') is not None:
                sample['state_t0']['geometry_tensor'] = normalize(
                    sample['state_t0']['geometry_tensor'], 'state_t0.geometry_tensor'
                )
            if sample['state_t0'].get('limiter_geometry_tensor') is not None:
                sample['state_t0']['limiter_geometry_tensor'] = normalize(
                    sample['state_t0']['limiter_geometry_tensor'], 'state_t0.limiter_geometry_tensor'
                )
            if len(sample['state_t0'].get('global_scalars', {})) > 0:
                global_vals = np.asarray(list(sample['state_t0']['global_scalars'].values()), dtype=np.float32)
                normalized_vals = normalize(global_vals, 'state_t0.global_scalars')
                for key, value in zip(sample['state_t0']['global_scalars'].keys(), normalized_vals):
                    sample['state_t0']['global_scalars'][key] = float(value)
            if sample['state_t0'].get('pre_shot_context') is not None:
                sample['state_t0']['pre_shot_context'] = normalize(
                    sample['state_t0']['pre_shot_context'], 'state_t0.pre_shot_context'
                )

            for state_t in sample['state_trajectory']:
                for var_name in list(state_t.get('kinetic_profiles', {}).keys()):
                    state_t['kinetic_profiles'][var_name] = normalize(
                        state_t['kinetic_profiles'][var_name], f'state_trajectory.kinetic_profiles.{var_name}'
                    )
                for var_name in list(state_t.get('transport_coefficients', {}).keys()):
                    state_t['transport_coefficients'][var_name] = normalize(
                        state_t['transport_coefficients'][var_name], f'state_trajectory.transport_coefficients.{var_name}'
                    )
                for var_name in list(state_t.get('nbi_sources', {}).keys()):
                    state_t['nbi_sources'][var_name] = normalize(
                        state_t['nbi_sources'][var_name], f'state_trajectory.nbi_sources.{var_name}'
                    )
                for var_name in list(state_t.get('other_profiles', {}).keys()):
                    state_t['other_profiles'][var_name] = normalize(
                        state_t['other_profiles'][var_name], f'state_trajectory.other_profiles.{var_name}'
                    )
                if state_t.get('geometry_tensor') is not None:
                    state_t['geometry_tensor'] = normalize(
                        state_t['geometry_tensor'], 'state_trajectory.geometry_tensor'
                    )
                if state_t.get('limiter_geometry_tensor') is not None:
                    state_t['limiter_geometry_tensor'] = normalize(
                        state_t['limiter_geometry_tensor'], 'state_trajectory.limiter_geometry_tensor'
                    )
                if len(state_t.get('global_scalars', {})) > 0:
                    global_vals = np.asarray(list(state_t['global_scalars'].values()), dtype=np.float32)
                    normalized_vals = normalize(global_vals, 'state_trajectory.global_scalars')
                    for key, value in zip(state_t['global_scalars'].keys(), normalized_vals):
                        state_t['global_scalars'][key] = float(value)
                if state_t.get('pre_shot_context') is not None:
                    state_t['pre_shot_context'] = normalize(
                        state_t['pre_shot_context'], 'state_trajectory.pre_shot_context'
                    )

            if 'NI' in normalization_stats:
                stats = normalization_stats['NI']
                sample['ni_trajectory'] = (sample['ni_trajectory'] - stats['mean']) / stats['std']

        self.normalization_stats = normalization_stats
        logger.info(f"✓ Normalization applied to {len(self.data)} samples across {len(normalization_stats)} component groups")

    def save(self, output_path: str) -> None:
        """
        Save the processed dataset to disk.

        The saved artifact includes the raw extracted samples and metadata so the
        expensive CDF parsing step does not need to run again on the next launch.
        Includes normalization statistics for reproducibility.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'data': self.data,
            'metadata': self.metadata,
            'cdf_paths': self.cdf_paths,
            'max_timesteps': self.max_timesteps,
            'normalize': self.normalize,
            'device': self.device,
            'normalization_stats': getattr(self, 'normalization_stats', None),
        }
        torch.save(payload, output_path)
        logger.info(f"✓ Saved dataset to {output_path}")

    @classmethod
    def load(cls, input_path: str, device: str = 'cpu') -> 'PPPLPlasmaDataset':
        """
        Load a previously saved dataset artifact.

        This reconstructs the dataset object without re-reading the original CDF files.
        Restores normalization statistics for reproducibility.
        """
        payload = torch.load(input_path, map_location='cpu', weights_only=False)
        dataset = cls.__new__(cls)
        dataset.cdf_paths = payload.get('cdf_paths', [])
        dataset.max_timesteps = payload.get('max_timesteps')
        dataset.normalize = payload.get('normalize', True)
        dataset.device = device
        dataset.data = payload['data']
        dataset.metadata = payload['metadata']
        dataset.normalization_stats = payload.get('normalization_stats', None)
        logger.info(f"✓ Loaded saved dataset from {input_path}")
        return dataset


def _slice_dataset_by_timestep_window(dataset: PPPLPlasmaDataset,
                                      window_start: int,
                                      window_end: int,
                                      split_name: str) -> PPPLPlasmaDataset:
    """Create a split dataset by slicing each sample trajectory to [start, end] inclusive."""
    split_dataset = PPPLPlasmaDataset.__new__(PPPLPlasmaDataset)
    split_dataset.cdf_paths = list(dataset.cdf_paths)
    split_dataset.max_timesteps = dataset.max_timesteps
    split_dataset.normalize = dataset.normalize
    split_dataset.device = dataset.device
    split_dataset.normalization_stats = getattr(dataset, 'normalization_stats', None)
    split_dataset.data = []
    split_dataset.metadata = []

    for idx, sample in enumerate(dataset.data):
        split_sample = {
            'cdf_path': sample.get('cdf_path'),
        }

        # Keep full initial condition to preserve interface across phases.
        split_sample['state_t0'] = copy.deepcopy(sample['state_t0'])

        ni_traj = np.asarray(sample['ni_trajectory'])
        max_index = ni_traj.shape[0] - 1
        start = max(0, window_start)
        end = min(window_end, max_index)
        if end < start:
            continue

        split_sample['ni_trajectory'] = ni_traj[start:end + 1].copy()

        if 'state_trajectory' in sample:
            state_traj = sample['state_trajectory']
            split_sample['state_trajectory'] = copy.deepcopy(state_traj[start:end + 1])

        split_dataset.data.append(split_sample)

        split_meta = dict(dataset.metadata[idx]) if idx < len(dataset.metadata) else {'shot': idx}
        split_meta.update({
            'split': split_name,
            'window_start': int(start),
            'window_end': int(end),
            'n_timesteps': int(end - start + 1),
        })
        split_dataset.metadata.append(split_meta)

    return split_dataset


# ==============================================================================
# SECTION 4: Main Execution and Testing
# ==============================================================================

def main():
    """
    Test the data pipeline with available CDF files.
    """
    # Find available CDF files
    mentorship_path = Path('/scratch/gpfs/USER')
    cdf_files = list(mentorship_path.glob('**/*.CDF'))
    
    if not cdf_files:
        logger.error("No CDF files found!")
        return
    
    logger.info(f"Found {len(cdf_files)} CDF files")
    
    # Test extraction from first file, then build a dataset from all files.
    cdf_path = str(cdf_files[0])
    logger.info(f"\n{'='*80}")
    logger.info("Testing extraction from: " + Path(cdf_path).name)
    logger.info('='*80)
    
    try:
        extractor = CDFVariableExtractor(cdf_path, verbose=True)
        
        # Extract components
        state = extractor.extract_full_state_vector_t0()
        
        # Print extraction summary
        logger.info("\n" + "="*80)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*80)
        
        for category, data in state.items():
            if isinstance(data, dict):
                logger.info(f"\n{category}: {len(data)} items")
                for key, val in list(data.items())[:3]:  # Show first 3
                    if isinstance(val, np.ndarray):
                        logger.info(f"  {key}: shape {val.shape}, dtype {val.dtype}")
                    else:
                        logger.info(f"  {key}: {val}")
            elif isinstance(data, np.ndarray):
                logger.info(f"\n{category}: shape {data.shape}, dtype {data.dtype}")
        
        extractor.close()
        
        # Test dataset loading for all available CDF files
        logger.info("\n" + "="*80)
        logger.info("Testing PyTorch Dataset loading...")
        logger.info("="*80)
        
        dataset = PPPLPlasmaDataset([str(path) for path in cdf_files], max_timesteps=1000, device='cpu')
        
        sample = dataset[0]
        logger.info(f"Sample state_t0 shape: {sample['state_t0'].shape}")
        logger.info(f"Sample trajectory shape: {sample['ni_trajectory'].shape}")
        logger.info(f"Shot metadata: {dataset.get_metadata(0)}")

        # Save the full processed dataset to disk so it can be reused without re-parsing CDFs.
        processed_dir = mentorship_path / 'PROJECT' / 'processed_data'
        dataset_path = processed_dir / 'dataset.pt'
        dataset.save(str(dataset_path))
        logger.info(f"Saved dataset artifact: {dataset_path}")

        # Save canonical split datasets for cross-phase consistency.
        train_dataset = _slice_dataset_by_timestep_window(
            dataset, TRAIN_TIMESTEPS[0], TRAIN_TIMESTEPS[1], 'train'
        )
        val_dataset = _slice_dataset_by_timestep_window(
            dataset, VAL_TIMESTEPS[0], VAL_TIMESTEPS[1], 'val'
        )
        test_dataset = _slice_dataset_by_timestep_window(
            dataset, TEST_TIMESTEPS[0], TEST_TIMESTEPS[1], 'test'
        )

        train_path = processed_dir / 'dataset_train.pt'
        val_path = processed_dir / 'dataset_val.pt'
        test_path = processed_dir / 'dataset_test.pt'

        train_dataset.save(str(train_path))
        val_dataset.save(str(val_path))
        test_dataset.save(str(test_path))

        split_manifest = {
            'train': {'path': str(train_path), 'timesteps': [TRAIN_TIMESTEPS[0], TRAIN_TIMESTEPS[1]]},
            'val': {'path': str(val_path), 'timesteps': [VAL_TIMESTEPS[0], VAL_TIMESTEPS[1]]},
            'test': {'path': str(test_path), 'timesteps': [TEST_TIMESTEPS[0], TEST_TIMESTEPS[1]]},
        }
        split_manifest_path = processed_dir / 'dataset_splits.json'
        with open(split_manifest_path, 'w') as split_file:
            json.dump(split_manifest, split_file, indent=2)

        logger.info(f"Saved split artifacts: {train_path.name}, {val_path.name}, {test_path.name}")
        logger.info(f"Saved split manifest: {split_manifest_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
