"""Build and save limiter reference NPZ files per machine.

This script reads the manifest produced by `import_cdfs.py`, opens each CDF
and attempts to extract the RLIM/YLIM limiter contour. It writes a per-machine
`limiter_{MACHINE}.npz` under output directory. If a CDF lacks RLIM/YLIM, the
script will skip that file but will try others; if no contour is found for a
machine a synthetic fallback is saved (but a warning is emitted).

Usage:
  python build_limiters.py --manifest /scratch/gpfs/USER/PROJECT/processed_data/cdf_manifest.json \
      --out /scratch/gpfs/USER/PROJECT/processed_data/limiters
"""
import argparse
import json
from pathlib import Path
import numpy as np
import logging
import sys

# Ensure repo path for local imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dgknet_baseline.phases.phase0_data_pipeline import CDFVariableExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('build_limiters')


def extract_first_contour(cdf_paths, limiter_out_dir: Path):
    found = False
    for p in cdf_paths:
        try:
            extractor = CDFVariableExtractor(p, verbose=False)
            contour = extractor._extract_limiter_contour(extractor.cdf, time_index=0)
            extractor.close()
            if contour is not None:
                rlim, ylim = contour
                limiter_out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(limiter_out_dir / 'limiter.npz', rlim=rlim.astype(np.float32), ylim=ylim.astype(np.float32))
                logger.info(f"Saved limiter from {p} to {limiter_out_dir / 'limiter.npz'}")
                found = True
                break
        except Exception as e:
            logger.warning(f"Failed to read contour from {p}: {e}")

    if not found:
        # Create synthetic fallback (small circular limiter)
        logger.warning(f"No limiter contour found in provided CDFs for {limiter_out_dir.name}. Creating synthetic fallback.")
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        rlim = (0.5 + 0.1 * np.cos(theta)).astype(np.float32)
        ylim = (0.0 + 0.1 * np.sin(theta)).astype(np.float32)
        limiter_out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(limiter_out_dir / 'limiter.npz', rlim=rlim, ylim=ylim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path, help='Manifest JSON produced by import_cdfs.py')
    parser.add_argument('--out', required=True, type=Path, help='Output directory to write limiters (per-machine subdirs)')
    args = parser.parse_args()

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    for machine, files in manifest.items():
        logger.info(f"Processing machine {machine} ({len(files)} CDFs)")
        out_dir = args.out / machine
        extract_first_contour(files, out_dir)

    # Also write a combined reference (first available)
    # find first machine limiter and copy to root
    for machine in manifest.keys():
        candidate = args.out / machine / 'limiter.npz'
        if candidate.exists():
            dest = args.out / 'limiter_reference.npz'
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(candidate, dest)
            logger.info(f"Wrote combined limiter reference: {dest}")
            break


if __name__ == '__main__':
    main()
