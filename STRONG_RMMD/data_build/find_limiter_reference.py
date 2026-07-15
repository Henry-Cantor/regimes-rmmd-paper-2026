#!/usr/bin/env python3
"""Find and store a shared NSTX limiter contour from CDF files."""

import argparse
import json
import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import List, Optional

CANONICAL_N_PSI = 40


def discover_cdfs(source_dirs: List[Path]) -> List[Path]:
    cdf_files = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        cdf_files.extend(sorted(source_dir.glob('*.CDF')))
        cdf_files.extend(sorted(source_dir.glob('*.cdf')))
    return sorted(cdf_files)


def extract_limiter_contour(cdf_path: Path, time_index: int = 0) -> Optional[np.ndarray]:
    try:
        with nc.Dataset(str(cdf_path)) as ds:
            if 'RLIM' not in ds.variables or 'YLIM' not in ds.variables:
                return None
            rlim_raw = np.asarray(ds.variables['RLIM'][time_index], dtype=np.float32).reshape(-1)
            ylim_raw = np.asarray(ds.variables['YLIM'][time_index], dtype=np.float32).reshape(-1)
            if rlim_raw.shape[0] != ylim_raw.shape[0]:
                return None
            return np.stack([rlim_raw, ylim_raw], axis=1)
    except Exception:
        return None


def resample_contour(contour: np.ndarray, target_n_psi: int = CANONICAL_N_PSI) -> np.ndarray:
    n = contour.shape[0]
    if n == target_n_psi:
        return contour.astype(np.float32)
    s_old = np.linspace(0.0, 1.0, n)
    s_new = np.linspace(0.0, 1.0, target_n_psi)
    rlim = np.interp(s_new, s_old, contour[:, 0]).astype(np.float32)
    ylim = np.interp(s_new, s_old, contour[:, 1]).astype(np.float32)
    return np.stack([rlim, ylim], axis=1)


def contour_stats(contour: np.ndarray) -> dict:
    return {
        'shape': contour.shape,
        'norm': float(np.linalg.norm(contour)),
        'max_abs': float(np.max(np.abs(contour))),
        'mean_abs': float(np.mean(np.abs(contour))),
    }


def compare_contours(reference: np.ndarray, contour: np.ndarray) -> dict:
    if reference.shape != contour.shape:
        raise ValueError('Contour shapes must match for comparison')
    diff = contour - reference
    return {
        'l2': float(np.linalg.norm(diff)),
        'l2_rel': float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-6)),
        'max_abs': float(np.max(np.abs(diff))),
        'mean_abs': float(np.mean(np.abs(diff))),
    }


def save_reference(output_path: Path, contour: np.ndarray, source_cdf: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        rlim=contour[:, 0].astype(np.float32),
        ylim=contour[:, 1].astype(np.float32),
        source_cdf=source_cdf,
    )
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'source_cdf': source_cdf,
            'n_psi': contour.shape[0],
            'norm': float(np.linalg.norm(contour)),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Find a shared limiter contour for NSTX CDFs.')
    parser.add_argument('--data-dirs', nargs='+', default=['/scratch/gpfs/USER/cdf'],
                        help='Directories to search for CDF files.')
    parser.add_argument('--max-cdfs', type=int, default=None,
                        help='Limit number of CDFs scanned for debugging.')
    parser.add_argument('--output-file', default='processed_data/limiter_reference.npz',
                        help='Path to save the shared limiter contour.')
    parser.add_argument('--tolerance', type=float, default=1e-2,
                        help='Relative tolerance for contour similarity.')
    args = parser.parse_args()

    source_dirs = [Path(p) for p in args.data_dirs]
    cdf_files = discover_cdfs(source_dirs)
    if args.max_cdfs is not None:
        cdf_files = cdf_files[:args.max_cdfs]

    if not cdf_files:
        raise SystemExit('No CDF files found in provided directories.')

    print(f'Found {len(cdf_files)} CDF files to scan.')

    found = []
    for cdf_path in cdf_files:
        contour = extract_limiter_contour(cdf_path)
        if contour is None:
            continue
        contour = resample_contour(contour)
        found.append((cdf_path, contour))

    if not found:
        raise SystemExit('No CDF files contained RLIM/YLIM limiter contour data.')

    print(f'Found limiter contour in {len(found)} CDF files.')

    reference_path, reference_contour = found[0]
    print(f'Using reference contour from {reference_path.name}')
    stats = contour_stats(reference_contour)
    print(f'  Reference contour: shape={stats["shape"]}, norm={stats["norm"]:.4e}, max_abs={stats["max_abs"]:.4e}')

    outliers = []
    for cdf_path, contour in found[1:]:
        cmp_stats = compare_contours(reference_contour, contour)
        if cmp_stats['l2_rel'] > args.tolerance or cmp_stats['max_abs'] > 0.05:
            outliers.append((cdf_path.name, cmp_stats))

    print(f'  Outliers above tolerance: {len(outliers)}')
    if outliers:
        for name, cmp_stats in outliers[:10]:
            print(f'    {name}: l2_rel={cmp_stats["l2_rel"]:.4e}, max_abs={cmp_stats["max_abs"]:.4e}')

    save_reference(Path(args.output_file), reference_contour, str(reference_path))
    print(f'Saved limiter reference to: {args.output_file}')
    if outliers:
        print('WARNING: Some contours differ from reference beyond tolerance.')
    else:
        print('Limiter contours are consistent across scanned CDFs.')


if __name__ == '__main__':
    main()
