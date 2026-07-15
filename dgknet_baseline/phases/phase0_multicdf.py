"""
Phase 0: Multi-CDF Data Pipeline for PPPL TRANSP Plasma Simulation
"""




import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import netCDF4 as nc
import torch
try:
    from .phase0_data_pipeline import CDFVariableExtractor
except ImportError:
    from phase0_data_pipeline import CDFVariableExtractor




import numpy as np
import concurrent.futures




logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




CDF_DIRECTORIES = [
  Path('/scratch/gpfs/USER/cdf')
]




OUTPUT_DIR = Path('/scratch/gpfs/USER/datasets')


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_device(device_arg: Optional[str] = None) -> torch.device:
  if device_arg:
      device = torch.device(device_arg)
      if device.type.startswith('cuda') and not torch.cuda.is_available():
          logger.warning('CUDA requested but unavailable; using CPU instead.')
          return torch.device('cpu')
      return device
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def discover_cdf_files(source_dirs: List[Path]) -> List[Path]:
  cdf_files: List[Path] = []
  for source_dir in source_dirs:
      if not source_dir.exists():
          continue
      cdf_files.extend(sorted(source_dir.glob('*.CDF')))
      cdf_files.extend(sorted(source_dir.glob('*.cdf')))
  return sorted(cdf_files)


def has_limiter_contour(cdf_path: Path) -> bool:
  try:
      with nc.Dataset(str(cdf_path)) as dataset:
          return 'RLIM' in dataset.variables and 'YLIM' in dataset.variables
  except Exception:
      return False


def select_reference_limiter_cdf(cdf_paths: List[Path], explicit_reference: Optional[Path] = None) -> Optional[Path]:
  if explicit_reference is not None:
      if explicit_reference.exists():
          logger.info(f'Using explicit limiter reference CDF: {explicit_reference.name}')
          return explicit_reference
      logger.warning(f'Explicit limiter reference CDF not found: {explicit_reference}')

  for cdf_path in cdf_paths:
      if has_limiter_contour(cdf_path):
          logger.info(f'Using shared limiter reference CDF: {cdf_path.name}')
          return cdf_path

  logger.warning('No CDF with RLIM/YLIM found; limiter fallback will be unavailable.')
  return None




def split_cdfs_by_shot(cdf_paths: List[Path],
                     train_ratio: float = 0.70,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
  if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
      raise ValueError('train/val/test ratios must sum to 1.0')




  rng = random.Random(seed)
  indices = list(range(len(cdf_paths)))
  rng.shuffle(indices)




  n_total = len(cdf_paths)
  n_train = int(n_total * train_ratio)
  n_val = int(n_total * val_ratio)
  n_test = n_total - n_train - n_val




  train = [cdf_paths[i] for i in indices[:n_train]]
  val = [cdf_paths[i] for i in indices[n_train:n_train + n_val]]
  test = [cdf_paths[i] for i in indices[n_train + n_val:]]




  logger.info(f'Split {n_total} CDFs into {len(train)} train, {len(val)} val, {len(test)} test shots')
  return train, val, test








def build_sample(cdf_path: Path, verbose: bool = False, limiter_reference_path: Optional[Path] = None) -> Optional[Dict]:
    cdf_path = Path(cdf_path)
    extractor = None
    try:
        extractor = CDFVariableExtractor(
                str(cdf_path),
                verbose=verbose,
                limiter_reference_path=str(limiter_reference_path) if limiter_reference_path else None,
        )
        state_t0 = extractor.extract_full_state_vector_t0()
        state_trajectory = extractor.extract_full_state_trajectory()
        ni_trajectory = extractor.extract_ion_density_trajectory()
    except Exception as exc:
        logger.warning(f'Skipping unsupported CDF {cdf_path.name}: {exc}')
        return None
    finally:
        if extractor is not None:
            extractor.close()




    if state_t0 is None or state_trajectory is None or ni_trajectory is None:
        logger.warning(f'Skipping incomplete CDF: {cdf_path.name}')
        return None




    return {
        'cdf_path': str(cdf_path),
        'shot_id': cdf_path.stem,
        'state_t0': state_t0,
        'state_trajectory': state_trajectory,
        'ni_trajectory': ni_trajectory,
    }








def build_sample_metadata(sample: Dict, shot_index: int) -> Dict[str, object]:
  return {
      'shot': shot_index,
      'cdf_path': sample['cdf_path'],
      'n_timesteps': int(sample['ni_trajectory'].shape[0]),
      'n_radial': int(sample['ni_trajectory'].shape[1])
  }








def _collect_values(values_by_key: Dict[str, List[np.ndarray]], key: str, value) -> None:
  if value is None:
      return
  if isinstance(value, dict):
      for sub_key, sub_value in value.items():
          _collect_values(values_by_key, f'{key}.{sub_key}', sub_value)
      return
  arr = np.asarray(value, dtype=np.float32)
  values_by_key.setdefault(key, []).append(arr)








def _update_normalization_accumulators(accumulators: Dict[str, Dict[str, float]], key: str, value) -> None:
  if value is None:
      return
  if isinstance(value, dict):
      for sub_key, sub_value in value.items():
          _update_normalization_accumulators(accumulators, f'{key}.{sub_key}', sub_value)
      return
  arr = np.asarray(value, dtype=np.float32).reshape(-1)
  if arr.size == 0:
      return
  stats = accumulators.setdefault(key, {'count': 0.0, 'sum': 0.0, 'sum_sq': 0.0})
  stats['count'] += float(arr.size)
  stats['sum'] += float(arr.sum())
  stats['sum_sq'] += float((arr ** 2).sum())


def _finalize_normalization_stats(accumulators: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
  normalization_stats: Dict[str, Dict[str, float]] = {}
  for key, stats in accumulators.items():
      count = stats['count']
      if count <= 0.0:
          continue
      mean_val = stats['sum'] / count
      var_val = max(stats['sum_sq'] / count - mean_val ** 2, 0.0)
      std_val = float(np.sqrt(var_val))
      if std_val < 1e-10:
          std_val = 1.0
      normalization_stats[key] = {'mean': float(mean_val), 'std': std_val}
      logger.info(f'  Normalization stat {key}: mean={mean_val:.4e}, std={std_val:.4e}')
  return normalization_stats


def compute_normalization_stats(train_samples: List[Dict]) -> Dict[str, Dict[str, float]]:
  accumulators: Dict[str, Dict[str, float]] = {}




  for sample in train_samples:
      _update_normalization_accumulators(accumulators, 'state_t0.kinetic_profiles', sample['state_t0'].get('kinetic_profiles', {}))
      _update_normalization_accumulators(accumulators, 'state_t0.transport_coefficients', sample['state_t0'].get('transport_coefficients', {}))
      _update_normalization_accumulators(accumulators, 'state_t0.nbi_sources', sample['state_t0'].get('nbi_sources', {}))
      _update_normalization_accumulators(accumulators, 'state_t0.other_profiles', sample['state_t0'].get('other_profiles', {}))
      _update_normalization_accumulators(accumulators, 'state_t0.geometry_tensor', sample['state_t0'].get('geometry_tensor'))
      _update_normalization_accumulators(accumulators, 'state_t0.global_scalars', np.asarray(list(sample['state_t0'].get('global_scalars', {}).values()), dtype=np.float32))




      for state_t in sample['state_trajectory']:
          _update_normalization_accumulators(accumulators, 'state_trajectory.kinetic_profiles', state_t.get('kinetic_profiles', {}))
          _update_normalization_accumulators(accumulators, 'state_trajectory.transport_coefficients', state_t.get('transport_coefficients', {}))
          _update_normalization_accumulators(accumulators, 'state_trajectory.nbi_sources', state_t.get('nbi_sources', {}))
          _update_normalization_accumulators(accumulators, 'state_trajectory.other_profiles', state_t.get('other_profiles', {}))
          _update_normalization_accumulators(accumulators, 'state_trajectory.geometry_tensor', state_t.get('geometry_tensor'))
          _update_normalization_accumulators(accumulators, 'state_trajectory.global_scalars', np.asarray(list(state_t.get('global_scalars', {}).values()), dtype=np.float32))




      _update_normalization_accumulators(accumulators, 'NI', sample['ni_trajectory'])



  return _finalize_normalization_stats(accumulators)








  return _finalize_normalization_stats(accumulators)


def compute_normalization_stats_from_paths(cdf_paths: List[Path],
                                           max_timesteps: Optional[int] = None,
                                           verbose: bool = False,
                                           max_workers: Optional[int] = None,
                                           batch_size: int = 50,
                                           limiter_reference_path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
  accumulators: Dict[str, Dict[str, float]] = {}

  for i in range(0, len(cdf_paths), batch_size):
      batch_paths = cdf_paths[i:i + batch_size]
      logger.info(f"Computing normalization stats from batch {i//batch_size + 1} with {len(batch_paths)} CDFs...")

      if max_workers is None or max_workers == 1:
          batch_samples = []
          for cdf_path in batch_paths:
              sample = build_sample(cdf_path, verbose=verbose, limiter_reference_path=limiter_reference_path)
              if sample is not None:
                  if max_timesteps is not None:
                      sample['state_trajectory'] = sample['state_trajectory'][:max_timesteps]
                      sample['ni_trajectory'] = sample['ni_trajectory'][:max_timesteps, :]
                  batch_samples.append(sample)
      else:
          with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
              futures = [executor.submit(build_sample, cdf_path, verbose, limiter_reference_path) for cdf_path in batch_paths]
              batch_samples = [f.result() for f in concurrent.futures.as_completed(futures) if f.result() is not None]

      for sample in batch_samples:
          _update_normalization_accumulators(accumulators, 'state_t0.kinetic_profiles', sample['state_t0'].get('kinetic_profiles', {}))
          _update_normalization_accumulators(accumulators, 'state_t0.transport_coefficients', sample['state_t0'].get('transport_coefficients', {}))
          _update_normalization_accumulators(accumulators, 'state_t0.nbi_sources', sample['state_t0'].get('nbi_sources', {}))
          _update_normalization_accumulators(accumulators, 'state_t0.other_profiles', sample['state_t0'].get('other_profiles', {}))
          _update_normalization_accumulators(accumulators, 'state_t0.geometry_tensor', sample['state_t0'].get('geometry_tensor'))
          _update_normalization_accumulators(accumulators, 'state_t0.global_scalars', np.asarray(list(sample['state_t0'].get('global_scalars', {}).values()), dtype=np.float32))

          for state_t in sample['state_trajectory']:
              _update_normalization_accumulators(accumulators, 'state_trajectory.kinetic_profiles', state_t.get('kinetic_profiles', {}))
              _update_normalization_accumulators(accumulators, 'state_trajectory.transport_coefficients', state_t.get('transport_coefficients', {}))
              _update_normalization_accumulators(accumulators, 'state_trajectory.nbi_sources', state_t.get('nbi_sources', {}))
              _update_normalization_accumulators(accumulators, 'state_trajectory.other_profiles', state_t.get('other_profiles', {}))
              _update_normalization_accumulators(accumulators, 'state_trajectory.geometry_tensor', state_t.get('geometry_tensor'))
              _update_normalization_accumulators(accumulators, 'state_trajectory.global_scalars', np.asarray(list(state_t.get('global_scalars', {}).values()), dtype=np.float32))

          _update_normalization_accumulators(accumulators, 'NI', sample['ni_trajectory'])

      del batch_samples

  return _finalize_normalization_stats(accumulators)


def apply_normalization(samples: List[Dict], normalization_stats: Dict[str, Dict[str, float]]) -> None:
  def normalize(value, key):
      if value is None or key not in normalization_stats:
          return value
      stats = normalization_stats[key]
      return (np.asarray(value, dtype=np.float32) - stats['mean']) / stats['std']




  for sample in samples:
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
      if len(sample['state_t0'].get('global_scalars', {})) > 0:
          global_vals = np.asarray(list(sample['state_t0']['global_scalars'].values()), dtype=np.float32)
          normalized_vals = normalize(global_vals, 'state_t0.global_scalars')
          for key, value in zip(sample['state_t0']['global_scalars'].keys(), normalized_vals):
              sample['state_t0']['global_scalars'][key] = float(value)




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
          if len(state_t.get('global_scalars', {})) > 0:
              global_vals = np.asarray(list(state_t['global_scalars'].values()), dtype=np.float32)
              normalized_vals = normalize(global_vals, 'state_trajectory.global_scalars')
              for key, value in zip(state_t['global_scalars'].keys(), normalized_vals):
                  state_t['global_scalars'][key] = float(value)




      sample['ni_trajectory'] = normalize(sample['ni_trajectory'], 'NI')








def save_dataset_payload(samples: List[Dict], metadata: List[Dict], cdf_paths: List[str], output_path: Path, device: str, normalization_stats: Dict[str, Dict[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            'data': samples,
            'metadata': metadata,
            'cdf_paths': cdf_paths,
            'max_timesteps': None,
            'normalize': True,
            'device': device,
            'normalization_stats': normalization_stats,
    }
    torch.save(payload, str(output_path))
    logger.info(f'Saved dataset payload: {output_path} ({len(samples)} shots)')


def build_split_samples(cdf_paths: List[Path],
                      max_timesteps: Optional[int] = None,
                      verbose: bool = False,
                      max_workers: Optional[int] = None,
                      batch_size: int = 50,
                      limiter_reference_path: Optional[Path] = None) -> Tuple[List[Dict], List[Dict]]:
  """Build samples and metadata for a shot split in batches to manage memory."""
  samples: List[Dict] = []
  metadata: List[Dict] = []




  for i in range(0, len(cdf_paths), batch_size):
      batch_paths = cdf_paths[i:i + batch_size]
      logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_paths)} CDFs...")




      if max_workers is None or max_workers == 1:
          batch_samples = []
          for cdf_path in batch_paths:
              sample = build_sample(cdf_path, verbose=verbose, limiter_reference_path=limiter_reference_path)
              if sample is not None:
                  if max_timesteps is not None:
                      sample['state_trajectory'] = sample['state_trajectory'][:max_timesteps]
                      sample['ni_trajectory'] = sample['ni_trajectory'][:max_timesteps, :]
                  batch_samples.append(sample)
      else:
          with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
              futures = [executor.submit(build_sample, cdf_path, verbose, limiter_reference_path) for cdf_path in batch_paths]
              batch_samples = [f.result() for f in concurrent.futures.as_completed(futures) if f.result() is not None]




      for idx, sample in enumerate(batch_samples):
          if max_timesteps is not None and 'state_trajectory' in sample:
              sample['state_trajectory'] = sample['state_trajectory'][:max_timesteps]
              sample['ni_trajectory'] = sample['ni_trajectory'][:max_timesteps, :]
          samples.append(sample)
          metadata.append(build_sample_metadata(sample, i + idx))
          print(f'completed {idx} for sample')




  return samples, metadata


def combine_temp_files(output_path: Path, device: str, normalization_stats: Dict[str, Dict[str, float]], chunk_size: int = 10) -> bool:
  """Combine existing temp files into the final dataset payload, processing in chunks to manage memory."""
  import glob
  temp_pattern = str(output_path.parent / f'{output_path.stem}.temp_*.pt')
  temp_files = sorted(glob.glob(temp_pattern))
  temp_files = [Path(f) for f in temp_files]
  
  if not temp_files:
      logger.info(f'No temp files found for {output_path.name}, will process from scratch.')
      return False
  
  logger.info(f'Found {len(temp_files)} temp files for {output_path.name}, combining in chunks of {chunk_size}...')
  
  # Combine in chunks to avoid memory issues
  partial_files = []
  for i in range(0, len(temp_files), chunk_size):
      chunk = temp_files[i:i + chunk_size]
      logger.info(f'Combining chunk {i//chunk_size + 1} with {len(chunk)} files...')
      
      chunk_samples = []
      chunk_metadata = []
      chunk_cdf_paths = []
      for temp_file in chunk:
          payload = torch.load(temp_file, weights_only=False)
          chunk_samples.extend(payload['data'])
          chunk_metadata.extend(payload['metadata'])
          chunk_cdf_paths.extend(payload['cdf_paths'])
      
      partial_file = output_path.with_suffix(f'.partial_{i//chunk_size}.pt')
      save_dataset_payload(chunk_samples, chunk_metadata, chunk_cdf_paths, partial_file, device, normalization_stats)
      partial_files.append(partial_file)
      del chunk_samples, chunk_metadata, chunk_cdf_paths  # Free memory
  
  # Now combine the partials
  all_samples = []
  all_metadata = []
  all_cdf_paths = []
  for partial_file in partial_files:
      logger.info(f'Loading partial file: {partial_file}')
      payload = torch.load(partial_file, weights_only=False)
      all_samples.extend(payload['data'])
      all_metadata.extend(payload['metadata'])
      all_cdf_paths.extend(payload['cdf_paths'])
  
  save_dataset_payload(all_samples, all_metadata, all_cdf_paths, output_path, device, normalization_stats)
  
  # Clean up temp and partial files
  for temp_file in temp_files:
      temp_file.unlink(missing_ok=True)
  for partial_file in partial_files:
      partial_file.unlink(missing_ok=True)
  logger.info(f'Successfully combined {len(temp_files)} temp files into {output_path}')
  return True

def process_split_in_batches(cdf_paths: List[Path],
                           normalization_stats: Dict[str, Dict[str, float]],
                           output_path: Path,
                           device: str,
                           max_timesteps: Optional[int] = None,
                           verbose: bool = False,
                           max_workers: Optional[int] = None,
                           batch_size: int = 50,
                           limiter_reference_path: Optional[Path] = None) -> None:
  """Process a split in batches, apply normalization, save partial payloads, then merge to manage memory."""
  temp_files = []




  for i in range(0, len(cdf_paths), batch_size):
      batch_paths = cdf_paths[i:i + batch_size]
      logger.info(f"Processing batch {i//batch_size + 1} for {output_path.name} with {len(batch_paths)} CDFs...")




      if max_workers is None or max_workers == 1:
          batch_samples = []
          for cdf_path in batch_paths:
              sample = build_sample(cdf_path, verbose=verbose, limiter_reference_path=limiter_reference_path)
              if sample is not None:
                  if max_timesteps is not None:
                      sample['state_trajectory'] = sample['state_trajectory'][:max_timesteps]
                      sample['ni_trajectory'] = sample['ni_trajectory'][:max_timesteps, :]
                  batch_samples.append(sample)
      else:
          with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
              futures = [executor.submit(build_sample, cdf_path, verbose, limiter_reference_path) for cdf_path in batch_paths]
              batch_samples = [f.result() for f in concurrent.futures.as_completed(futures) if f.result() is not None]




      apply_normalization(batch_samples, normalization_stats)




      batch_metadata = [build_sample_metadata(sample, i + idx) for idx, sample in enumerate(batch_samples)]
      batch_cdf_paths = [s['cdf_path'] for s in batch_samples]




      # Save partial to temp file
      temp_file = output_path.with_suffix(f'.temp_{i//batch_size}.pt')
      save_dataset_payload(batch_samples, batch_metadata, batch_cdf_paths, temp_file, device, normalization_stats)
      temp_files.append(temp_file)
      del batch_samples, batch_metadata, batch_cdf_paths  # Scrub memory




  # Merge all temp files into final payload
  all_samples = []
  all_metadata = []
  all_cdf_paths = []
  for temp_file in temp_files:
      payload = torch.load(temp_file, weights_only=False)
      all_samples.extend(payload['data'])
      all_metadata.extend(payload['metadata'])
      all_cdf_paths.extend(payload['cdf_paths'])




  save_dataset_payload(all_samples, all_metadata, all_cdf_paths, output_path, device, normalization_stats)




  # Clean up temp files
  for temp_file in temp_files:
      temp_file.unlink(missing_ok=True)








def main():
  parser = argparse.ArgumentParser(description='Build multi-shot Phase 0 datasets from CDF files.')
  parser.add_argument('--data-dirs', nargs='+', default=[str(p) for p in CDF_DIRECTORIES],
                      help='Input directories to search for CDF files.')
  parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Directory to store output dataset files.')
  parser.add_argument('--device', default='cuda', help='Torch device to use for processing (cpu or cuda).')
  parser.add_argument('--seed', type=int, default=42, help='Random seed for shot splitting.')
  parser.add_argument('--max-cdfs', type=int, default=None, help='Limit the number of CDF shots processed for debugging.')
  parser.add_argument('--max-timesteps', type=int, default=None, help='Limit timesteps per shot when extracting trajectories.')
  parser.add_argument('--max-workers', type=int, default=6, help='Number of parallel workers for CDF processing (default: sequential).')
  parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing CDFs to manage memory.')
  parser.add_argument('--reference-limiter-cdf', default='', help='Optional CDF or NPZ path whose RLIM/YLIM contour is reused when shots omit limiter geometry.')
  parser.add_argument('--verbose', action='store_true', help='Print extra progress information.')
  args = parser.parse_args()

  device = get_device(args.device)
  source_dirs = [Path(path) for path in args.data_dirs]
  cdf_files = discover_cdf_files(source_dirs)
  if args.max_cdfs is not None:
      cdf_files = cdf_files[:args.max_cdfs]

  explicit_reference = Path(args.reference_limiter_cdf) if args.reference_limiter_cdf else None

  # Automatically use a stored reference limiter contour if available.
  default_reference = Path('processed_data/limiter_reference.npz')
  if explicit_reference is None and default_reference.exists():
      explicit_reference = default_reference
      logger.info(f'Using stored limiter reference file: {default_reference}')

  reference_limiter_path = select_reference_limiter_cdf(cdf_files, explicit_reference)

  if not cdf_files:
      logger.warning('No CDF files found in configured input directories.')
      return

  logger.info(f'Found {len(cdf_files)} CDF files')
  for cdf_file in cdf_files[:5]:
      logger.info(f'  - {cdf_file.name}')
  if len(cdf_files) > 5:
      logger.info(f'  ... and {len(cdf_files) - 5} more')

  train_paths, val_paths, test_paths = split_cdfs_by_shot(
      cdf_files,
      train_ratio=TRAIN_RATIO,
      val_ratio=VAL_RATIO,
      test_ratio=TEST_RATIO,
      seed=args.seed,
  )

  logger.info('Computing normalization stats from first 10 batches of train split...')
  train_subset = train_paths[:500]  # First 10 batches (50*10=500 CDFs)
  normalization_stats = compute_normalization_stats_from_paths(
      train_subset,
      max_timesteps=args.max_timesteps,
      verbose=args.verbose,
      max_workers=args.max_workers,
      batch_size=args.batch_size,
      limiter_reference_path=reference_limiter_path,
  )

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  train_path = output_dir / 'dataset_train.pt'
  val_path = output_dir / 'dataset_val.pt'
  test_path = output_dir / 'dataset_test.pt'

  # Check if train dataset already exists or can be combined from temps
  if train_path.exists():
      logger.info(f'Train dataset already exists: {train_path}')
  else:
      # Try to combine existing temp files
      combined = combine_temp_files(train_path, str(device), normalization_stats)
      if not combined:
          logger.info('Building and saving train split samples in batches...')
          process_split_in_batches(
              train_paths,
              normalization_stats,
              train_path,
              str(device),
              args.max_timesteps,
              args.verbose,
              args.max_workers,
              args.batch_size,
              reference_limiter_path,
          )

  logger.info('Building and saving validation split samples in batches...')
  process_split_in_batches(
      val_paths,
      normalization_stats,
      val_path,
      str(device),
      args.max_timesteps,
      args.verbose,
      args.max_workers,
      args.batch_size,
      reference_limiter_path,
  )

  logger.info('Building and saving test split samples in batches...')
  process_split_in_batches(
      test_paths,
      normalization_stats,
      test_path,
      str(device),
      args.max_timesteps,
      args.verbose,
      args.max_workers,
      args.batch_size,
      reference_limiter_path,
  )

  metadata = {
      'n_cdfs': len(cdf_files),
      'train_cdfs': len(train_paths),
      'val_cdfs': len(val_paths),
      'test_cdfs': len(test_paths),
      'split_strategy': 'CDF-based (80/10/10 by CDF shot)',
      'device': str(device),
      'train_path': str(train_path),
      'val_path': str(val_path),
      'test_path': str(test_path),
      'normalization_keys': list(normalization_stats.keys()),
      'max_timesteps': args.max_timesteps,
      'seed': args.seed,
  }

  with open(output_dir / 'multicdf_metadata.json', 'w') as f:
      json.dump(metadata, f, indent=2)

  logger.info('✓ Multi-CDF pipeline complete.')
  logger.info(f'  Train: {train_path}')
  logger.info(f'  Val:   {val_path}')
  logger.info(f'  Test:  {test_path}')








if __name__ == '__main__':
  main()












