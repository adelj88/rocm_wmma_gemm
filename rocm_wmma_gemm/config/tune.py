#!/usr/bin/env python3

import subprocess
import json
import numpy as np
import optuna
import warnings
import re
import argparse
import sys
import random
from pathlib import Path

warnings.filterwarnings('ignore')

class OptunaWMMATuner:
    """Optuna-based TPE tuner for WMMA kernel optimization."""

    def __init__(self, max_shared_memory=65336, gpu_arch="gfx1100", baselines=None, random_seed=42):
        self.max_shared_memory = max_shared_memory
        self.gpu_arch = gpu_arch
        self.random_seed = random_seed

        # Set random seed for reproducible results
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Enhanced baseline configurations for WMMA
        if baselines is None:
            base_configs = [
                (4, 4, 4, 4, 256),
                (4, 4, 4, 4, 128),
                (4, 4, 4, 4, 64),
                (2, 4, 4, 4, 256),
                (2, 4, 4, 4, 128),
                (2, 4, 4, 4, 64),
                (4, 2, 4, 4, 256),
                (2, 2, 4, 4, 256),
                (2, 2, 4, 4, 128),
                (2, 2, 4, 4, 64),
                (4, 2, 2, 4, 256),
                (4, 2, 2, 4, 128),
                (4, 2, 2, 4, 64),
                (2, 4, 4, 2, 256),
                (2, 4, 4, 2, 128),
                (2, 4, 4, 2, 64),
                (4, 4, 4, 2, 256),
                (4, 4, 2, 2, 256),
                (4, 4, 2, 2, 128),
                (4, 4, 2, 2, 64),
                (2, 8, 4, 4, 256),
                (2, 8, 4, 4, 128),
                (2, 8, 4, 4, 64),
                (4, 8, 2, 1, 256),
                (4, 8, 2, 1, 128),
                (4, 8, 2, 1, 64),
                (4, 4, 1, 1, 256),
                (4, 4, 1, 1, 128),
                (4, 4, 1, 1, 64),
                (2, 1, 2, 2, 256),
                (2, 1, 2, 2, 128),
                (2, 1, 2, 2, 64),
            ]
            # Add both direct write variants
            self.baselines = []
            for cfg in base_configs:
                self.baselines.append(cfg + (False,))
                self.baselines.append(cfg + (True,))
        else:
            self.baselines = baselines

        # Parameter space definitions - now 6 parameters
        self.param_space = {
            'warps_m': [1, 2, 4, 8],
            'warps_n': [1, 2, 4, 8],
            'warp_tile_m': [1, 2, 4, 8],
            'warp_tile_n': [1, 2, 4, 8],
            'bits': [64, 128, 256],
            'use_direct_write': [False, True]
        }

        # Generate all valid configurations for reference
        self.valid_configs = self._generate_valid_configs()
        print(f"Generated {len(self.valid_configs)} valid configurations")

        # State tracking
        self.current_problem = None
        self.total_evaluations = 0
        self.best_config = None
        self.best_time = float('inf')
        self.improvement_history = []

    def _generate_valid_configs(self):
        """Generate all valid discrete configurations."""
        configs = []
        count = 0
        for warps_m in self.param_space['warps_m']:
            for warps_n in self.param_space['warps_n']:
                for warp_tile_m in self.param_space['warp_tile_m']:
                    for warp_tile_n in self.param_space['warp_tile_n']:
                        for bits in self.param_space['bits']:
                            for use_direct_write in self.param_space['use_direct_write']:
                                count += 1
                                config = (warps_m, warps_n, warp_tile_m, warp_tile_n, bits, use_direct_write)
                                if self._check_constraints(config):
                                    configs.append(config)

        print(f"Checked {count} total combinations, {len(configs)} valid after constraints")
        return configs

    def _check_constraints(self, config):
        """Check memory and resource constraints for WMMA."""
        warps_m, warps_n, warp_tile_m, warp_tile_n, bits, use_direct_write = config

        if warps_m == 0 or warps_n == 0 or warp_tile_m == 0 or warp_tile_n == 0:
            return False

        if bits == 0:
            return False

        # Memory constraint
        wmma_tile = 16
        block_m = warps_m * warp_tile_m * wmma_tile
        block_n = warps_n * warp_tile_n * wmma_tile
        block_k = wmma_tile
        lds_size = (block_m * block_k) + (block_k * block_n)
        memory_bytes = 2 * lds_size * 2  # Double buffering * sizeof(half)

        if memory_bytes > self.max_shared_memory:
            return False

        # Resource constraint - RDNA GPUs: max 32 warps per block
        total_warps = warps_m * warps_n
        if total_warps > 32:
            return False

        return True

    def _parse_benchmark_output(self, output):
        """Parse benchmark output to extract minimum timing."""
        lines = output.strip().split('\n')
        for line in lines:
            if 'dynamic_kernel/manual_time' in line:
                # Look for min_time_ms with optional suffix (k for kilo, M for mega)
                match = re.search(r'min_time_ms=([\d.]+)([kM])?', line)
                if match:
                    value = float(match.group(1))
                    suffix = match.group(2)
                    if suffix == 'k':
                        value *= 1000
                    elif suffix == 'M':
                        value *= 1000000
                    return value
        return None

    def _evaluate_config(self, M, N, K, layout_a, layout_b, layout_c, config):
        """Evaluate a configuration and return timing."""
        warps_m, warps_n, warp_tile_m, warp_tile_n, bits, use_direct_write = config

        try:
            result = subprocess.run([
                "./rocm_wmma_gemm/tuner",
                str(M), str(N), str(K),
                str(warps_m), str(warps_n),
                str(warp_tile_m), str(warp_tile_n),
                str(bits),
                str(1 if use_direct_write else 0),
                str(layout_a), str(layout_b), str(layout_c),
                self.gpu_arch
            ], capture_output=True, text=True, timeout=60, check=False)

            if result.returncode != 0:
                return float('inf')

            time_ms = self._parse_benchmark_output(result.stdout)
            if time_ms is None:
                return float('inf')

            return time_ms

        except Exception:
            return float('inf')

    def _objective_function_with_params(self, trial):
        """Optuna objective function that works with actual parameters and prunes invalid configs."""
        # Get parameters suggested by Optuna
        warps_m = trial.suggest_categorical('warps_m', self.param_space['warps_m'])
        warps_n = trial.suggest_categorical('warps_n', self.param_space['warps_n'])
        warp_tile_m = trial.suggest_categorical('warp_tile_m', self.param_space['warp_tile_m'])
        warp_tile_n = trial.suggest_categorical('warp_tile_n', self.param_space['warp_tile_n'])
        bits = trial.suggest_categorical('bits', self.param_space['bits'])
        use_direct_write = trial.suggest_categorical('use_direct_write', self.param_space['use_direct_write'])

        config = (warps_m, warps_n, warp_tile_m, warp_tile_n, bits, use_direct_write)

        # Check constraints - if invalid, PRUNE the trial (don't count it)
        if not self._check_constraints(config):
            trial.set_user_attr("invalid_combination", True)
            raise optuna.TrialPruned("Invalid parameter combination")

        # Evaluate the valid configuration
        M, N, K, layout_a, layout_b, layout_c = self.current_problem
        time_ms = self._evaluate_config(M, N, K, layout_a, layout_b, layout_c, config)

        self.total_evaluations += 1

        # Track improvements
        if time_ms < self.best_time:
            self.best_time = time_ms
            self.best_config = config
            self.improvement_history.append(self.total_evaluations)
            print(f"  New best: {config} -> {time_ms:.3f}ms (trial {self.total_evaluations})")
        elif time_ms == self.best_time and self.best_time != float('inf'):
            # Equal performance - update to newer config
            self.best_config = config
            print(f"  New best: {config} -> {time_ms:.3f}ms (equal, trial {self.total_evaluations})")
        else:
            print(f"  Trial {self.total_evaluations}: {config} -> {time_ms:.3f}ms")

        return time_ms

    def tune_layout(self, M, N, K, layout_a, layout_b, layout_c, max_evaluations=50, existing_baseline=None):
        """Tune for specific (A,B,C) layout combination using Optuna TPE."""
        print(f"\nTuning {M}×{N}×{K} with layout A={layout_a}, B={layout_b}, C={layout_c}")
        print("Using Optuna TPE (Tree-structured Parzen Estimators)")

        # Reset state for this problem
        self.current_problem = (M, N, K, layout_a, layout_b, layout_c)
        self.total_evaluations = 0
        self.best_config = None
        self.best_time = float('inf')
        self.improvement_history = []

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(
                seed=self.random_seed,
                n_startup_trials=min(20, max_evaluations // 3),
                n_ei_candidates=24,
                multivariate=True
            )
        )

        print("Starting Optuna optimization...")

        # Add baseline configurations
        baselines_to_use = self.baselines if existing_baseline is None else [existing_baseline]

        for i, baseline in enumerate(baselines_to_use):
            if self._check_constraints(baseline):
                warps_m, warps_n, warp_tile_m, warp_tile_n, bits, use_direct_write = baseline

                study.enqueue_trial({
                    'warps_m': warps_m,
                    'warps_n': warps_n,
                    'warp_tile_m': warp_tile_m,
                    'warp_tile_n': warp_tile_n,
                    'bits': bits,
                    'use_direct_write': use_direct_write
                })

        if existing_baseline is not None:
            print(f"  Using existing config as baseline: {existing_baseline}")

        # Run optimization
        study.optimize(self._objective_function_with_params, n_trials=max_evaluations, show_progress_bar=False)

        if self.best_config is None:
            print("  No valid configuration found!")
            return None

        # Calculate memory usage
        wmma_tile = 16
        block_m = self.best_config[0] * self.best_config[2] * wmma_tile
        block_n = self.best_config[1] * self.best_config[3] * wmma_tile
        block_k = wmma_tile
        lds_size = (block_m * block_k) + (block_k * block_n)
        memory_used = 2 * lds_size * 2

        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

        coverage = (self.total_evaluations / len(self.valid_configs)) * 100
        print(f"  Best config: {self.best_config} -> {self.best_time:.3f}ms")
        print(f"  Memory usage: {memory_used}/{self.max_shared_memory} bytes")
        print(f"  Improvements found: {len(self.improvement_history)}")
        print(f"  Valid evaluations: {self.total_evaluations}")
        print(f"  Completed trials: {completed_trials}, Pruned trials: {pruned_trials}")
        print(f"  Valid rate: {completed_trials/(completed_trials + pruned_trials)*100:.1f}%")
        print(f"  Space coverage: {coverage:.1f}%")

        return {
            'config': {
                'warps_m': int(self.best_config[0]),
                'warps_n': int(self.best_config[1]),
                'warp_tile_m': int(self.best_config[2]),
                'warp_tile_n': int(self.best_config[3]),
                'bits': int(self.best_config[4]),
                'use_direct_write': bool(self.best_config[5])
            },
            'time_ms': float(self.best_time),
            'evaluations': self.total_evaluations,
            'memory_used_bytes': memory_used,
            'space_coverage_percent': coverage
        }

    def tune_all(self, sizes=None, layouts=None, max_evaluations=50, existing_configs=None):
        """Tune all size and layout combinations."""
        if sizes is None:
            sizes = [
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                (8192, 8192, 8192)
            ]

        if layouts is None:
            # All 8 combinations of (A, B, C) layouts
            layouts = [
                (0, 0, 0), (0, 0, 1),
                (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1),
                (1, 1, 0), (1, 1, 1)
            ]

        results = {}

        for M, N, K in sizes:
            size_key = f"{M}x{N}x{K}"
            results[size_key] = {}

            for layout_a, layout_b, layout_c in layouts:
                layout_key = f"{layout_a}_{layout_b}_{layout_c}"

                # Check if we have an existing config for this size/layout
                existing_baseline = None
                if existing_configs:
                    existing_baseline = existing_configs.get(size_key, {}).get(layout_key)

                result = self.tune_layout(M, N, K, layout_a, layout_b, layout_c, max_evaluations, existing_baseline)

                if result:
                    results[size_key][layout_key] = {
                        "M": M, "N": N, "K": K,
                        "layout": {
                            "A": "row_major" if layout_a == 0 else "col_major",
                            "B": "row_major" if layout_b == 0 else "col_major",
                            "C": "row_major" if layout_c == 0 else "col_major"
                        },
                        "config": result['config'],
                        "avg_time_ms": result['time_ms'],
                        "evaluations": result['evaluations'],
                        "memory_used_bytes": result['memory_used_bytes'],
                        "space_coverage_percent": result['space_coverage_percent']
                    }

        return results

def load_existing_json(input_file):
    """Load existing JSON configuration file and extract configs by size and layout."""
    if not Path(input_file).exists():
        print(f"Warning: Input file '{input_file}' not found. Starting fresh.")
        return None

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Convert to internal format: size_key -> layout_key -> baseline tuple
        existing_configs = {}

        for config_entry in data.get('configurations', []):
            M = config_entry['range']['M']
            N = config_entry['range']['N']
            K = config_entry['range']['K']
            size_key = f"{M}x{N}x{K}"

            layout = config_entry['layout']
            layout_a = 0 if layout['A'] == 'row_major' else 1
            layout_b = 0 if layout['B'] == 'row_major' else 1
            layout_c = 0 if layout['C'] == 'row_major' else 1
            layout_key = f"{layout_a}_{layout_b}_{layout_c}"

            cfg = config_entry['config']
            baseline = (
                cfg['warps_m'],
                cfg['warps_n'],
                cfg['warp_tile_m'],
                cfg['warp_tile_n'],
                cfg['bits'],
                cfg['use_direct_write']
            )

            if size_key not in existing_configs:
                existing_configs[size_key] = {}
            existing_configs[size_key][layout_key] = baseline

        print(f"Loaded {len(data.get('configurations', []))} existing configurations from '{input_file}'")
        return existing_configs

    except Exception as e:
        print(f"Error loading input file '{input_file}': {e}")
        print("Starting fresh.")
        return None

def merge_results(existing_configs_raw, new_results):
    """Merge new results with existing configurations, preferring new results."""
    if existing_configs_raw is None:
        return new_results

    # Convert existing raw configs back to result format
    merged = {}

    # First, add all existing configs
    for size_key, layout_dict in existing_configs_raw.items():
        merged[size_key] = {}
        for layout_key, baseline in layout_dict.items():
            # Parse size
            M, N, K = map(int, size_key.split('x'))
            # Parse layout
            layout_a, layout_b, layout_c = map(int, layout_key.split('_'))

            merged[size_key][layout_key] = {
                "M": M, "N": N, "K": K,
                "layout": {
                    "A": "row_major" if layout_a == 0 else "col_major",
                    "B": "row_major" if layout_b == 0 else "col_major",
                    "C": "row_major" if layout_c == 0 else "col_major"
                },
                "config": {
                    'warps_m': int(baseline[0]),
                    'warps_n': int(baseline[1]),
                    'warp_tile_m': int(baseline[2]),
                    'warp_tile_n': int(baseline[3]),
                    'bits': int(baseline[4]),
                    'use_direct_write': bool(baseline[5])
                },
                "avg_time_ms": None,  # Unknown from existing
                "evaluations": 0,
                "memory_used_bytes": None,
                "space_coverage_percent": 0
            }

    # Now overwrite with new results
    for size_key, layout_dict in new_results.items():
        if size_key not in merged:
            merged[size_key] = {}
        for layout_key, result in layout_dict.items():
            merged[size_key][layout_key] = result

    return merged

def parse_matrix_sizes(size_strings):
    """Parse matrix size strings like '1024,1024,1024' into tuples."""
    sizes = []
    for size_str in size_strings:
        try:
            parts = size_str.split(',')
            if len(parts) != 3:
                raise ValueError(f"Invalid size format: {size_str}. Expected M,N,K")
            M, N, K = map(int, parts)
            sizes.append((M, N, K))
        except ValueError as e:
            print(f"Error parsing size '{size_str}': {e}")
            sys.exit(1)
    return sizes

def parse_layouts(layout_strings):
    """Parse layout strings like 'row_major,col_major,row_major' into tuples (A,B,C)."""
    layouts = []
    layout_map = {'row_major': 0, 'col_major': 1, 'r': 0, 'c': 1}

    for layout_str in layout_strings:
        try:
            parts = layout_str.split(',')
            if len(parts) != 3:
                raise ValueError(f"Invalid layout format: {layout_str}. Expected A,B,C")

            layout_tuple = []
            for part in parts:
                part = part.strip().lower()
                if part not in layout_map:
                    raise ValueError(f"Invalid layout '{part}'. Use 'row_major'/'r' or 'col_major'/'c'")
                layout_tuple.append(layout_map[part])

            layouts.append(tuple(layout_tuple))
        except ValueError as e:
            print(f"Error parsing layout '{layout_str}': {e}")
            sys.exit(1)

    return layouts

def parse_baselines(baseline_strings):
    """Parse baseline strings like '4,4,4,4,256,0' into tuples."""
    baselines = []
    for baseline_str in baseline_strings:
        try:
            parts = baseline_str.split(',')
            if len(parts) != 6:
                raise ValueError(f"Invalid baseline format: {baseline_str}. Expected warps_m,warps_n,warp_tile_m,warp_tile_n,bits,use_direct_write")
            warps_m, warps_n, warp_tile_m, warp_tile_n, bits = map(int, parts[:5])
            use_direct_write = bool(int(parts[5]))
            baselines.append((warps_m, warps_n, warp_tile_m, warp_tile_n, bits, use_direct_write))
        except ValueError as e:
            print(f"Error parsing baseline '{baseline_str}': {e}")
            sys.exit(1)
    return baselines

def main():
    parser = argparse.ArgumentParser(
        description='Optuna TPE WMMA Tuner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run with Optuna TPE
  python tune.py

  # Load existing config and re-tune specific layouts
  python tune.py --input gemm_config.json --layouts r,r,r c,c,c

  # Load existing config, re-tune specific size, use existing as baseline
  python tune.py --input gemm_config.json --sizes 4096,4096,4096

  # Load and reduce budget (faster re-tuning)
  python tune.py --input gemm_config.json --budget 50

  # Specific seed for reproducibility
  python tune.py --seed 123

  # Larger budget for thorough search
  python tune.py --budget 150

  # Test specific sizes
  python tune.py --sizes 1024,1024,1024 2048,2048,2048

  # Custom baselines (overrides input file baselines)
  python tune.py --baselines 4,4,4,4,256,0 2,2,4,4,128,1 8,2,2,2,64,0

  # Specific layouts (A,B,C)
  python tune.py --layouts row_major,col_major,row_major r,r,c

  # Different GPU architecture
  python tune.py --gpu-arch gfx1103
        """)

    parser.add_argument('--input', '-i',
                       help='Input JSON file to load existing configurations (will be used as baselines)')
    parser.add_argument('--sizes', nargs='*',
                       help='Matrix sizes as M,N,K (e.g., 1024,1024,1024 2048,2048,2048)')
    parser.add_argument('--layouts', nargs='*',
                       help='Matrix (A,B,C) layouts as A,B,C (e.g., row_major,col_major,row_major or r,c,r)')
    parser.add_argument('--baselines', nargs='*',
                       help='Baseline configs as warps_m,warps_n,warp_tile_m,warp_tile_n,bits,use_direct_write (overrides input file)')
    parser.add_argument('--budget', type=int, default=150,
                       help='Evaluation budget per layout combination (default: 150)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--gpu-arch', default='gfx1100', help='GPU architecture (default: gfx1100)')
    parser.add_argument('--max-memory', type=int, default=65336,
                       help='Maximum shared memory in bytes (default: 65336)')
    parser.add_argument('--output', default='gemm_config_tuned.json',
                       help='Output JSON file (default: gemm_config_tuned.json)')

    args = parser.parse_args()

    # Load existing configurations if provided
    existing_configs_raw = None
    if args.input:
        existing_configs_raw = load_existing_json(args.input)

    # Parse inputs
    if args.sizes:
        sizes = parse_matrix_sizes(args.sizes)
    else:
        sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (4096, 4096, 1024),
            (8192, 8192, 1024),
            (4096, 2048, 64),
            (8192, 4096, 128),
            (4096, 16384, 4096),
            (4096, 4096, 16384),
            (2048, 5120, 5120),
            (4096, 5120, 5120),
            (32768, 4096, 4096),
            (65536, 2048, 2048),
        ]

    if args.layouts:
        layouts = parse_layouts(args.layouts)
    else:
        # All 8 combinations of (A, B, C) layouts
        layouts = [
            (0, 0, 0), (0, 0, 1),
            (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1),
            (1, 1, 0), (1, 1, 1)
        ]

    # Determine baselines: explicit > None (will use existing configs per-problem) > default
    if args.baselines:
        baselines = parse_baselines(args.baselines)
        print("Using explicit baselines (overriding input file)")
    elif args.input and existing_configs_raw:
        baselines = None  # Will use existing configs as baselines per-problem
        print("Using existing configurations as baselines")
    else:
        baselines = None  # Will use default baselines
        print("Using default baselines")

    print("Optuna TPE WMMA Tuner (6-parameter version)")
    print(f"Parameters: warps_m, warps_n, warp_tile_m, warp_tile_n, bits, use_direct_write")
    print(f"Random seed: {args.seed}")
    print(f"GPU Architecture: {args.gpu_arch}")
    print(f"Evaluation budget per layout: {args.budget}")
    print(f"Shared memory limit: {args.max_memory} bytes")
    print(f"Matrix sizes to test: {len(sizes)}")
    for size in sizes:
        print(f"  {size[0]}×{size[1]}×{size[2]}")
    print(f"Layout combinations to test: {len(layouts)}")
    for i, layout in enumerate(layouts):
        layout_names = ["row_major" if x == 0 else "col_major" for x in layout]
        print(f"  {i+1}: A={layout_names[0]}, B={layout_names[1]}, C={layout_names[2]}")

    if args.baselines:
        print(f"Custom baselines: {len(baselines)}")
        for baseline in baselines:
            print(f"  {baseline}")

    tuner = OptunaWMMATuner(
        max_shared_memory=args.max_memory,
        gpu_arch=args.gpu_arch,
        baselines=baselines,
        random_seed=args.seed
    )

    results = tuner.tune_all(sizes=sizes, layouts=layouts, max_evaluations=args.budget,
                            existing_configs=existing_configs_raw)

    # Merge with existing configurations
    merged_results = merge_results(existing_configs_raw, results)

    # Generate configuration JSON
    configs = []
    for size_results in merged_results.values():
        for result in size_results.values():
            config = {
                "range": {"M": result["M"], "N": result["N"], "K": result["K"]},
                "layout": result["layout"],
                "config": result["config"]
            }
            configs.append(config)

    config_data = {"configurations": configs}

    # Save results
    with open(args.output, "w") as f:
        json.dump(config_data, f, indent=4)

    # Print summary
    print("\n" + "="*80)
    print("OPTUNA TPE OPTIMIZATION WMMA RESULTS:")
    print("="*80)

    total_evaluations = 0
    total_coverage = 0
    count = 0

    for size_key, size_results in results.items():
        print(f"\n{size_key}:")
        for layout_key, result in size_results.items():
            config = result['config']
            coverage = result.get('space_coverage_percent', 0)
            print(f"  {layout_key}: {config['warps_m']},{config['warps_n']},"
                  f"{config['warp_tile_m']},{config['warp_tile_n']},"
                  f"{config['bits']},{int(config['use_direct_write'])} -> "
                  f"{result['avg_time_ms']:.3f}ms ({result['evaluations']} evals, {coverage:.1f}% coverage)")
            total_evaluations += result['evaluations']
            total_coverage += coverage
            count += 1

    if count > 0:
        avg_evals = total_evaluations / count
        avg_coverage = total_coverage / count
        print(f"\nTotal evaluations: {total_evaluations}")
        print(f"Average evaluations per problem: {avg_evals:.1f}")
        print(f"Average space coverage: {avg_coverage:.1f}%")

    # Show what was preserved vs updated
    if existing_configs_raw:
        total_configs = sum(len(layout_dict) for layout_dict in merged_results.values())
        updated_configs = sum(len(layout_dict) for layout_dict in results.values())
        preserved_configs = total_configs - updated_configs
        print(f"\nConfigurations preserved from input: {preserved_configs}")
        print(f"Configurations updated/added: {updated_configs}")
        print(f"Total configurations in output: {total_configs}")

    print(f"\nConfiguration saved to: {args.output}")

if __name__ == "__main__":
    main()
