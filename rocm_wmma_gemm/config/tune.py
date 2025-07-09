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
            self.baselines = [
                (4, 4, 4, 4),  # Original baseline
                (2, 4, 4, 4),  # Fewer warps_m
                (4, 2, 4, 4),  # Fewer warps_n
                (2, 2, 4, 4),  # Smaller warps, larger tiles
                (4, 2, 2, 4),  # Asymmetric warps
                (2, 4, 4, 2),  # Asymmetric warps (flipped)
                (4, 4, 4, 2),  # Original warps, smaller tiles
                (4, 4, 2, 2),  # Original warps, smaller tiles
                (2, 8, 4, 4),  # Larger tile
            ]
        else:
            self.baselines = baselines

        # Parameter space definitions - 4 parameters only
        self.param_space = {
            'warps_m': [1, 2, 4, 8],
            'warps_n': [1, 2, 4, 8],
            'warp_tile_m': [1, 2, 4, 8],
            'warp_tile_n': [1, 2, 4, 8]
        }

        # Generate all valid configurations for sampling
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
                        count += 1
                        config = (warps_m, warps_n, warp_tile_m, warp_tile_n)
                        if self._check_constraints(config):
                            configs.append(config)

        print(f"Checked {count} total combinations, {len(configs)} valid after constraints")
        return configs

    def _check_constraints(self, config):
        """Check memory and resource constraints for WMMA."""
        warps_m, warps_n, warp_tile_m, warp_tile_n = config

        if warps_m == 0 or warps_n == 0 or warp_tile_m == 0 or warp_tile_n == 0:
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
        """Parse benchmark output to extract timing."""
        lines = output.strip().split('\n')
        for line in lines:
            if 'dynamic_kernel/manual_time' in line:
                match = re.search(r'(\d+\.?\d*)\s*ms', line)
                if match:
                    return float(match.group(1))
        return None

    def _evaluate_config(self, M, N, K, layout_a, layout_b, layout_c, config):
        """Evaluate a configuration and return timing."""
        warps_m, warps_n, warp_tile_m, warp_tile_n = config

        try:
            result = subprocess.run([
                "./rocm_wmma_gemm/tuner",
                str(M), str(N), str(K),
                str(warps_m), str(warps_n),
                str(warp_tile_m), str(warp_tile_n),
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

    def _objective_function(self, trial):
        """Optuna objective function - samples from pre-filtered valid configurations."""
        # Sample directly from valid configurations
        config_idx = trial.suggest_int('config_idx', 0, len(self.valid_configs) - 1)
        config = self.valid_configs[config_idx]

        # All configs are guaranteed valid - no constraint checking needed
        M, N, K, layout_a, layout_b = self.current_problem
        time_ms = self._evaluate_config(M, N, K, layout_a, layout_b, 0, config)

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

    def _objective_function_for_baseline(self, config):
        """Evaluate baseline configuration."""
        M, N, K, layout_a, layout_b = self.current_problem
        time_ms = self._evaluate_config(M, N, K, layout_a, layout_b, 0, config)

        self.total_evaluations += 1

        if time_ms < self.best_time:
            self.best_time = time_ms
            self.best_config = config
            self.improvement_history.append(self.total_evaluations)
            print(f"  Baseline result: {config} -> {time_ms:.3f}ms")
        elif time_ms == self.best_time and self.best_time != float('inf'):
            # Equal performance - update to newer config
            self.best_config = config
            print(f"  Baseline result: {config} -> {time_ms:.3f}ms (equal)")
        else:
            print(f"  Baseline result: {config} -> {time_ms:.3f}ms")

        return time_ms

    def tune_ab_layout(self, M, N, K, layout_a, layout_b, max_evaluations=50):
        """Tune for (A,B) layout pair using Optuna TPE."""
        print(f"\nTuning {M}×{N}×{K} with layout A={layout_a}, B={layout_b}")
        print("Using Optuna TPE (Tree-structured Parzen Estimators)")

        # Reset state for this problem
        self.current_problem = (M, N, K, layout_a, layout_b)
        self.total_evaluations = 0
        self.best_config = None
        self.best_time = float('inf')
        self.improvement_history = []

        # Create Optuna study - adjusted parameters for smaller space
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(
                seed=self.random_seed,
                n_startup_trials=min(8, max_evaluations // 4),  # Fewer startup trials for smaller space
                n_ei_candidates=16,  # Smaller candidate set
                multivariate=True
            )
        )

        print("Starting Optuna optimization...")

        # Add baseline configurations first
        baseline_budget = min(len(self.baselines), max_evaluations // 3)
        for i, baseline in enumerate(self.baselines[:baseline_budget]):
            if baseline in self.valid_configs:
                # Find index of baseline in valid configs
                baseline_idx = self.valid_configs.index(baseline)

                # Enqueue trial with the config index (Optuna will evaluate it)
                study.enqueue_trial({'config_idx': baseline_idx})

        # Run remaining optimization trials
        remaining_trials = max_evaluations - self.total_evaluations
        if remaining_trials > 0:
            study.optimize(self._objective_function, n_trials=remaining_trials, show_progress_bar=False)

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

        coverage = (self.total_evaluations / len(self.valid_configs)) * 100
        print(f"  Best config: {self.best_config} -> {self.best_time:.3f}ms")
        print(f"  Memory usage: {memory_used}/{self.max_shared_memory} bytes")
        print(f"  Improvements found: {len(self.improvement_history)}")
        print(f"  Total evaluations: {self.total_evaluations}/{max_evaluations}")
        print(f"  Space coverage: {coverage:.1f}%")

        return {
            'config': {
                'warps_m': int(self.best_config[0]),
                'warps_n': int(self.best_config[1]),
                'warp_tile_m': int(self.best_config[2]),
                'warp_tile_n': int(self.best_config[3])
            },
            'time_ms': float(self.best_time),
            'evaluations': self.total_evaluations,
            'memory_used_bytes': memory_used,
            'space_coverage_percent': coverage
        }

    def tune_all(self, sizes=None, ab_layouts=None, max_evaluations=50):
        """Tune all size and (A,B) layout combinations."""
        if sizes is None:
            sizes = [
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                (8192, 8192, 8192)
            ]

        if ab_layouts is None:
            ab_layouts = [(0, 0), (0, 1), (1, 0), (1, 1)]

        results = {}

        for M, N, K in sizes:
            size_key = f"{M}x{N}x{K}"
            results[size_key] = {}

            for layout_a, layout_b in ab_layouts:
                layout_key = f"{layout_a}_{layout_b}"

                result = self.tune_ab_layout(M, N, K, layout_a, layout_b, max_evaluations)

                if result:
                    # Create entries for both C layouts using the same optimized config
                    for layout_c in [0, 1]:
                        full_layout_key = f"{layout_a}_{layout_b}_{layout_c}"

                        results[size_key][full_layout_key] = {
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
    """Parse layout strings like 'row_major,col_major' into tuples (A,B only)."""
    layouts = []
    layout_map = {'row_major': 0, 'col_major': 1, 'r': 0, 'c': 1}

    for layout_str in layout_strings:
        try:
            parts = layout_str.split(',')
            if len(parts) != 2:
                raise ValueError(f"Invalid layout format: {layout_str}. Expected A,B")

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
    """Parse baseline strings like '4,4,4,4' into tuples."""
    baselines = []
    for baseline_str in baseline_strings:
        try:
            parts = baseline_str.split(',')
            if len(parts) != 4:
                raise ValueError(f"Invalid baseline format: {baseline_str}. Expected warps_m,warps_n,warp_tile_m,warp_tile_n")
            warps_m, warps_n, warp_tile_m, warp_tile_n = map(int, parts)
            baselines.append((warps_m, warps_n, warp_tile_m, warp_tile_n))
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

  # Specific seed for reproducibility
  python tune.py --seed 123

  # Larger budget for thorough search
  python tune.py --budget 60

  # Test specific sizes
  python tune.py --sizes 1024,1024,1024 2048,2048,2048

  # Custom baselines
  python tune.py --baselines 4,4,4,4 2,2,4,4 8,2,2,2

  # Different GPU architecture
  python tune.py --gpu-arch gfx1103
        """)

    parser.add_argument('--sizes', nargs='*',
                       help='Matrix sizes as M,N,K (e.g., 1024,1024,1024 2048,2048,2048)')
    parser.add_argument('--layouts', nargs='*',
                       help='Matrix (A,B) layouts as A,B (e.g., row_major,col_major or r,c)')
    parser.add_argument('--baselines', nargs='*',
                       help='Baseline configs as warps_m,warps_n,warp_tile_m,warp_tile_n (e.g., 4,4,4,4 2,2,4,4)')
    parser.add_argument('--budget', type=int, default=50,
                       help='Evaluation budget per (A,B) layout combination (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--gpu-arch', default='gfx1100', help='GPU architecture (default: gfx1100)')
    parser.add_argument('--max-memory', type=int, default=65336,
                       help='Maximum shared memory in bytes (default: 65336)')
    parser.add_argument('--output', default='gemm_config_tuned.json',
                       help='Output JSON file (default: gemm_config_tuned.json)')

    args = parser.parse_args()

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
        ab_layouts = parse_layouts(args.layouts)
    else:
        ab_layouts = [(0, 0), (0, 1), (1, 0), (1, 1)]

    if args.baselines:
        baselines = parse_baselines(args.baselines)
    else:
        baselines = None

    print("Optuna TPE WMMA Tuner (4-parameter version)")
    print(f"Random seed: {args.seed}")
    print(f"GPU Architecture: {args.gpu_arch}")
    print(f"Evaluation budget per (A,B) layout: {args.budget}")
    print(f"Shared memory limit: {args.max_memory} bytes")
    print(f"Matrix sizes to test: {len(sizes)}")
    for size in sizes:
        print(f"  {size[0]}×{size[1]}×{size[2]}")
    print(f"(A,B) layout combinations to test: {len(ab_layouts)}")
    for i, layout in enumerate(ab_layouts):
        layout_names = ["row_major" if x == 0 else "col_major" for x in layout]
        print(f"  {i+1}: A={layout_names[0]}, B={layout_names[1]}")

    if baselines:
        print(f"Custom baselines: {len(baselines)}")
        for baseline in baselines:
            print(f"  {baseline}")

    tuner = OptunaWMMATuner(
        max_shared_memory=args.max_memory,
        gpu_arch=args.gpu_arch,
        baselines=baselines,
        random_seed=args.seed
    )

    results = tuner.tune_all(sizes=sizes, ab_layouts=ab_layouts, max_evaluations=args.budget)

    # Generate configuration JSON
    configs = []
    processed_configs = set()

    for size_results in results.values():
        for result in size_results.values():
            layout_a = result["layout"]["A"]
            layout_b = result["layout"]["B"]
            config_key = (result["M"], result["N"], result["K"], layout_a, layout_b)

            if config_key not in processed_configs:
                processed_configs.add(config_key)
                config = {
                    "range": {"M": result["M"], "N": result["N"], "K": result["K"]},
                    "layout": {"A": layout_a, "B": layout_b},
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
        # Only show one result per (A,B) since both C layouts use same config
        ab_results = {}
        for layout_key, result in size_results.items():
            ab_key = "_".join(layout_key.split("_")[:2])  # A_B only
            if ab_key not in ab_results:
                ab_results[ab_key] = result

        for ab_key, result in ab_results.items():
            config = result['config']
            coverage = result.get('space_coverage_percent', 0)
            print(f"  {ab_key}: {config['warps_m']},{config['warps_n']},"
                  f"{config['warp_tile_m']},{config['warp_tile_n']} -> "
                  f"{result['avg_time_ms']:.3f}ms ({result['evaluations']} evals, {coverage:.1f}% coverage)")
            total_evaluations += result['evaluations']
            total_coverage += coverage
            count += 1

    if count > 0:
        avg_evals = total_evaluations / count
        avg_coverage = total_coverage / count
        print(f"\nTotal evaluations: {total_evaluations}")
        print(f"Average evaluations per (A,B) problem: {avg_evals:.1f}")
        print(f"Average space coverage: {avg_coverage:.1f}%")
    print(f"Configuration saved to: {args.output}")

if __name__ == "__main__":
    main()
