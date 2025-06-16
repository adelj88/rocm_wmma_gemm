#!/usr/bin/env python3

import subprocess
import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
import warnings
import re
import argparse
import sys

warnings.filterwarnings('ignore')

class UCBBayesianTuner:
    """Clean UCB-based Bayesian Optimization for GEMM kernel tuning."""

    def __init__(self, max_shared_memory=65336, gpu_arch="gfx1100", baselines=None):
        self.max_shared_memory = max_shared_memory
        self.gpu_arch = gpu_arch

        # Define baseline configurations to always try first
        if baselines is None:
            self.baselines = [
                (4, 4, 4, 4),  # Original baseline
                (2, 4, 4, 4),  # Fewer warps_m
                (2, 2, 4, 4),  # Smaller warps, larger tiles
                (4, 2, 2, 4),  # Asymmetric warps
                (2, 4, 4, 2),  # Asymmetric warps (flipped)
                (4, 4, 2, 2),  # Original warps, smaller tiles
                (2, 2, 2, 2),  # Conservative baseline
            ]
        else:
            self.baselines = baselines

        # Parameter space
        self.valid_values = {
            'warps_m': [1, 2, 4, 8],
            'warps_n': [1, 2, 4, 8],
            'warp_tile_m': [1, 2, 4, 8],
            'warp_tile_n': [1, 2, 4, 8]
        }

        # Generate all valid configurations
        self.valid_configs = self._generate_valid_configs()
        print(f"Generated {len(self.valid_configs)} valid configurations")

        # Gaussian Process setup
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=8,
            alpha=1e-6,
            normalize_y=True
        )
        self.scaler = StandardScaler()

        # UCB parameters
        self.initial_beta = 3.0  # Higher exploration at start
        self.final_beta = 0.1    # Low exploration at end

        # State tracking
        self.X_observed = []
        self.y_observed = []
        self.total_evaluations = 0
        self.improvement_history = []
        self.best_found_time = float('inf')
        self.stagnation_counter = 0

    def _generate_valid_configs(self):
        """Generate all valid discrete configurations."""
        configs = []
        for warps_m in self.valid_values['warps_m']:
            for warps_n in self.valid_values['warps_n']:
                for warp_tile_m in self.valid_values['warp_tile_m']:
                    for warp_tile_n in self.valid_values['warp_tile_n']:
                        config = (warps_m, warps_n, warp_tile_m, warp_tile_n)
                        if self._check_constraints(config):
                            configs.append(config)
        return configs

    def _check_constraints(self, config):
        """Check memory and resource constraints."""
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

        # Resource constraint
        total_warps = warps_m * warps_n
        if total_warps > 16:
            return False

        return True

    def _config_to_features(self, config):
        """Convert config tuple to feature vector for GP."""
        return np.array(config, dtype=np.float64)

    def _latin_hypercube_sampling(self, untried_configs, n_samples):
        """Generate Latin Hypercube samples from untried configurations."""
        if len(untried_configs) <= n_samples:
            return untried_configs

        # Convert configs to feature matrix
        feature_matrix = np.array([self._config_to_features(config) for config in untried_configs])

        # For each dimension, divide into n_samples strata and sample one from each
        n_dims = feature_matrix.shape[1]
        selected_indices = []

        for dim in range(n_dims):
            # Get unique values in this dimension and sort them
            unique_values = np.unique(feature_matrix[:, dim])

            # If we have fewer unique values than samples, just take all
            if len(unique_values) <= n_samples:
                continue

            # Divide into strata
            strata_size = len(unique_values) // n_samples
            strata_indices = []

            for i in range(n_samples):
                start_idx = i * strata_size
                end_idx = (i + 1) * strata_size if i < n_samples - 1 else len(unique_values)

                # Randomly select one value from this stratum
                selected_value = unique_values[np.random.randint(start_idx, end_idx)]

                # Find configs with this value in this dimension
                valid_indices = np.where(feature_matrix[:, dim] == selected_value)[0]
                strata_indices.extend(valid_indices)

            selected_indices = strata_indices if not selected_indices else list(set(selected_indices) & set(strata_indices))

        # If stratification gives us too few samples, fall back to random
        if len(selected_indices) < n_samples:
            selected_indices = np.random.choice(len(untried_configs), size=min(n_samples, len(untried_configs)), replace=False)
        else:
            # Randomly sample from stratified candidates
            selected_indices = np.random.choice(selected_indices, size=min(n_samples, len(selected_indices)), replace=False)

        return [untried_configs[i] for i in selected_indices]

    def _get_beta(self, evaluation_num, max_evaluations):
        """Calculate UCB beta parameter with budget-aware adaptive scheduling."""
        progress = evaluation_num / max_evaluations

        # Adjust beta schedule based on budget size
        if max_evaluations <= 30:
            # Small budget: faster decay to exploitation
            decay_rate = 1.5
        elif max_evaluations <= 60:
            # Medium budget: balanced decay
            decay_rate = 1.0
        else:
            # Large budget: slower decay for more exploration
            decay_rate = 0.7

        # Adaptive based on improvement rate
        if len(self.improvement_history) >= 3:
            recent_improvements = len([x for x in self.improvement_history
                                     if x >= evaluation_num - min(10, max_evaluations // 4)])
            if recent_improvements >= 2:
                # Recent improvements -> keep exploring
                beta = max(self.final_beta, self.initial_beta - decay_rate * 0.5 * progress)
            else:
                # No recent improvements -> exploit more
                beta = self.final_beta + (self.initial_beta - self.final_beta) * (1 - progress * decay_rate) ** 2
        else:
            # Not enough data -> use linear decay with budget-aware rate
            beta = self.initial_beta - (self.initial_beta - self.final_beta) * progress * decay_rate

        return max(self.final_beta, beta)
        """Calculate UCB beta parameter with adaptive scheduling."""
        progress = evaluation_num / max_evaluations

        # Adaptive based on improvement rate
        if len(self.improvement_history) >= 3:
            recent_improvements = len([x for x in self.improvement_history
                                     if x >= evaluation_num - 10])
            if recent_improvements >= 2:
                # Recent improvements -> keep exploring
                beta = max(self.final_beta, self.initial_beta - 0.5 * progress)
            else:
                # No recent improvements -> exploit more
                beta = self.final_beta + (self.initial_beta - self.final_beta) * (1 - progress) ** 2
        else:
            # Not enough data -> use linear decay
            beta = self.initial_beta - (self.initial_beta - self.final_beta) * progress

        return max(self.final_beta, beta)

    def _ucb_acquisition(self, configs, beta):
        """Upper Confidence Bound acquisition function for minimization."""
        if len(self.X_observed) == 0:
            return np.random.random(len(configs))

        # Convert configs to features and scale
        X_candidates = np.array([self._config_to_features(config) for config in configs])
        X_scaled = self.scaler.transform(X_candidates)

        # Get GP predictions
        mu, sigma = self.gp.predict(X_scaled, return_std=True)

        # UCB for minimization: μ - β * σ (lower is better)
        ucb_values = mu - beta * sigma

        # Convert to acquisition scores (higher = better candidate)
        acquisition_scores = -ucb_values

        return acquisition_scores

    def _calculate_meaningful_improvement_threshold(self):
        """Calculate threshold for meaningful improvements from historical data."""
        if len(self.y_observed) < 3:
            return 0.02  # Default 2%

        improvements = []
        for i in range(1, len(self.y_observed)):
            current_best = min(self.y_observed[:i])
            if self.y_observed[i] < current_best:
                rel_improvement = (current_best - self.y_observed[i]) / current_best
                improvements.append(rel_improvement)

        if improvements:
            return np.median(improvements)
        else:
            return 0.02

    def _should_stop_early(self, configs_tried, max_evaluations):
        """Simple principled stopping: only when all configs tried."""
        untried_configs = [c for c in self.valid_configs if c not in configs_tried]
        if not untried_configs:
            print("  → All configurations evaluated")
            return True
        return False  # Otherwise, run to budget completion

    def _select_next_config(self, configs_tried, max_evaluations):
        """Select next configuration using UCB acquisition function."""
        untried_configs = [c for c in self.valid_configs if c not in configs_tried]
        if not untried_configs:
            return None

        # Try baselines first (only at the beginning)
        if len(configs_tried) < len(self.baselines):
            for baseline in self.baselines:
                if baseline in untried_configs:
                    return baseline

        # If not enough data for GP, use Latin Hypercube Sampling
        if len(self.X_observed) < 3:
            # Try to get 5-8 good initial samples using LHS
            n_lhs_samples = min(8, len(untried_configs), max_evaluations - len(configs_tried))
            if n_lhs_samples > 0:
                lhs_candidates = self._latin_hypercube_sampling(untried_configs, n_lhs_samples)
                return lhs_candidates[0]  # Return first LHS candidate
            else:
                return untried_configs[np.random.randint(len(untried_configs))]

        # Use UCB acquisition function
        current_beta = self._get_beta(self.total_evaluations, max_evaluations)
        acquisition_scores = self._ucb_acquisition(untried_configs, current_beta)

        # Select config with highest acquisition score
        best_idx = np.argmax(acquisition_scores)
        return untried_configs[best_idx]

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

    def tune_ab_layout(self, M, N, K, layout_a, layout_b, max_evaluations=40):
        """Tune for (A,B) layout pair using UCB-based BO."""
        print(f"\nTuning {M}×{N}×{K} with layout A={layout_a}, B={layout_b}")

        # Reset state for this problem
        self.X_observed = []
        self.y_observed = []
        self.total_evaluations = 0
        self.improvement_history = []
        self.best_found_time = float('inf')
        self.stagnation_counter = 0

        configs_tried = set()
        best_config = None
        best_time = float('inf')

        print("Starting UCB-based Bayesian Optimization...")
        print(f"Beta schedule: initial={self.initial_beta}, final={self.final_beta}")

        while self.total_evaluations < max_evaluations:
            # Select next config using UCB
            config = self._select_next_config(configs_tried, max_evaluations)

            if config is None:
                print("  No more valid configs to try")
                break

            configs_tried.add(config)

            # Evaluate with C=0 (row_major output)
            time_ms = self._evaluate_config(M, N, K, layout_a, layout_b, 0, config)

            self.total_evaluations += 1

            if time_ms == float('inf'):
                continue

            # Update GP training data
            self.X_observed.append(self._config_to_features(config))
            self.y_observed.append(time_ms)

            # Fit GP model (need at least 3 points)
            if len(self.X_observed) >= 3:
                try:
                    X_array = np.array(self.X_observed)
                    self.scaler.fit(X_array)
                    X_scaled = self.scaler.transform(X_array)
                    self.gp.fit(X_scaled, np.array(self.y_observed))
                except Exception as e:
                    print(f"    GP fitting failed: {e}")

            # Track best config and improvements
            if time_ms < best_time:
                best_time = time_ms
                best_config = config
                self.improvement_history.append(self.total_evaluations)
                self.stagnation_counter = 0

                current_beta = self._get_beta(self.total_evaluations, max_evaluations)
                print(f"  New best: {config} -> {time_ms:.3f}ms (β={current_beta:.3f})")
            elif time_ms == best_time:
                # Exactly equal - update to the newer config
                best_config = config
                print(f"  New best: {config} -> {time_ms:.3f}ms (equal)")
            else:
                self.stagnation_counter += 1
                current_beta = self._get_beta(self.total_evaluations, max_evaluations)
                print(f"  Config: {config} -> {time_ms:.3f}ms (β={current_beta:.3f})")

            # Update best found time for stagnation tracking
            if time_ms < self.best_found_time:
                self.best_found_time = time_ms

            # Check for early stopping
            if self._should_stop_early(configs_tried, max_evaluations):
                break

        if best_config is None:
            print("  No valid configuration found!")
            return None

        # Calculate memory usage
        wmma_tile = 16
        block_m = best_config[0] * best_config[2] * wmma_tile
        block_n = best_config[1] * best_config[3] * wmma_tile
        block_k = wmma_tile
        lds_size = (block_m * block_k) + (block_k * block_n)
        memory_used = 2 * lds_size * 2

        efficiency = (self.total_evaluations / max_evaluations) * 100
        print(f"  Best config: {best_config} -> {best_time:.3f}ms")
        print(f"  Memory usage: {memory_used}/{self.max_shared_memory} bytes")
        print(f"  Improvements found: {len(self.improvement_history)}")
        print(f"  Total evaluations: {self.total_evaluations}/{max_evaluations} ({efficiency:.1f}% of budget)")

        return {
            'config': {
                'warps_m': int(best_config[0]),
                'warps_n': int(best_config[1]),
                'warp_tile_m': int(best_config[2]),
                'warp_tile_n': int(best_config[3])
            },
            'time_ms': float(best_time),
            'evaluations': self.total_evaluations,
            'memory_used_bytes': memory_used
        }

    def tune_all(self, sizes=None, ab_layouts=None, max_evaluations=40):
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
                            "memory_used_bytes": result['memory_used_bytes']
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
        description='UCB-based Bayesian Optimization Tuner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default sizes and layouts
  python tune.py

  # Tune specific matrix sizes
  python tune.py --sizes 1024,1024,1024 2048,2048,2048

  # Quick run: 25 evaluations
  python tune.py --budget 25

  # Tune specific (A,B) layouts
  python tune.py --layouts row_major,col_major col_major,col_major

  # Use shorthand for layouts
  python tune.py --layouts r,c c,c

  # Add custom baselines
  python tune.py --baselines 4,4,4,4 2,2,4,4 8,2,2,2

  # Custom GPU architecture
  python tune.py --gpu-arch gfx1103
        """)

    parser.add_argument('--sizes', nargs='*',
                       help='Matrix sizes as M,N,K (e.g., 1024,1024,1024 2048,2048,2048)')
    parser.add_argument('--layouts', nargs='*',
                       help='Matrix (A,B) layouts as A,B (e.g., row_major,col_major or r,c)')
    parser.add_argument('--baselines', nargs='*',
                       help='Baseline configs as warps_m,warps_n,warp_tile_m,warp_tile_n (e.g., 4,4,4,4 2,2,4,4)')
    parser.add_argument('--budget', type=int, default=40,
                       help='Evaluation budget per (A,B) layout combination (default: 40)')
    parser.add_argument('--gpu-arch', default='gfx1100', help='GPU architecture (default: gfx1100)')
    parser.add_argument('--max-memory', type=int, default=65336,
                       help='Maximum shared memory in bytes (default: 65336)')
    parser.add_argument('--output', default='gemm_config_tuned.json',
                       help='Output JSON file (default: gemm_config_tuned.json)')

    args = parser.parse_args()

    # Parse sizes and layouts
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
        baselines = None  # Use default baselines

    print("UCB-based Bayesian Optimization Tuner")
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

    tuner = UCBBayesianTuner(
        max_shared_memory=args.max_memory,
        gpu_arch=args.gpu_arch,
        baselines=baselines
    )

    results = tuner.tune_all(sizes=sizes, ab_layouts=ab_layouts, max_evaluations=args.budget)

    # Generate configuration JSON
    configs = []
    processed_configs = set()  # Track (M,N,K,A,B) combinations to avoid duplicates

    for size_results in results.values():
        for result in size_results.values():
            # Extract A,B layout from the result
            layout_a = result["layout"]["A"]
            layout_b = result["layout"]["B"]

            # Create a key to avoid duplicates for same (M,N,K,A,B)
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
    print("UCB BAYESIAN OPTIMIZATION RESULTS:")
    print("="*80)

    total_evaluations = 0
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
            print(f"  {ab_key}: {config['warps_m']},{config['warps_n']},"
                  f"{config['warp_tile_m']},{config['warp_tile_n']} -> "
                  f"{result['avg_time_ms']:.3f}ms ({result['evaluations']} evals)")
            total_evaluations += result['evaluations']
            count += 1

    if count > 0:
        avg_evals = total_evaluations / count
        print(f"\nTotal evaluations: {total_evaluations}")
        print(f"Average evaluations per (A,B) problem: {avg_evals:.1f}")
    print(f"Configuration saved to: {args.output}")

if __name__ == "__main__":
    main()
