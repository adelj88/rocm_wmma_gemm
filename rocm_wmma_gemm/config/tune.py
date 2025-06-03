#!/usr/bin/env python3

import subprocess
import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import re
warnings.filterwarnings('ignore')

# Matrix sizes to test (M, N, K)
SIZES = [
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

# Layout combinations (A, B, C) where 0=row_major, 1=col_major
LAYOUTS = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
]

class BayesianGEMMTuner:
    def __init__(self, n_initial_samples=15, n_iterations=30, max_shared_memory=65336):
        """
        Bayesian optimization tuner for GEMM kernels.

        Args:
            n_initial_samples: Number of random initial samples
            n_iterations: Number of BO iterations
            max_shared_memory: Maximum shared memory in bytes (RDNA3 limit: 65336)
        """
        self.n_initial_samples = n_initial_samples
        self.n_iterations = n_iterations
        self.max_shared_memory = max_shared_memory

        # Valid discrete values for each parameter
        self.valid_values = {
            'warps_m': [2, 4, 8],
            'warps_n': [2, 4, 8],
            'warp_tile_m': [2, 4, 8],
            'warp_tile_n': [2, 4, 8]
        }

        # Initialize GP model with more stable kernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True
        )

        self.scaler = StandardScaler()

    def _check_memory_constraint(self, config):
        """
        Check if configuration violates shared memory constraints for RDNA3.

        Uses the exact formula from your kernel code:
        block_m = warps_m * warp_tile_m * wmma_tile (16)
        block_n = warps_n * warp_tile_n * wmma_tile (16)
        block_k = wmma_tile (16)
        lds_size = (block_m * block_k) + (block_k * block_n)
        total_memory = 2 * lds_size * sizeof(half) (2 bytes)
        """
        warps_m, warps_n, warp_tile_m, warp_tile_n = config

        wmma_tile = 16
        block_m = warps_m * warp_tile_m * wmma_tile
        block_n = warps_n * warp_tile_n * wmma_tile
        block_k = wmma_tile

        lds_size = (block_m * block_k) + (block_k * block_n)
        total_memory_bytes = 2 * lds_size * 2  # 2 for double buffering, 2 for sizeof(half)

        return total_memory_bytes <= self.max_shared_memory

    def _parse_benchmark_output(self, output):
        """Parse the benchmark output to extract timing in milliseconds."""
        lines = output.strip().split('\n')

        # Look for the dynamic_kernel line with timing
        for line in lines:
            if 'dynamic_kernel/manual_time' in line:
                # Extract time value (format: "14.3 ms")
                match = re.search(r'(\d+\.?\d*)\s*ms', line)
                if match:
                    return float(match.group(1))

        return None

    def _evaluate_config(self, M, N, K, layout_a, layout_b, layout_c, config):
        """Evaluate a single configuration and return timing in milliseconds."""
        warps_m, warps_n, warp_tile_m, warp_tile_n = config

        # Check memory constraints first
        if not self._check_memory_constraint(config):
            return float('inf')

        # Check reasonable resource limits
        total_warps = warps_m * warps_n
        if total_warps > 16:  # Reasonable upper limit for GPU occupancy
            return float('inf')

        try:
            result = subprocess.run([
                "./rocm_wmma_gemm/tuner",
                str(M), str(N), str(K),
                str(warps_m), str(warps_n),
                str(warp_tile_m), str(warp_tile_n),
                str(layout_a), str(layout_b), str(layout_c)
            ], capture_output=True, text=True, timeout=60, check=False)

            if result.returncode != 0:
                print(f"Tuner failed for config {config}: {result.stderr}")
                return float('inf')

            # Parse timing from benchmark output
            time_ms = self._parse_benchmark_output(result.stdout)

            if time_ms is None:
                print(f"Could not parse timing from output for config {config}")
                return float('inf')

            return time_ms

        except subprocess.TimeoutExpired:
            print(f"Timeout for config {config}")
            return float('inf')
        except Exception as e:
            print(f"Error evaluating config {config}: {e}")
            return float('inf')

    def _discretize_config(self, config):
        """Convert continuous config to valid discrete values."""
        discrete_config = []
        param_names = ['warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n']

        for i, (param_name, value) in enumerate(zip(param_names, config)):
            valid_vals = self.valid_values[param_name]
            # Find closest valid value
            closest_idx = np.argmin(np.abs(np.array(valid_vals) - value))
            discrete_config.append(valid_vals[closest_idx])

        return tuple(discrete_config)

    def _config_to_features(self, config):
        """Convert config tuple to feature vector for GP."""
        return np.array(config, dtype=np.float64)

    def _acquisition_function(self, X):
        """Expected Improvement acquisition function."""
        X = np.atleast_2d(X)

        if len(self.X_observed) == 0:
            return np.zeros(X.shape[0])

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1).flatten()

        # Expected Improvement for minimization
        best_y = np.min(self.y_observed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = (best_y - mu) / (sigma + 1e-9)
            ei = (best_y - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei

    def _get_next_point(self):
        """Get next point to evaluate using acquisition function optimization."""
        if len(self.X_observed) == 0:
            # Random sampling for initial points
            config = []
            for param_name in ['warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n']:
                config.append(np.random.choice(self.valid_values[param_name]))
            return tuple(config)

        # Optimize acquisition function over valid discrete configurations
        best_score = -float('inf')
        best_config = None

        # Evaluate acquisition function for all valid combinations
        for warps_m in self.valid_values['warps_m']:
            for warps_n in self.valid_values['warps_n']:
                for warp_tile_m in self.valid_values['warp_tile_m']:
                    for warp_tile_n in self.valid_values['warp_tile_n']:
                        config = (warps_m, warps_n, warp_tile_m, warp_tile_n)

                        # Check if this config satisfies constraints
                        if not self._check_memory_constraint(config):
                            continue

                        if warps_m * warps_n > 16:  # Resource limit
                            continue

                        # Calculate acquisition score
                        x_features = self._config_to_features(config).reshape(1, -1)
                        x_scaled = self.scaler.transform(x_features)
                        score = self._acquisition_function(x_scaled)[0]

                        if score > best_score:
                            best_score = score
                            best_config = config

        if best_config is None:
            # Fallback to random valid config
            valid_configs = []
            for warps_m in self.valid_values['warps_m']:
                for warps_n in self.valid_values['warps_n']:
                    for warp_tile_m in self.valid_values['warp_tile_m']:
                        for warp_tile_n in self.valid_values['warp_tile_n']:
                            config = (warps_m, warps_n, warp_tile_m, warp_tile_n)
                            if (self._check_memory_constraint(config) and
                                warps_m * warps_n <= 16):
                                valid_configs.append(config)

            if valid_configs:
                return valid_configs[np.random.randint(len(valid_configs))]
            else:
                return (2, 2, 2, 2)  # Minimal safe config

        return best_config

    def tune_size_layout(self, M, N, K, layout_a, layout_b, layout_c):
        """Tune a specific matrix size and layout combination."""
        print(f"\nTuning {M}×{N}×{K} with layout A={layout_a}, B={layout_b}, C={layout_c}")

        # Reset for this size/layout combination
        self.X_observed = []
        self.y_observed = []
        configs_tried = set()

        # Initial random sampling
        print("Phase 1: Initial random sampling...")
        valid_initial_configs = []

        # Generate all valid configurations first
        all_valid_configs = []
        for warps_m in self.valid_values['warps_m']:
            for warps_n in self.valid_values['warps_n']:
                for warp_tile_m in self.valid_values['warp_tile_m']:
                    for warp_tile_n in self.valid_values['warp_tile_n']:
                        config = (warps_m, warps_n, warp_tile_m, warp_tile_n)
                        if (self._check_memory_constraint(config) and
                            warps_m * warps_n <= 16):
                            all_valid_configs.append(config)

        if not all_valid_configs:
            print("No valid configurations found!")
            return None

        print(f"Found {len(all_valid_configs)} valid configurations")

        # Sample initial configurations, ensuring (4,4,4,4) baseline is included
        baseline_config = (4, 4, 4, 4)
        initial_configs_to_test = []

        # Always include baseline if it's valid
        if baseline_config in all_valid_configs:
            initial_configs_to_test.append(baseline_config)
            print(f"  Including baseline config {baseline_config}")

        # Sample remaining configurations randomly
        n_remaining = min(self.n_initial_samples - len(initial_configs_to_test),
                         len(all_valid_configs) - len(initial_configs_to_test))

        if n_remaining > 0:
            available_configs = [c for c in all_valid_configs if c not in initial_configs_to_test]
            if available_configs:
                remaining_indices = np.random.choice(len(available_configs),
                                                   size=min(n_remaining, len(available_configs)),
                                                   replace=False)
                for idx in remaining_indices:
                    initial_configs_to_test.append(available_configs[idx])

        for config in initial_configs_to_test:
            configs_tried.add(config)

            time_ms = self._evaluate_config(M, N, K, layout_a, layout_b, layout_c, config)

            if time_ms != float('inf'):
                self.X_observed.append(self._config_to_features(config))
                self.y_observed.append(time_ms)

                print(f"  Config {config}: {time_ms:.3f} ms")

        if len(self.X_observed) == 0:
            print("No valid configurations found in initial sampling!")
            return None

        # Convert to numpy arrays and fit scaler
        self.X_observed = np.array(self.X_observed)
        self.y_observed = np.array(self.y_observed)
        self.scaler.fit(self.X_observed)

        # Bayesian optimization iterations
        print("Phase 2: Bayesian optimization...")

        for iteration in range(self.n_iterations):
            # Fit GP model
            X_scaled = self.scaler.transform(self.X_observed)
            self.gp.fit(X_scaled, self.y_observed)

            # Get next configuration to try
            config = self._get_next_point()

            if config in configs_tried:
                # Try random untested config if we get a duplicate
                untested_configs = [c for c in all_valid_configs if c not in configs_tried]
                if untested_configs:
                    config = untested_configs[np.random.randint(len(untested_configs))]
                else:
                    break  # All configs tested

            configs_tried.add(config)

            # Evaluate new configuration
            time_ms = self._evaluate_config(M, N, K, layout_a, layout_b, layout_c, config)

            if time_ms != float('inf'):
                # Add to observed data
                new_X = self._config_to_features(config).reshape(1, -1)
                self.X_observed = np.vstack([self.X_observed, new_X])
                self.y_observed = np.append(self.y_observed, time_ms)

                # Update scaler
                self.scaler.fit(self.X_observed)

                best_time = np.min(self.y_observed)
                best_idx = np.argmin(self.y_observed)
                best_config = self._discretize_config(self.X_observed[best_idx])

                print(f"  Iter {iteration+1}: Config {config} -> {time_ms:.3f} ms "
                      f"(best: {best_config} -> {best_time:.3f} ms)")

        # Return best configuration
        best_idx = np.argmin(self.y_observed)
        best_config = self._discretize_config(self.X_observed[best_idx])
        best_time = self.y_observed[best_idx]

        # Calculate memory usage for the best config
        wmma_tile = 16
        block_m = best_config[0] * best_config[2] * wmma_tile
        block_n = best_config[1] * best_config[3] * wmma_tile
        block_k = wmma_tile
        lds_size = (block_m * block_k) + (block_k * block_n)
        memory_used = 2 * lds_size * 2  # Double buffering * sizeof(half)

        print(f"  Best config uses {memory_used}/{self.max_shared_memory} bytes shared memory")

        return {
            'config': {
                'warps_m': int(best_config[0]),
                'warps_n': int(best_config[1]),
                'warp_tile_m': int(best_config[2]),
                'warp_tile_n': int(best_config[3])
            },
            'time_ms': float(best_time),
            'total_evaluations': len(configs_tried),
            'memory_used_bytes': memory_used
        }

    def tune_all(self):
        """Tune all size and layout combinations."""
        results = {}

        for M, N, K in SIZES:
            size_key = f"{M}x{N}x{K}"
            results[size_key] = {}

            for layout_a, layout_b, layout_c in LAYOUTS:
                layout_key = f"{layout_a}_{layout_b}_{layout_c}"

                result = self.tune_size_layout(M, N, K, layout_a, layout_b, layout_c)

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
                        "evaluations": result['total_evaluations'],
                        "memory_used_bytes": result['memory_used_bytes']
                    }

        return results

def main():
    print("Starting Bayesian Optimization GEMM Tuning...")
    print(f"RDNA3 shared memory limit: 65336 bytes")
    print(f"Will tune {len(SIZES)} matrix sizes with {len(LAYOUTS)} layout combinations each")

    tuner = BayesianGEMMTuner(
        n_initial_samples=12,  # Reasonable initial exploration
        n_iterations=25,       # Good balance of exploration/exploitation
        max_shared_memory=65336  # RDNA3 limit
    )

    results = tuner.tune_all()

    # Generate configuration JSON
    configs = []
    for size_results in results.values():
        for result in size_results.values():
            config = {
                "range": {"M": result["M"], "N": result["N"], "K": result["K"]},
                "layout": result["layout"],
                "config": result["config"]
            }
            configs.append(config)

    config_data = {"configurations": configs}

    # Save results
    with open("gemm_config_tuned.json", "w") as f:
        json.dump(config_data, f, indent=4)

    # Print summary
    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION RESULTS:")
    print("="*80)

    total_evaluations = 0
    for size_key, size_results in results.items():
        print(f"\n{size_key}:")
        for layout_key, result in size_results.items():
            config = result['config']
            memory_used = result['memory_used_bytes']
            print(f"  {layout_key}: {config['warps_m']},{config['warps_n']},"
                  f"{config['warp_tile_m']},{config['warp_tile_n']} -> "
                  f"{result['avg_time_ms']:.3f} ms ({result['evaluations']} evals, "
                  f"{memory_used} bytes)")
            total_evaluations += result['evaluations']

    print(f"\nTotal configurations evaluated: {total_evaluations}")
    print(f"Average evaluations per problem: {total_evaluations / (len(SIZES) * len(LAYOUTS)):.1f}")
    print(f"Configurations saved to: gemm_config_tuned.json")

if __name__ == "__main__":
    main()
