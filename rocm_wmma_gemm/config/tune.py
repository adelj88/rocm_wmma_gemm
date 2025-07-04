#!/usr/bin/env python3

import subprocess
import json
import numpy as np
import random
import re
import argparse
import sys
import warnings

warnings.filterwarnings('ignore')

class GAWMMATuner:
    """Genetic Algorithm-based WMMA kernel tuner for 4-parameter discrete space."""

    def __init__(self, max_shared_memory=65336, gpu_arch="gfx1100", baselines=None, random_seed=42):
        self.max_shared_memory = max_shared_memory
        self.gpu_arch = gpu_arch

        # Set random seed for reproducible results
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.random_seed = random_seed

        # Define baseline configurations to seed initial population
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
            ]
        else:
            self.baselines = baselines

        # Parameter space - 4 parameters only
        self.valid_values = {
            'warps_m': [1, 2, 4, 8],
            'warps_n': [1, 2, 4, 8],
            'warp_tile_m': [1, 2, 4, 8],
            'warp_tile_n': [1, 2, 4, 8]
        }

        # Generate all valid configurations
        self.valid_configs = self._generate_valid_configs()
        print(f"Generated {len(self.valid_configs)} valid configurations")

        # GA Parameters - optimized for ~200 configuration space
        self.population_size = 15
        self.elite_size = 3         # Keep best 3 each generation (20%)
        self.mutation_rate = 0.2    # 20% mutation rate for smaller space
        self.crossover_rate = 0.85  # 85% crossover rate

        # State tracking
        self.evaluated_configs = {}  # Cache results to avoid re-evaluation
        self.generation_history = []
        self.all_time_best = None
        self.all_time_best_fitness = float('inf')

    def _generate_valid_configs(self):
        """Generate all valid discrete configurations."""
        configs = []
        count = 0
        for warps_m in self.valid_values['warps_m']:
            for warps_n in self.valid_values['warps_n']:
                for warp_tile_m in self.valid_values['warp_tile_m']:
                    for warp_tile_n in self.valid_values['warp_tile_n']:
                        count += 1
                        config = (warps_m, warps_n, warp_tile_m, warp_tile_n)
                        if self._check_constraints(config):
                            configs.append(config)

        print(f"Checked {count} total combinations, {len(configs)} valid after constraints")
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

        # Resource constraint - RDNA GPUs: max 1024 threads per block, warp size = 32
        total_warps = warps_m * warps_n
        if total_warps > 32:  # 1024 threads ÷ 32 threads per warp = 32 warps max
            return False

        return True

    def _generate_random_individual(self):
        """Generate a random valid individual."""
        max_attempts = 100
        for _ in range(max_attempts):
            individual = tuple(
                random.choice(self.valid_values[param])
                for param in ['warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n']
            )
            if self._check_constraints(individual):
                return individual

        # Fallback to first baseline if random generation fails
        return self.baselines[0]

    def _initialize_population(self):
        """Initialize population with baselines + random individuals."""
        population = []

        # Add baselines first (ensure they're valid)
        for baseline in self.baselines:
            if self._check_constraints(baseline):
                population.append(baseline)

        # Fill remaining slots with random individuals
        while len(population) < self.population_size:
            individual = self._generate_random_individual()
            # Avoid duplicates
            if individual not in population:
                population.append(individual)

        return population

    def _crossover(self, parent1, parent2):
        """Uniform crossover - randomly pick each gene from either parent."""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        child1, child2 = [], []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])

        return tuple(child1), tuple(child2)

    def _mutate(self, individual):
        """Mutate individual by randomly changing some parameters."""
        if random.random() > self.mutation_rate:
            return individual

        individual_list = list(individual)
        param_names = ['warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n']

        # Mutate 1 parameter (smaller space, less aggressive mutation)
        num_mutations = 1
        for _ in range(num_mutations):
            param_idx = random.randint(0, len(param_names) - 1)
            param_name = param_names[param_idx]
            individual_list[param_idx] = random.choice(self.valid_values[param_name])

        return tuple(individual_list)

    def _repair_individual(self, individual):
        """Repair invalid individual by adjusting parameters."""
        if self._check_constraints(individual):
            return individual

        # Try several repair attempts
        for _ in range(20):
            # Try mutating to fix constraints
            repaired = self._mutate(individual)
            if self._check_constraints(repaired):
                return repaired

        # If repair fails, generate a new random individual
        return self._generate_random_individual()

    def _tournament_selection(self, population, fitnesses, tournament_size=3):
        """Tournament selection for parent selection."""
        selected = []
        for _ in range(len(population)):
            # Select tournament_size random individuals
            tournament_indices = random.sample(range(len(population)),
                                             min(tournament_size, len(population)))
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

            # Find winner (lowest fitness for minimization)
            winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
            selected.append(population[winner_idx])

        return selected

    def _evaluate_config(self, M, N, K, layout_a, layout_b, layout_c, config):
        """Evaluate a configuration and return timing."""
        # Check cache first
        cache_key = (M, N, K, layout_a, layout_b, layout_c, config)
        if cache_key in self.evaluated_configs:
            return self.evaluated_configs[cache_key]

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
                time_ms = float('inf')
            else:
                time_ms = self._parse_benchmark_output(result.stdout)
                if time_ms is None:
                    time_ms = float('inf')

            # Cache result
            self.evaluated_configs[cache_key] = time_ms
            return time_ms

        except Exception:
            self.evaluated_configs[cache_key] = float('inf')
            return float('inf')

    def _parse_benchmark_output(self, output):
        """Parse benchmark output to extract timing."""
        lines = output.strip().split('\n')
        for line in lines:
            if 'dynamic_kernel/manual_time' in line:
                match = re.search(r'(\d+\.?\d*)\s*ms', line)
                if match:
                    return float(match.group(1))
        return None

    def _evaluate_population(self, population, M, N, K, layout_a, layout_b):
        """Evaluate entire population and return fitness values."""
        fitnesses = []
        for individual in population:
            fitness = self._evaluate_config(M, N, K, layout_a, layout_b, 0, individual)
            fitnesses.append(fitness)

            # Track all-time best
            if fitness < self.all_time_best_fitness:
                self.all_time_best_fitness = fitness
                self.all_time_best = individual
            elif fitness == self.all_time_best_fitness and self.all_time_best_fitness != float('inf'):
                # Equal performance - update to newer config
                self.all_time_best = individual

        return fitnesses

    def tune_ab_layout(self, M, N, K, layout_a, layout_b, max_evaluations=60):
        """Tune for (A,B) layout pair using Genetic Algorithm."""
        print(f"\nTuning {M}×{N}×{K} with layout A={layout_a}, B={layout_b}")
        print(f"GA Parameters: pop_size={self.population_size}, elite={self.elite_size}, "
              f"mutation_rate={self.mutation_rate}, crossover_rate={self.crossover_rate}")

        # Reset state for this problem
        self.evaluated_configs = {}
        self.generation_history = []
        self.all_time_best = None
        self.all_time_best_fitness = float('inf')

        # Initialize population
        population = self._initialize_population()
        evaluations_used = 0
        generation = 0

        print("Starting Genetic Algorithm...")

        while evaluations_used < max_evaluations:
            generation += 1

            # Evaluate population
            fitnesses = self._evaluate_population(population, M, N, K, layout_a, layout_b)
            evaluations_used += len(population)

            # Track generation stats
            valid_fitnesses = [f for f in fitnesses if f != float('inf')]
            if valid_fitnesses:
                gen_best = min(fitnesses)
                gen_avg = np.mean(valid_fitnesses)
                gen_best_idx = np.argmin(fitnesses)
                gen_best_individual = population[gen_best_idx]

                self.generation_history.append({
                    'generation': generation,
                    'best_fitness': gen_best,
                    'avg_fitness': gen_avg,
                    'best_individual': gen_best_individual
                })

                print(f"  Gen {generation}: Best = {gen_best:.3f}ms, "
                      f"Avg = {gen_avg:.3f}ms, Valid = {len(valid_fitnesses)}/{len(population)}, "
                      f"Evals = {evaluations_used}")
            else:
                print(f"  Gen {generation}: No valid solutions found!")

            # Check if we have budget for next generation
            if evaluations_used + self.population_size > max_evaluations:
                print(f"  Stopping: Next generation would exceed budget ({evaluations_used + self.population_size} > {max_evaluations})")
                break

            # Create next generation
            # 1. Elitism - keep best individuals
            elite_indices = np.argsort(fitnesses)[:self.elite_size]
            next_population = [population[i] for i in elite_indices]

            # 2. Tournament selection for breeding
            selected_parents = self._tournament_selection(population, fitnesses)

            # 3. Crossover and mutation to fill remaining slots
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(selected_parents, 2)
                child1, child2 = self._crossover(parent1, parent2)

                # Mutate children
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                # Repair invalid children
                child1 = self._repair_individual(child1)
                child2 = self._repair_individual(child2)

                next_population.extend([child1, child2])

            # Trim to exact population size
            population = next_population[:self.population_size]

        if self.all_time_best is None:
            print("  No valid configuration found!")
            return None

        # Calculate memory usage for best config
        wmma_tile = 16
        block_m = self.all_time_best[0] * self.all_time_best[2] * wmma_tile
        block_n = self.all_time_best[1] * self.all_time_best[3] * wmma_tile
        block_k = wmma_tile
        lds_size = (block_m * block_k) + (block_k * block_n)
        memory_used = 2 * lds_size * 2

        print(f"  Best config: {self.all_time_best} -> {self.all_time_best_fitness:.3f}ms")
        print(f"  Memory usage: {memory_used}/{self.max_shared_memory} bytes")
        print(f"  Generations: {generation}")
        print(f"  Total evaluations: {evaluations_used}")

        return {
            'config': {
                'warps_m': int(self.all_time_best[0]),
                'warps_n': int(self.all_time_best[1]),
                'warp_tile_m': int(self.all_time_best[2]),
                'warp_tile_n': int(self.all_time_best[3])
            },
            'time_ms': float(self.all_time_best_fitness),
            'evaluations': evaluations_used,
            'generations': generation,
            'memory_used_bytes': memory_used,
            'generation_history': self.generation_history
        }

    def tune_all(self, sizes=None, ab_layouts=None, max_evaluations=60):
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
                            "generations": result['generations'],
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
        description='Genetic Algorithm WMMA Tuner (4-parameter version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default sizes and layouts
  python ga_wmma_tune.py

  # Tune specific matrix sizes
  python ga_wmma_tune.py --sizes 1024,1024,1024 2048,2048,2048

  # Use specific seed for reproducible results
  python ga_wmma_tune.py --seed 123

  # Different seed for different exploration
  python ga_wmma_tune.py --seed 456

  # Adjust GA parameters for larger search
  python ga_wmma_tune.py --budget 60 --pop-size 20

  # Tune specific (A,B) layouts
  python ga_wmma_tune.py --layouts row_major,col_major col_major,col_major

  # Use shorthand for layouts
  python ga_wmma_tune.py --layouts r,c c,c

  # Add custom baselines
  python ga_wmma_tune.py --baselines 4,4,4,4 2,2,4,4 8,2,2,2

  # Custom GPU architecture
  python ga_wmma_tune.py --gpu-arch gfx1103
        """)

    parser.add_argument('--sizes', nargs='*',
                       help='Matrix sizes as M,N,K (e.g., 1024,1024,1024 2048,2048,2048)')
    parser.add_argument('--layouts', nargs='*',
                       help='Matrix (A,B) layouts as A,B (e.g., row_major,col_major or r,c)')
    parser.add_argument('--baselines', nargs='*',
                       help='Baseline configs as warps_m,warps_n,warp_tile_m,warp_tile_n (e.g., 4,4,4,4 2,2,4,4)')
    parser.add_argument('--budget', type=int, default=60,
                       help='Evaluation budget per (A,B) layout combination (default: 60)')
    parser.add_argument('--pop-size', type=int, default=15,
                       help='Population size (default: 15)')
    parser.add_argument('--mutation-rate', type=float, default=0.2,
                       help='Mutation rate (default: 0.2)')
    parser.add_argument('--crossover-rate', type=float, default=0.85,
                       help='Crossover rate (default: 0.85)')
    parser.add_argument('--elite-size', type=int, default=3,
                       help='Number of elite individuals to keep (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
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

    print("Genetic Algorithm WMMA Tuner (4-parameter version)")
    print(f"GPU Architecture: {args.gpu_arch}")
    print(f"Random seed: {args.seed}")
    print(f"Population size: {args.pop_size}")
    print(f"Mutation rate: {args.mutation_rate}")
    print(f"Crossover rate: {args.crossover_rate}")
    print(f"Elite size: {args.elite_size}")
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

    # Create tuner with command line parameters
    tuner = GAWMMATuner(
        max_shared_memory=args.max_memory,
        gpu_arch=args.gpu_arch,
        baselines=baselines,
        random_seed=args.seed
    )

    # Override GA parameters from command line
    tuner.population_size = args.pop_size
    tuner.mutation_rate = args.mutation_rate
    tuner.crossover_rate = args.crossover_rate
    tuner.elite_size = args.elite_size

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
    print("GENETIC ALGORITHM WMMA RESULTS:")
    print("="*80)

    total_evaluations = 0
    total_generations = 0
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
                  f"{result['avg_time_ms']:.3f}ms ({result['evaluations']} evals, {result['generations']} gens)")
            total_evaluations += result['evaluations']
            total_generations += result['generations']
            count += 1

    if count > 0:
        avg_evals = total_evaluations / count
        avg_gens = total_generations / count
        print(f"\nTotal evaluations: {total_evaluations}")
        print(f"Total generations: {total_generations}")
        print(f"Average evaluations per (A,B) problem: {avg_evals:.1f}")
        print(f"Average generations per (A,B) problem: {avg_gens:.1f}")
    print(f"Configuration saved to: {args.output}")

if __name__ == "__main__":
    main()
