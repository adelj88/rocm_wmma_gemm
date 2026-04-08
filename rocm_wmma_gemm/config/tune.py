#!/usr/bin/env python3
"""WMMA kernel configuration tuner using Parameter-less GOMEA.

This script optimizes GPU kernel configurations (block sizes, warps, booleans)
using the Gene-pool Optimal Mixing Evolutionary Algorithm (GOMEA). GPU tuning
has strong "epistasis" (parameters interact heavily, e.g., block size and bits).

Glossary of Advanced Mechanics Included:
1. Niching (Hall of Fame): Instead of a single elitist, the algorithm maintains
   up to 4 diverse elites globally and locally. Stagnant individuals mix with
   their *closest* elite. This preserves diverse lineages and destroys gravity wells.
2. Global Elite Linkage Learning (FOS): Calculates Mutual Information on the top
   50% of ALL historical evaluations to mathematically prove which parameters must
   move together. Includes Spurious Linkage Prevention to keep building blocks pure.
3. Multi-Niche Seeding (Knowledge Transfer): Injects the entire Global Hall of Fame
   into newly spawned larger populations to give them a massive, diverse head start.
4. Interleaved Multi-Start (IMS): Parameter-less population sizing. Spawns
   concurrent populations of increasing sizes (4, 8, 16...) and interleaves them.
5. Strict Forced Improvement (FI): If an individual stagnates, it is forced to
   mix with its closest elite. If it still fails, it is randomized.
6. Descending Stratified Initialization: Intelligently seeds new populations by
   testing the largest valid block sizes first, preventing wasted budget.
7. Layout Sharing & Competitive Baselines: Reuses row-major C configs as baselines
   for col-major C runs, racing them against input files to halve the budget.
8. Stagnation Termination: Terminates small populations that bounce between local
   optima, reallocating the evaluation budget to larger, exploratory populations.
"""

import subprocess
import json
import re
import argparse
import sys
import random
import itertools
import math
from collections import defaultdict
from pathlib import Path


class WMMATuner:
    """Parameter-less GOMEA tuner for WMMA kernel configurations."""

    VALID_BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, max_shared_memory=65336, gpu_arch="gfx1100",
                 random_seed=42):
        self.max_shared_memory = max_shared_memory
        self.gpu_arch = gpu_arch
        self.random_seed = random_seed

        random.seed(random_seed)

        # All tunable parameters. Edit this dictionary to add/remove parameters.
        self.param_space = {
            'warps_m':          [1, 2, 4, 8],
            'warps_n':          [1, 2, 4, 8],
            'warp_tile_m':      [1, 2, 4, 8],
            'warp_tile_n':      [1, 2, 4, 8],
            'swizzle':          [4, 8, 16],
            'bits':             [128, 256],
            'buffer_first':     [False, True],
            'use_async':        [False, True],
            'use_direct_write': [False, True],
        }

        self.param_names = list(self.param_space.keys())
        self.n_params = len(self.param_names)
        self.IDX = {name: i for i, name in enumerate(self.param_names)}

        # Pre-generate all valid configurations to sample from later
        self.valid_configs = self._generate_valid_configs()
        print(f"Total valid configurations: {len(self.valid_configs)}")

    # ------------------------------------------------------------------
    # Configuration & Constraint Helpers
    # ------------------------------------------------------------------

    def _generate_valid_configs(self):
        """Generates the full Cartesian product of parameters and filters invalid ones."""
        all_values =[self.param_space[p] for p in self.param_names]
        configs =[]
        for combo in itertools.product(*all_values):
            if self._check_constraints(combo):
                configs.append(combo)
        return configs

    def _check_constraints(self, config, layout_a=None, layout_b=None):
        """Hardware constraint checker. Rejects configs that exceed shared memory or warp limits."""
        wm = config[self.IDX['warps_m']]
        wn = config[self.IDX['warps_n']]
        wtm = config[self.IDX['warp_tile_m']]
        wtn = config[self.IDX['warp_tile_n']]
        bits = config[self.IDX['bits']]
        swizzle = config[self.IDX['swizzle']]

        if wm == 0 or wn == 0 or wtm == 0 or wtn == 0 or bits == 0 or swizzle == 0:
            return False

        if layout_a == 0 and layout_b == 0:
            if swizzle != 8:
                return False

        wmma_tile = 16
        block_m = wm * wtm * wmma_tile
        block_n = wn * wtn * wmma_tile
        block_k = wmma_tile

        if layout_a is not None and layout_b is not None:
            pad_a = 8 if layout_a == 0 else 0
            pad_b = 8 if layout_b == 1 else 0
        else:
            pad_a, pad_b = 8, 8

        lds = block_m * (block_k + pad_a) + (block_k + pad_b) * block_n
        if 2 * lds * 2 > self.max_shared_memory:
            return False
        if wm * wn > 32:
            return False
        return True

    def _config_to_dict(self, config):
        """Converts an internal tuple config back to a readable dictionary."""
        return {name: config[i] for i, name in enumerate(self.param_names)}

    # ------------------------------------------------------------------
    # Benchmarking Execution
    # ------------------------------------------------------------------

    def _parse_benchmark_output(self, output):
        """Extracts the execution time (in ms) from the C++ tuner stdout."""
        for line in output.strip().split('\n'):
            if 'manual_time_mean' in line and 'repeats:' in line:
                m = re.search(r'(\d+\.?\d*)\s+ms', line)
                if m: return float(m.group(1))
            elif ('dynamic_kernel' in line and 'manual_time' in line
                  and '_mean' not in line):
                m = re.search(r'(\d+\.?\d*)\s+ms', line)
                if m: return float(m.group(1))
        return None

    def _evaluate_config(self, M, N, K, la, lb, lc, config):
        """Compiles and runs the kernel configuration on the GPU."""
        d = self._config_to_dict(config)
        try:
            result = subprocess.run([
                "./rocm_wmma_gemm/tuner",
                str(M), str(N), str(K),
                str(d['warps_m']), str(d['warps_n']),
                str(d['warp_tile_m']), str(d['warp_tile_n']),
                str(d['swizzle']),
                str(d['bits']),
                str(1 if d['buffer_first'] else 0),
                str(1 if d['use_async'] else 0),
                str(1 if d['use_direct_write'] else 0),
                str(la), str(lb), str(lc), self.gpu_arch
            ], capture_output=True, text=True, timeout=60, check=False)
            if result.returncode != 0:
                return float('inf')
            t = self._parse_benchmark_output(result.stdout)
            return t if t is not None else float('inf')
        except Exception:
            return float('inf')

    # ------------------------------------------------------------------
    # Linkage Learning (Dynamic FOS via Mutual Information + UPGMA)
    # ------------------------------------------------------------------

    def _learn_linkage_tree(self, population_configs):
        """
        Builds the Family of Subsets (FOS) dynamically.
        Calculates Mutual Information (MI) between all parameters to see which ones
        are correlated (e.g., specific booleans that only work well together).
        Uses UPGMA hierarchical clustering to group them into a linkage tree.
        """
        n_params = self.n_params
        pop_size = len(population_configs)

        # Level 0: Univariate (each parameter alone)
        fos_list = [(i,) for i in range(n_params)]

        if pop_size < 2:
            return fos_list

        # 1. Compute Pairwise Mutual Information (MI)
        MI = [[0.0] * n_params for _ in range(n_params)]
        for i in range(n_params):
            for j in range(i + 1, n_params):
                counts_i = defaultdict(int)
                counts_j = defaultdict(int)
                counts_ij = defaultdict(int)
                for cfg in population_configs:
                    counts_i[cfg[i]] += 1
                    counts_j[cfg[j]] += 1
                    counts_ij[(cfg[i], cfg[j])] += 1

                mi = 0.0
                for (val_i, val_j), c_ij in counts_ij.items():
                    p_ij = c_ij / pop_size
                    p_i = counts_i[val_i] / pop_size
                    p_j = counts_j[val_j] / pop_size
                    mi += p_ij * math.log(p_ij / (p_i * p_j))

                MI[i][j] = mi
                MI[j][i] = mi

        # 2. UPGMA Hierarchical Clustering
        active_clusters = {i: [i] for i in range(n_params)}
        next_id = n_params

        while len(active_clusters) > 1:
            best_pair = None
            max_mi = -float('inf')
            c_ids = list(active_clusters.keys())

            for idx1 in range(len(c_ids)):
                for idx2 in range(idx1 + 1, len(c_ids)):
                    id1, id2 = c_ids[idx1], c_ids[idx2]
                    c1, c2 = active_clusters[id1], active_clusters[id2]
                    sum_mi = sum(MI[p1][p2] for p1 in c1 for p2 in c2)
                    avg_mi = sum_mi / (len(c1) * len(c2))

                    if avg_mi > max_mi:
                        max_mi = avg_mi
                        best_pair = (id1, id2)

            # --- Spurious Linkage Prevention (FOS Purity) ---
            # If the best remaining parameters have zero correlation, stop merging!
            # Forcing independent parameters into a group reduces mixing efficiency.
            if max_mi <= 1e-6:
                break
            # ------------------------------------------------

            id1, id2 = best_pair
            merged = active_clusters[id1] + active_clusters[id2]
            del active_clusters[id1]
            del active_clusters[id2]
            active_clusters[next_id] = merged
            next_id += 1

            if len(merged) < n_params:
                fos_list.append(tuple(merged))

        return fos_list

    # ------------------------------------------------------------------
    # GOMEA Optimal Mixing Operators
    # ------------------------------------------------------------------

    def _optimal_mix(self, individual, ind_time, population, ind_idx, evaluate,
                     layout_a, layout_b, rng, fos):
        """Standard Gene-pool mixing."""
        current = list(individual)
        current_time = ind_time
        strict_improved = False

        shuffled_fos = list(fos)
        rng.shuffle(shuffled_fos)

        for group in shuffled_fos:
            donor_idx = rng.randrange(len(population))
            while donor_idx == ind_idx and len(population) > 1:
                donor_idx = rng.randrange(len(population))
            _, donor_cfg = population[donor_idx]

            if all(current[i] == donor_cfg[i] for i in group):
                continue

            saved =[current[i] for i in group]
            for i in group:
                current[i] = donor_cfg[i]

            candidate = tuple(current)
            if not self._check_constraints(candidate, layout_a, layout_b):
                for i, v in zip(group, saved): current[i] = v
                continue

            t = evaluate(candidate)
            if t < current_time:
                current_time = t
                strict_improved = True
            elif t == current_time:
                current_time = t # Keep neutral walk
            else:
                for i, v in zip(group, saved): current[i] = v

        # ESCAPE HATCH: Full-config swap when no partial group improved
        if not strict_improved and len(population) > 1:
            donor_idx = rng.randrange(len(population))
            while donor_idx == ind_idx:
                donor_idx = rng.randrange(len(population))
            _, donor_cfg = population[donor_idx]

            if donor_cfg != tuple(current):
                if self._check_constraints(donor_cfg, layout_a, layout_b):
                    t = evaluate(donor_cfg)
                    if t < current_time:
                        current = list(donor_cfg)
                        current_time = t
                        strict_improved = True

        return tuple(current), current_time, strict_improved

    def _optimal_mix_donor(self, individual, ind_time, donor_cfg, evaluate,
                           layout_a, layout_b, rng, fos):
        """Forced Improvement mixing using a specific donor (the closest elite)."""
        current = list(individual)
        current_time = ind_time
        strict_improved = False

        shuffled_fos = list(fos)
        rng.shuffle(shuffled_fos)

        for group in shuffled_fos:
            if all(current[i] == donor_cfg[i] for i in group):
                continue

            saved = [current[i] for i in group]
            for i in group:
                current[i] = donor_cfg[i]

            candidate = tuple(current)
            if not self._check_constraints(candidate, layout_a, layout_b):
                for i, v in zip(group, saved): current[i] = v
                continue

            t = evaluate(candidate)
            if t < current_time:
                current_time = t
                strict_improved = True
            elif t == current_time:
                current_time = t
            else:
                for i, v in zip(group, saved): current[i] = v

        return tuple(current), current_time, strict_improved

    # ------------------------------------------------------------------
    # Main Tuning Loop (IMS)
    # ------------------------------------------------------------------

    def tune_layout(self, M, N, K, layout_a, layout_b, layout_c,
                    max_evaluations=300, existing_baseline=None):
        """Tunes a specific matrix size and layout using Parameter-less GOMEA."""
        print(f"\nTuning {M}x{N}x{K} layout A={layout_a} B={layout_b} C={layout_c}")
        print(f"  Budget: {max_evaluations} evals")

        rng = random.Random(self.random_seed)

        layout_valid =[c for c in self.valid_configs
                        if self._check_constraints(c, layout_a, layout_b)]
        print(f"  {len(layout_valid)} layout-valid configs")
        if not layout_valid:
            return None

        total_evals = 0
        best_config = None
        best_time = float('inf')
        seen = {}
        cache_hits = 0

        def evaluate(config):
            """Evaluates a config, handles caching, and tracks the global best."""
            nonlocal total_evals, cache_hits, best_config, best_time
            if config in seen:
                cache_hits += 1
                return seen[config]

            t = self._evaluate_config(M, N, K, layout_a, layout_b, layout_c, config)
            seen[config] = t
            total_evals += 1

            if t < best_time:
                best_time = t
                best_config = config
                print(f"  New best: {config} -> {t:.3f}ms (eval {total_evals})")
            elif t == best_time and best_time != float('inf'):
                best_config = config
                print(f"  Tied best (updated global best): {config} -> {t:.3f}ms (eval {total_evals})")
            # else:
                # print(f"  Eval {total_evals}: {config} -> {t:.3f}ms")

            return t

        # ==============================================================
        # Interleaved Multi-Start (IMS) Loop
        # ==============================================================
        populations = []
        pop_stagnation =[]
        pop_best_times = []
        global_elites =[]  # Tracks up to 4 diverse elites across the entire run
        ims_step = 0

        while total_evals < max_evaluations:
            ims_step += 1

            # IMS Scheduling: Creates the cascade 0, 1, 0, 2, 0, 1, 0, 3...
            c = (ims_step & -ims_step).bit_length() - 1

            # Initialize a new, larger population if we reach a new 'c' level
            if c == len(populations):
                pop_size = 4 * (2 ** c)
                new_pop =[]

                # Inject baseline into the very first population
                if c == 0 and existing_baseline is not None:
                    if self._check_constraints(existing_baseline, layout_a, layout_b):
                        new_pop.append((evaluate(existing_baseline), existing_baseline))

                # --- Multi-Niche Seeding (Knowledge Transfer) ---
                # Inject ALL diverse elites from the Global Hall of Fame into the new population.
                # Because we use Niching, this gives the population a massive head start on
                # multiple different hills without collapsing into a single gravity well!
                elif c > 0 and global_elites:
                    for e_t, e_cfg in global_elites:
                        if len(new_pop) < pop_size:
                            new_pop.append((e_t, e_cfg))
                # ------------------------------------------------

                unseen = [cfg for cfg in layout_valid if cfg not in seen]

                # --- Descending Stratified Initialization ---
                # Groups unseen configs by total block area and sorts descending.
                # Guarantees the algorithm tests the largest valid block sizes first.
                groups = defaultdict(list)
                for cfg in unseen:
                    area = cfg[0] * cfg[2] * cfg[1] * cfg[3]
                    groups[area].append(cfg)

                for g in groups.values():
                    rng.shuffle(g)

                keys = sorted(list(groups.keys()), reverse=True)

                sampled_cfgs =[]
                robin = 0
                while len(sampled_cfgs) < (pop_size - len(new_pop)) and keys:
                    added = False
                    for k in keys:
                        if len(sampled_cfgs) >= (pop_size - len(new_pop)): break
                        if robin < len(groups[k]):
                            sampled_cfgs.append(groups[k][robin])
                            added = True
                    robin += 1
                    if not added: break
                # --------------------------------------------

                for cfg in sampled_cfgs:
                    if total_evals >= max_evaluations: break
                    new_pop.append((evaluate(cfg), cfg))

                if new_pop:
                    populations.append(new_pop)
                    pop_stagnation.append(0)
                    pop_best_times.append(float('inf'))
                    print(f"  [IMS] Spawned Population {c} (size {len(new_pop)})")

            if c >= len(populations):
                continue

            # Stagnation Termination
            if pop_stagnation[c] >= 4 or len(set(cfg for _, cfg in populations[c])) <= 1:
                continue

            pop = populations[c]
            if len(pop) < 2:
                continue

            # --- Global Elite Archive Linkage Learning ---
            # Calculates Mutual Information using the top 50% of ALL valid configurations
            # found so far. Mathematically proves which parameters (like booleans or
            # block factorizations) must move together to achieve high performance.
            valid_history =[cfg for cfg, t in seen.items() if t != float('inf')]
            if len(valid_history) >= 8:
                valid_history.sort(key=lambda cfg: seen[cfg])
                elite_size = min(128, max(8, len(valid_history) // 2))
                learning_configs = valid_history[:elite_size]
            else:
                learning_configs =[cfg for _, cfg in pop]

            dynamic_fos = self._learn_linkage_tree(learning_configs)
            # ---------------------------------------------

            # --- Niching: Build the Hall of Fame ---
            # Instead of a single elitist, we find up to 4 diverse elites in this population.
            # An elite is considered diverse if it has a Hamming distance >= 2.
            pop_elites =[]
            for t, cfg in sorted(pop, key=lambda x: x[0]):
                is_diverse = True
                for e_t, e_cfg in pop_elites:
                    dist = sum(1 for a, b in zip(cfg, e_cfg) if a != b)
                    if dist < 2:
                        is_diverse = False
                        break
                if is_diverse:
                    pop_elites.append((t, cfg))
                if len(pop_elites) >= 4:
                    break

            # Sync local elites to the Global Hall of Fame for future populations to inherit
            for e_t, e_cfg in pop_elites:
                is_diverse = True
                for idx, (g_t, g_cfg) in enumerate(global_elites):
                    dist = sum(1 for a, b in zip(e_cfg, g_cfg) if a != b)
                    if dist < 2:
                        is_diverse = False
                        # Niche Climbing: If we found a faster config in the SAME niche, update the global elite!
                        if e_t < g_t:
                            global_elites[idx] = (e_t, e_cfg)
                        break
                if is_diverse:
                    global_elites.append((e_t, e_cfg))

            # Keep only the top 4 best diverse elites globally
            global_elites.sort(key=lambda x: x[0])
            global_elites = global_elites[:4]
            # ---------------------------------------

            new_pop =[]
            for i in range(len(pop)):
                if total_evals >= max_evaluations:
                    new_pop.append(pop[i])
                    continue

                ind_time, ind_cfg = pop[i]

                # 1. Gene-pool mixing
                new_cfg, new_time, improved = self._optimal_mix(
                    ind_cfg, ind_time, pop, i, evaluate,
                    layout_a, layout_b, rng, dynamic_fos)

                # 2. Strict Forced Improvement (FI) via Closest Elite
                # Find the elite in the Hall of Fame that is most similar to this individual.
                # This pulls the individual up its *current* hill instead of dragging it
                # across the landscape to a global optimum, preserving diverse niches!
                closest_elite = min(pop_elites, key=lambda e: sum(1 for a, b in zip(new_cfg, e[1]) if a != b))
                elite_time, elite_cfg = closest_elite

                if not improved and new_time > elite_time:
                    new_cfg, new_time, improved = self._optimal_mix_donor(
                        new_cfg, new_time, elite_cfg, evaluate,
                        layout_a, layout_b, rng, dynamic_fos)

                    # 3. Randomization if STILL no improvement
                    if not improved:
                        unseen =[cfg for cfg in layout_valid if cfg not in seen]
                        if unseen:
                            new_cfg = rng.choice(unseen)
                            new_time = evaluate(new_cfg)

                new_pop.append((new_time, new_cfg))

            populations[c] = new_pop

            # Update stagnation tracker
            current_best = min(new_pop, key=lambda x: x[0])[0]
            if current_best < pop_best_times[c]:
                pop_best_times[c] = current_best
                pop_stagnation[c] = 0
            else:
                pop_stagnation[c] += 1

        print(f"  GOMEA done (Evaluated {total_evals} configs)")

        # ==============================================================
        # Report
        # ==============================================================
        if best_config is None:
            print("  No valid configurations found!")
            return None

        d = self._config_to_dict(best_config)
        wmma_tile = 16
        bm = d['warps_m'] * d['warp_tile_m'] * wmma_tile
        bn = d['warps_n'] * d['warp_tile_n'] * wmma_tile
        lds = (bm * wmma_tile) + (wmma_tile * bn)
        mem = 2 * lds * 2

        unique = len(seen)
        coverage = (unique / len(layout_valid)) * 100
        print(f"\n  Best: {best_config} -> {best_time:.3f}ms")
        print(f"  Memory: {mem}/{self.max_shared_memory} bytes")
        print(f"  Evals: {unique} unique, {cache_hits} cache hits")
        print(f"  Coverage: {coverage:.1f}%")

        return {
            'config': self._config_to_dict(best_config),
            'raw_config': best_config,
            'time_ms': float(best_time),
            'evaluations': total_evals,
            'memory_used_bytes': mem,
            'space_coverage_percent': coverage,
        }

    def tune_all(self, sizes=None, layouts=None, max_evaluations=300,
                 existing_configs=None, overwrite=False):
        """Iterates over all requested matrix sizes and layouts to tune them."""
        if sizes is None:
            sizes =[
                (1024, 1024, 1024), (2048, 2048, 2048),
                (4096, 4096, 4096), (8192, 8192, 8192)]
        if layouts is None:
            layouts =[
                (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

        results = {}
        for M, N, K in sizes:
            sk = f"{M}x{N}x{K}"
            results[sk] = {}

            # Sort layouts so row-major C (lc=0) is always evaluated before col-major C (lc=1)
            sorted_layouts = sorted(layouts, key=lambda x: (x[0], x[1], x[2]))

            # Dictionary to cache the best config for a given (la, lb) pair
            shared_baselines = {}

            for la, lb, lc in sorted_layouts:
                lk = f"{la}_{lb}_{lc}"
                baseline = None if overwrite else (existing_configs or {}).get(sk, {}).get(lk)

                # --- Layout Sharing & Competitive Baselines ---
                current_budget = max_evaluations
                if lc == 1 and (la, lb) in shared_baselines:
                    shared_cfg = shared_baselines[(la, lb)]

                    if baseline is None:
                        baseline = shared_cfg
                        current_budget = max(10, max_evaluations // 2)
                        print(f"\n  [Optimization] Inherited row-major config for col-major C. Budget halved to {current_budget}.")
                    elif baseline == shared_cfg:
                        # FIX: If the input file config is exactly the same as the shared config, skip the race!
                        current_budget = max(10, max_evaluations // 2)
                        print(f"\n  [Optimization] Input baseline matches shared row-major config perfectly! Budget halved to {current_budget}.")
                    else:
                        print(f"\n  [Optimization] Racing input baseline vs shared row-major config for C=1...")
                        t_input = self._evaluate_config(M, N, K, la, lb, lc, baseline)
                        t_shared = self._evaluate_config(M, N, K, la, lb, lc, shared_cfg)

                        if t_shared <= t_input:
                            baseline = shared_cfg
                            current_budget = max(10, max_evaluations // 2)
                            print(f"  [Optimization] Shared config won ({t_shared:.3f}ms < {t_input:.3f}ms). Budget halved to {current_budget}.")
                        else:
                            print(f"  [Optimization] Input baseline won ({t_input:.3f}ms <= {t_shared:.3f}ms). Keeping full budget.")
                # ----------------------------------------------

                result = self.tune_layout(
                    M, N, K, la, lb, lc, current_budget, baseline)

                if result:
                    # Cache the row-major result to share with the col-major run
                    if lc == 0:
                        shared_baselines[(la, lb)] = result['raw_config']

                    results[sk][lk] = {
                        "M": M, "N": N, "K": K,
                        "layout": {
                            "A": "row_major" if la == 0 else "col_major",
                            "B": "row_major" if lb == 0 else "col_major",
                            "C": "row_major" if lc == 0 else "col_major"},
                        "config": result['config'],
                        "avg_time_ms": result['time_ms'],
                        "evaluations": result['evaluations'],
                        "memory_used_bytes": result['memory_used_bytes'],
                        "space_coverage_percent": result['space_coverage_percent']
                    }
        return results


# ======================================================================
# I/O helpers
# ======================================================================

def load_existing_json(path):
    if not Path(path).exists():
        print(f"Warning: '{path}' not found.")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        out = {}
        for e in data.get('configurations', []):
            sk = f"{e['range']['M']}x{e['range']['N']}x{e['range']['K']}"
            l = e['layout']
            lk = (f"{'0' if l['A']=='row_major' else '1'}_"
                  f"{'0' if l['B']=='row_major' else '1'}_"
                  f"{'0' if l['C']=='row_major' else '1'}")
            c = e['config']

            out.setdefault(sk, {})[lk] = tuple(
                c[p] for p in[
                    'warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n',
                    'swizzle', 'bits', 'buffer_first', 'use_async', 'use_direct_write'])
        print(f"Loaded {len(data.get('configurations',[]))} configs from '{path}'")
        return out
    except Exception as e:
        print(f"Error loading '{path}': {e}")
        return None


def merge_results(existing, new):
    if existing is None:
        return new
    merged = {}
    for sk, ld in existing.items():
        merged[sk] = {}
        for lk, bl in ld.items():
            M, N, K = map(int, sk.split('x'))
            la, lb, lc = map(int, lk.split('_'))
            pnames =['warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n',
                       'swizzle', 'bits', 'buffer_first', 'use_async', 'use_direct_write']
            merged[sk][lk] = {
                "M": M, "N": N, "K": K,
                "layout": {
                    "A": "row_major" if la == 0 else "col_major",
                    "B": "row_major" if lb == 0 else "col_major",
                    "C": "row_major" if lc == 0 else "col_major"},
                "config": {p: (int(bl[i]) if isinstance(bl[i], (int, float))
                               and not isinstance(bl[i], bool)
                               else bool(bl[i]) if isinstance(bl[i], bool)
                               else bl[i])
                           for i, p in enumerate(pnames)},
                "avg_time_ms": None, "evaluations": 0,
                "memory_used_bytes": None, "space_coverage_percent": 0}
    for sk, ld in new.items():
        merged.setdefault(sk, {}).update(ld)
    return merged


def parse_matrix_sizes(ss):
    sizes =[]
    for s in ss:
        parts = s.split(',')
        if len(parts) != 3:
            print(f"Error: expected M,N,K, got '{s}'")
            sys.exit(1)
        sizes.append(tuple(map(int, parts)))
    return sizes


def parse_layouts(ss):
    lm = {'row_major': 0, 'col_major': 1, 'r': 0, 'c': 1}
    layouts =[]
    for s in ss:
        parts = s.split(',')
        if len(parts) != 3:
            print(f"Error: expected A,B,C, got '{s}'")
            sys.exit(1)
        layouts.append(tuple(lm[p.strip().lower()] for p in parts))
    return layouts


def main():
    p = argparse.ArgumentParser(
        description='WMMA kernel tuner (Parameter-less GOMEA)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tune.py
  python tune.py --input gemm_config.json --layouts r,r,r c,c,c
  python tune.py --input gemm_config.json --sizes 4096,4096,4096
  python tune.py --budget 200
  python tune.py --seed 123
  python tune.py --gpu-arch gfx1103
        """)
    p.add_argument('--input', '-i', help='Input JSON with existing configs')
    p.add_argument('--sizes', nargs='*', help='Matrix sizes as M,N,K')
    p.add_argument('--layouts', nargs='*', help='Layouts as A,B,C')
    p.add_argument('--budget', type=int, default=300,
                   help='Eval budget per layout (default: 300)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu-arch', default='gfx1100')
    p.add_argument('--max-memory', type=int, default=65336)
    p.add_argument('--output', default='gemm_config_tuned.json')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing configs without using them as baselines')
    args = p.parse_args()

    existing = load_existing_json(args.input) if args.input else None

    sizes = parse_matrix_sizes(args.sizes) if args.sizes else[
        (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
        (8192, 8192, 8192), (4096, 4096, 1024), (8192, 8192, 1024),
        (4096, 2048, 64), (8192, 4096, 128), (4096, 16384, 4096),
        (4096, 4096, 16384), (2048, 5120, 5120), (4096, 5120, 5120),
        (32768, 4096, 4096), (65536, 2048, 2048)]

    layouts = parse_layouts(args.layouts) if args.layouts else[
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    print("WMMA Tuner (Parameter-less GOMEA)")
    print(f"  Seed: {args.seed}  GPU: {args.gpu_arch}  "
          f"Budget: {args.budget}  Memory: {args.max_memory}")
    print(f"  Sizes: {len(sizes)}  Layouts: {len(layouts)}")

    tuner = WMMATuner(args.max_memory, args.gpu_arch, args.seed)
    results = tuner.tune_all(sizes, layouts, args.budget, existing, args.overwrite)
    merged = merge_results(existing, results)

    configs =[]
    for sr in merged.values():
        for r in sr.values():
            configs.append({
                "range": {"M": r["M"], "N": r["N"], "K": r["K"]},
                "layout": r["layout"], "config": r["config"]})

    with open(args.output, "w") as f:
        json.dump({"configurations": configs}, f, indent=4)

    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    total_evals = 0
    for sk, sr in results.items():
        print(f"\n{sk}:")
        for lk, r in sr.items():
            c = r['config']
            print(f"  {lk}: {c['warps_m']},{c['warps_n']},"
                  f"{c['warp_tile_m']},{c['warp_tile_n']},"
                  f"{c['swizzle']},{c['bits']},{int(c['buffer_first'])},"
                  f"{int(c['use_async'])},{int(c['use_direct_write'])}"
                  f" -> {r['avg_time_ms']:.3f}ms "
                  f"({r['evaluations']} evals)")
            total_evals += r['evaluations']
    n = sum(len(d) for d in results.values()) if results else 0
    if n:
        print(f"\nTotal evals: {total_evals}  Avg: {total_evals // n}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
