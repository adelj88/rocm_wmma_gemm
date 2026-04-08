#!/usr/bin/env python3
"""WMMA config racing tool.

This script allows racing two configurations against each other to determine the best one.
It can run in two modes:
1. File vs File racing: Compares two config files and keeps the best configs.
2. Cross-Layout Sanity Check: For a given matrix size and A/B layout, it checks
   if the row-major C config is better than the col-major C config (and vice versa)
   by evaluating both configs on both layouts.
"""

import subprocess
import json
import re
import argparse
import sys
import statistics
from pathlib import Path

# ======================================================================
# Execution Helpers
# ======================================================================

def _parse_benchmark_output(output):
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

def evaluate_config_once(M, N, K, la, lb, lc, config, gpu_arch):
    """Compiles and runs the kernel configuration on the GPU ONE time, returning execution time."""
    pnames = ['warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n',
              'swizzle', 'bits', 'buffer_first', 'use_async', 'use_direct_write']

    # config could be a dict or a tuple. Handle both.
    if isinstance(config, dict):
        d = config
    else:
        d = {name: config[i] for i, name in enumerate(pnames)}

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
            str(la), str(lb), str(lc), gpu_arch
        ], capture_output=True, text=True, timeout=60, check=False)
        if result.returncode == 0:
            t = _parse_benchmark_output(result.stdout)
            if t is not None:
                return t
    except Exception:
        pass
    return float('inf')

def run_head_to_head(M, N, K, la, lb, lc, cfg1, cfg2, gpu_arch, rounds=5):
    """Races cfg1 and cfg2 interleaved to handle noise. Returns winner, scores, and medians."""
    wins1 = 0
    wins2 = 0
    t1_avg = []
    t2_avg = []

    for i in range(rounds):
        # Swap order each round to be completely fair regarding thermal throttling
        if i % 2 == 0:
            t1 = evaluate_config_once(M, N, K, la, lb, lc, cfg1, gpu_arch)
            t2 = evaluate_config_once(M, N, K, la, lb, lc, cfg2, gpu_arch)
        else:
            t2 = evaluate_config_once(M, N, K, la, lb, lc, cfg2, gpu_arch)
            t1 = evaluate_config_once(M, N, K, la, lb, lc, cfg1, gpu_arch)

        if t1 != float('inf'): t1_avg.append(t1)
        if t2 != float('inf'): t2_avg.append(t2)

        if t1 < t2:
            wins1 += 1
        elif t2 < t1:
            wins2 += 1
        else:
            wins1 += 0.5
            wins2 += 0.5

    m1 = statistics.median(t1_avg) if t1_avg else float('inf')
    m2 = statistics.median(t2_avg) if t2_avg else float('inf')

    winner = 1 if wins1 >= wins2 else 2
    return winner, wins1, wins2, m1, m2

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

            pnames = ['warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n',
                      'swizzle', 'bits', 'buffer_first', 'use_async', 'use_direct_write']

            # Ensure booleans are true booleans internally
            tuple_cfg = tuple(
                (int(c[p]) if isinstance(c[p], (int, float)) and not isinstance(c[p], bool)
                 else bool(c[p]) if isinstance(c[p], bool) else c[p])
                for p in pnames
            )
            out.setdefault(sk, {})[lk] = tuple_cfg
        return out
    except Exception as e:
        print(f"Error loading '{path}': {e}")
        return None

def tuple_to_dict(config_tuple):
    pnames = ['warps_m', 'warps_n', 'warp_tile_m', 'warp_tile_n',
              'swizzle', 'bits', 'buffer_first', 'use_async', 'use_direct_write']
    return {name: config_tuple[i] for i, name in enumerate(pnames)}

def save_json(merged_data, output_path):
    configs = []
    for sk, sr in merged_data.items():
        M, N, K = map(int, sk.split('x'))
        for lk, cfg_tuple in sr.items():
            la, lb, lc = map(int, lk.split('_'))
            configs.append({
                "range": {"M": M, "N": N, "K": K},
                "layout": {
                    "A": "row_major" if la == 0 else "col_major",
                    "B": "row_major" if lb == 0 else "col_major",
                    "C": "row_major" if lc == 0 else "col_major"
                },
                "config": tuple_to_dict(cfg_tuple)
            })

    with open(output_path, "w") as f:
        json.dump({"configurations": configs}, f, indent=4)
    print(f"\nSaved final configurations to {output_path}")

# ======================================================================
# Main Logic
# ======================================================================

def main():
    p = argparse.ArgumentParser(description='WMMA Config Racing Tool')
    p.add_argument('--config1', required=True, help='Path to the first config JSON file')
    p.add_argument('--config2', help='Path to the second config JSON file (optional)')
    p.add_argument('--cross-check', action='store_true', help='Race row-major C and col-major C configs against each other for the same A/B layouts')
    p.add_argument('--gpu-arch', default='gfx1100', help='GPU architecture (default: gfx1100)')
    p.add_argument('--repeats', type=int, default=5, help='Number of executions per config to find median (default: 5)')
    p.add_argument('--output', default='gemm_config_raced.json', help='Output JSON file (default: gemm_config_raced.json)')
    args = p.parse_args()

    data1 = load_existing_json(args.config1)
    if not data1:
        print("Failed to load config1. Exiting.")
        sys.exit(1)

    data2 = load_existing_json(args.config2) if args.config2 else None

    # We will build the merged dictionary representing the best configs
    final_merged = {}

    # Deep copy data1 into final_merged
    for sk, layouts in data1.items():
        final_merged[sk] = {}
        for lk, cfg in layouts.items():
            final_merged[sk][lk] = cfg

    # 1. File vs File Racing
    if data2:
        print(f"\n--- Racing {args.config1} vs {args.config2} ---")
        for sk, layouts2 in data2.items():
            if sk not in final_merged:
                final_merged[sk] = {}
                for lk, cfg in layouts2.items():
                    final_merged[sk][lk] = cfg
                continue

            M, N, K = map(int, sk.split('x'))
            for lk, cfg2 in layouts2.items():
                if lk not in final_merged[sk]:
                    final_merged[sk][lk] = cfg2
                    continue

                cfg1 = final_merged[sk][lk]
                if cfg1 == cfg2:
                    continue # They are identical

                la, lb, lc = map(int, lk.split('_'))

                print(f"\n[{sk} | A={la} B={lb} C={lc}] Configurations differ. Racing for {args.repeats} rounds...")
                winner, wins1, wins2, m1, m2 = run_head_to_head(M, N, K, la, lb, lc, cfg1, cfg2, args.gpu_arch, args.repeats)

                if m1 == float('inf') and m2 == float('inf'):
                    print("  Both configs failed!")
                    continue

                if winner == 1:
                    print(f"  Config 1 Won! (Score: {wins1}-{wins2}) Median: {m1:.3f}ms vs {m2:.3f}ms")
                    # final_merged already has cfg1
                else:
                    print(f"  Config 2 Won! (Score: {wins2}-{wins1}) Median: {m2:.3f}ms vs {m1:.3f}ms - Overwriting...")
                    final_merged[sk][lk] = cfg2

    # 2. Cross-Layout Sanity Check
    if args.cross_check:
        print("\n--- Cross-Layout Sanity Check (Row vs Col C) ---")
        for sk, layouts in list(final_merged.items()):
            M, N, K = map(int, sk.split('x'))

            # Find matching (la, lb) pairs where both lc=0 and lc=1 exist
            la_lb_pairs = set(f"{la}_{lb}" for lk in layouts for la, lb, lc in [map(int, lk.split('_'))])

            for la_lb in la_lb_pairs:
                lk0 = f"{la_lb}_0"
                lk1 = f"{la_lb}_1"

                if lk0 in layouts and lk1 in layouts:
                    cfg0 = final_merged[sk][lk0]
                    cfg1 = final_merged[sk][lk1]

                    if cfg0 == cfg1:
                        continue # Already the same

                    la, lb = map(int, la_lb.split('_'))

                    print(f"\n[{sk} | A={la} B={lb}] Configs differ between C=0 and C=1. Cross-racing...")

                    # Race on layout C=0
                    print(f"  Racing on layout C=0 (Row-Major Output):")
                    winner, wins0, wins1, m0, m1 = run_head_to_head(M, N, K, la, lb, 0, cfg0, cfg1, args.gpu_arch, args.repeats)

                    if winner == 1:
                        print(f"    Original Row-Major config won (Score: {wins0}-{wins1}) Median: {m0:.3f}ms vs {m1:.3f}ms")
                    else:
                        print(f"    Col-Major config won on Row-Major layout! (Score: {wins1}-{wins0}) Median: {m1:.3f}ms vs {m0:.3f}ms - Updating C=0")
                        final_merged[sk][lk0] = cfg1

                    # Race on layout C=1
                    print(f"  Racing on layout C=1 (Col-Major Output):")
                    winner, wins0, wins1, m0, m1 = run_head_to_head(M, N, K, la, lb, 1, cfg0, cfg1, args.gpu_arch, args.repeats)

                    if winner == 2:
                        print(f"    Original Col-Major config won (Score: {wins1}-{wins0}) Median: {m1:.3f}ms vs {m0:.3f}ms")
                    else:
                        print(f"    Row-Major config won on Col-Major layout! (Score: {wins0}-{wins1}) Median: {m0:.3f}ms vs {m1:.3f}ms - Updating C=1")
                        final_merged[sk][lk1] = cfg0

    save_json(final_merged, args.output)

if __name__ == "__main__":
    main()
