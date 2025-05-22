#!/usr/bin/env python3

import subprocess
import json
import re

# Matrix sizes to test (M, N, K)
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

def parse_benchmark_name(name):
    """Parse benchmark name to extract configuration parameters"""
    # Pattern: config_WM_WN_TM_TN_AB_C where:
    # - WM, WN, TM, TN are integers
    # - AB is layout for A and B (rr, rc, cr, cc for row/col combinations)
    # - C is layout for C (r or c)

    pattern = r'config_(\d+)_(\d+)_(\d+)_(\d+)_([rc][rc])_([rc])'
    match = re.match(pattern, name)

    if not match:
        return None

    warps_m = int(match.group(1))
    warps_n = int(match.group(2))
    warp_tile_m = int(match.group(3))
    warp_tile_n = int(match.group(4))
    layout_ab = match.group(5)
    layout_c = match.group(6)

    # Convert layout codes to names
    layout_map = {'r': 'row_major', 'c': 'col_major'}
    layout_a = layout_map[layout_ab[0]]
    layout_b = layout_map[layout_ab[1]]
    layout_c = layout_map[layout_c]

    return {
        'warps_m': warps_m,
        'warps_n': warps_n,
        'warp_tile_m': warp_tile_m,
        'warp_tile_n': warp_tile_n,
        'layout_a': layout_a,
        'layout_b': layout_b,
        'layout_c': layout_c
    }

def parse_benchmark_output(output):
    """Parse benchmark output in console format"""
    results = {}
    best_times = {}  # Track best time for each layout
    lines = output.split('\n')

    # Skip header lines until we find the dashed line before results
    for i, line in enumerate(lines):
        if line.startswith('-----'):
            # Results start after this line
            results_lines = lines[i+2:]  # Skip the header line
            break

    for line in results_lines:
        if not line.strip():  # Skip empty lines
            continue

        # Split line by variable whitespace
        parts = line.split()
        if len(parts) < 4:  # Skip invalid lines
            continue

        # Parse benchmark name and time
        name = parts[0]
        time_ms = float(parts[1])  # Already in ms

        config = parse_benchmark_name(name)
        if config:
            layout_key = f"{config['layout_a']}_{config['layout_b']}_{config['layout_c']}"

            # Only update if this is a better (faster) time
            if layout_key not in best_times or time_ms < best_times[layout_key]:
                best_times[layout_key] = time_ms
                results[layout_key] = {
                    "config": config,
                    "avg_time_ms": time_ms
                }

    return results

def run_tuning():
    """Run tuner and collect results"""
    results = {}

    for M, N, K in sizes:
        size_key = f"{M}x{N}x{K}"
        print(f"\nTesting size {size_key}")
        results[size_key] = {}

        try:
            result = subprocess.run(
                ["./rocm_wmma_gemm/tuner", str(M), str(N), str(K)],
                capture_output=True,
                text=True,
                check=True
            )

            benchmark_results = parse_benchmark_output(result.stdout)

            # Process results
            for layout_key, bench_result in benchmark_results.items():
                time_ms = bench_result["avg_time_ms"]
                results[size_key][layout_key] = {
                    "M": M,
                    "N": N,
                    "K": K,
                    "config": bench_result["config"],
                    "layout": {
                        "A": bench_result["config"]["layout_a"],
                        "B": bench_result["config"]["layout_b"],
                        "C": bench_result["config"]["layout_c"]
                    },
                    "avg_time_ms": time_ms
                }
                print(f"Result for {size_key} {layout_key}: "
                      f"{bench_result['config']['warps_m']},"
                      f"{bench_result['config']['warps_n']},"
                      f"{bench_result['config']['warp_tile_m']},"
                      f"{bench_result['config']['warp_tile_n']} "
                      f"({time_ms:.3f} ms)")

        except subprocess.CalledProcessError as e:
            print(f"Error testing size {size_key}:")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error for size {size_key}: {e}")

    return results

def generate_json_config(results):
    """Convert tuning results to gemm_config.json format"""
    configs = []

    # For each size and layout combination, add the best configuration
    for size_results in results.values():
        for result in size_results.values():
            config = {
                "range": {
                    "M": result["M"],
                    "N": result["N"],
                    "K": result["K"]
                },
                "layout": result["layout"],
                "config": result["config"]
            }
            configs.append(config)

    return {"configurations": configs}

def print_csv_summary(results):
    """Print results in CSV format similar to original tuner"""
    print("\n" + "="*80)
    print("SUMMARY (CSV FORMAT):")
    print("="*80)
    print("M,N,K,warps_m,warps_n,warp_tile_m,warp_tile_n,layout_a,layout_b,layout_c,avg_time_ms")

    for size_results in results.values():
        for result in size_results.values():
            print(f"{result['M']},{result['N']},{result['K']},"
                  f"{result['config']['warps_m']},{result['config']['warps_n']},"
                  f"{result['config']['warp_tile_m']},{result['config']['warp_tile_n']},"
                  f"{result['layout']['A']},{result['layout']['B']},{result['layout']['C']},"
                  f"{result['avg_time_ms']:.3f}")

def main():
    print("Running tuning with Google Benchmark...")
    results = run_tuning()

    if not results:
        print("No results collected!")
        return

    print("\nGenerating configuration...")
    config = generate_json_config(results)

    # Print summary
    print_csv_summary(results)

    print(f"\nBest configurations found: {len(config['configurations'])}")

    with open("gemm_config_tuned.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Configuration saved to gemm_config_tuned.json")

if __name__ == "__main__":
    main()
