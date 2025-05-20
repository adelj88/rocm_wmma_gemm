#!/usr/bin/env python3

import subprocess
import json

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

            # Parse results
            for line in result.stdout.splitlines():
                if not line.startswith(str(M)):  # Skip non-data lines
                    continue

                vals = line.strip().split(',')
                if len(vals) != 12:
                    continue

                config = {
                    "warps_m": int(vals[3]),
                    "warps_n": int(vals[4]),
                    "warp_tile_m": int(vals[5]),
                    "warp_tile_n": int(vals[6])
                }

                layouts = {
                    "A": vals[7],
                    "B": vals[8],
                    "C": vals[9]
                }

                layout_key = f"{vals[7]}_{vals[8]}_{vals[9]}"
                avg_time_ms = float(vals[10])

                # Store only if this is the best configuration for this size and layout
                if layout_key not in results[size_key] or avg_time_ms < results[size_key][layout_key]["avg_time_ms"]:
                    results[size_key][layout_key] = {
                        "M": M,
                        "N": N,
                        "K": K,
                        "config": config,
                        "layout": layouts,
                        "avg_time_ms": avg_time_ms
                    }
                    print(f"New best for {size_key} {layout_key}: {config} ({avg_time_ms:.3f} ms)")

        except subprocess.CalledProcessError as e:
            print(f"Error testing size {size_key}:")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")

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

def main():
    print("Running tuning...")
    results = run_tuning()

    print("\nGenerating configuration...")
    config = generate_json_config(results)

    # Print summary
    print("\nBest configurations:")
    for cfg in config["configurations"]:
        print(f"M={cfg['range']['M']}, N={cfg['range']['N']}, K={cfg['range']['K']}")
        print(f"Layout: {cfg['layout']}")
        print(f"Config: {cfg['config']}\n")

    with open("gemm_config_tuned.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Configuration saved to gemm_config_tuned.json")

if __name__ == "__main__":
    main()
