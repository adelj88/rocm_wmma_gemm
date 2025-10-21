#!/usr/bin/env python3

import re
import argparse
import sys
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class BenchmarkParser:
    """Parse Google Benchmark output and extract performance metrics."""

    def __init__(self, debug=False):
        self.results = []
        self.debug = debug
        # Regex pattern to match ANSI escape codes
        self.ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    def strip_ansi(self, text):
        """Remove ANSI escape codes from text."""
        return self.ansi_escape.sub('', text)

    def parse_line(self, line):
        """Parse a single benchmark result line."""
        # Strip ANSI escape codes first
        line = self.strip_ansi(line)

        # Example line:
        # {A:m_layout::col_major,B:m_layout::col_major,C:m_layout::col_major,m:1024,n:1024,k:1024}/manual_time      0.207 ms        0.278 ms         3713    11.5848      56.6933Gi/s     24.7665     0.16149

        # Extract benchmark name
        match = re.match(r'\{A:m_layout::(\w+),B:m_layout::(\w+),C:m_layout::(\w+),m:(\d+),n:(\d+),k:(\d+)\}', line)
        if not match:
            if self.debug and '{A:m_layout' in line:
                print(f"Warning: Line contains pattern but didn't match regex:")
                print(f"  {repr(line[:150])}")
            return None

        layout_a, layout_b, layout_c, m, n, k = match.groups()

        if self.debug:
            print(f"Matched benchmark line: {layout_a}/{layout_b}/{layout_c} {m}x{n}x{k}")

        # Extract avg_tflops - it's in the column before bytes_per_second (Gi/s)
        # Pattern: iterations(whitespace)avg_tflops(whitespace)bytes_per_second
        # The avg_tflops comes right before the Gi/s value
        # Need to handle variable whitespace between columns
        tflops_match = re.search(r'\s+(\d+(?:\.\d+)?)\s+[\d.]+Gi/s', line)
        if not tflops_match:
            if self.debug:
                print(f"Warning: Could not extract TFLOPS from line: {line}")
                # Try to show what's near Gi/s
                gis_pos = line.find('Gi/s')
                if gis_pos > 0:
                    print(f"  Near Gi/s: ...{line[max(0, gis_pos-50):gis_pos+10]}...")
            return None

        tflops = float(tflops_match.group(1))

        if self.debug:
            print(f"Parsed: {layout_a}/{layout_b}/{layout_c} {m}x{n}x{k} -> {tflops} TFLOPS")

        return {
            'layout_a': layout_a,
            'layout_b': layout_b,
            'layout_c': layout_c,
            'm': int(m),
            'n': int(n),
            'k': int(k),
            'tflops': tflops
        }

    def parse_output(self, output):
        """Parse benchmark output from string."""
        if self.debug:
            print(f"Parsing output ({len(output)} characters, {len(output.split(chr(10)))} lines)")

        line_count = 0
        for line in output.split('\n'):
            line_count += 1
            if self.debug and line_count <= 5:
                # Show both original and stripped versions
                stripped = self.strip_ansi(line)
                print(f"Sample line {line_count} (original): {line[:100]}")
                print(f"Sample line {line_count} (stripped): {stripped[:100]}")

            result = self.parse_line(line)
            if result:
                self.results.append(result)

        if self.debug:
            print(f"Total lines processed: {line_count}, results found: {len(self.results)}")

        return self.results

def run_benchmark(binary_path, shapes=None, batch_count=1, verbose=False):
    """
    Run a benchmark binary and return its output.

    Args:
        binary_path: Path to the benchmark binary
        shapes: Colon-separated string of matrix shapes (e.g., "1024:2048:1024,2048,512")
        batch_count: Batch count for benchmarks
        verbose: Print benchmark output to console

    Returns:
        Benchmark output as string
    """
    if not Path(binary_path).exists():
        raise FileNotFoundError(f"Benchmark binary not found: {binary_path}")

    # Build command
    cmd = [str(binary_path)]

    if shapes:
        cmd.extend(["--shapes", shapes])

    if batch_count != 1:
        cmd.extend(["--batch_count", str(batch_count)])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300  # 5 minute timeout
        )

        # Check return code manually
        if result.returncode != 0:
            print(f"Warning: Benchmark returned non-zero exit code: {result.returncode}")
            print(f"stderr: {result.stderr}")
            # Don't exit, try to parse output anyway

        output = result.stdout

        # Check if output is in stderr instead
        if not output.strip() and result.stderr.strip():
            if verbose:
                print("Note: Output found in stderr, using that instead")
            output = result.stderr

        if verbose:
            print(output)

        if not output.strip():
            print("Warning: Benchmark produced no output!")
            print(f"stderr: {result.stderr}")
            print(f"returncode: {result.returncode}")

        return output

    except subprocess.TimeoutExpired:
        print(f"Error: Benchmark timed out after 300 seconds")
        sys.exit(1)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)

def format_matrix_size(m, n, k):
    """Format matrix size string (e.g., 1024x1024x1024 or 4096x2048x64)."""
    return f"{m}x{n}x{k}"

def generate_markdown_table(wmma_results, rocblas_results, output_file, title, gpu_name, os_info, rocm_version):
    """Generate a markdown table comparing rocm_wmma_gemm and rocBLAS performance."""

    # Organize results by size and layout
    wmma_data = defaultdict(lambda: defaultdict(dict))
    rocblas_data = defaultdict(lambda: defaultdict(dict))

    for result in wmma_results:
        size_key = (result['m'], result['n'], result['k'])
        layout_key = (result['layout_a'], result['layout_b'])
        layout_c = result['layout_c']
        wmma_data[size_key][layout_key][layout_c] = result['tflops']

    for result in rocblas_results:
        size_key = (result['m'], result['n'], result['k'])
        layout_key = (result['layout_a'], result['layout_b'])
        layout_c = result['layout_c']
        rocblas_data[size_key][layout_key][layout_c] = result['tflops']

    # Generate markdown
    lines = []
    lines.append(f"# {title}\n")

    # Build system info string
    sys_parts = []
    if gpu_name:
        sys_parts.append(gpu_name)
    if os_info:
        sys_parts.append(os_info)
    if rocm_version:
        sys_parts.append(f"ROCm {rocm_version}")

    if sys_parts:
        lines.append(f"Performance measured on {', '.join(sys_parts)} in TFLOPs.\n")
    else:
        lines.append("Performance measured in TFLOPs.\n")

    lines.append("## FP16-FP16 Performance Results\n")

    # Table header
    lines.append("| Matrix Size    | Input Layout (A,B) | `rocm_wmma_gemm`<br>(C=col) | `rocBLAS`<br>(C=col) | Ratio<br>(C=col / rocBLAS) | `rocm_wmma_gemm`<br>(C=row) | Ratio<br>(C=row / rocBLAS) |")
    lines.append("|:---------------|:-------------------|---------------------------:|--------------------:|--------------------------:|---------------------------:|--------------------------:|")

    # Sort sizes
    sizes = sorted(wmma_data.keys())

    for size in sizes:
        m, n, k = size
        size_str = format_matrix_size(m, n, k)

        # Define layout order
        layouts = [
            ('col_major', 'col_major'),
            ('row_major', 'col_major'),
            ('col_major', 'row_major'),
            ('row_major', 'row_major')
        ]

        for layout_a, layout_b in layouts:
            layout_key = (layout_a, layout_b)
            layout_str = f"{layout_a[:3]}, {layout_b[:3]}"

            # Get performance values
            wmma_col = wmma_data[size].get(layout_key, {}).get('col_major', 0)
            wmma_row = wmma_data[size].get(layout_key, {}).get('row_major', 0)
            rocblas_col = rocblas_data[size].get(layout_key, {}).get('col_major', 0)

            # Calculate ratios
            ratio_col = wmma_col / rocblas_col if rocblas_col > 0 else 0
            ratio_row = wmma_row / rocblas_col if rocblas_col > 0 else 0

            lines.append(f"| {size_str} | {layout_str} | {wmma_col:>26.2f} | {rocblas_col:>19.2f} | {ratio_col:>25.2f} | {wmma_row:>26.2f} | {ratio_row:>25.2f} |")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Markdown table written to {output_file}")

def generate_performance_plots(wmma_results, rocblas_results, output_file, title, gpu_name):
    """Generate performance comparison plots."""

    # Organize results
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for result in wmma_results:
        size = (result['m'], result['n'], result['k'])
        layout_ab = (result['layout_a'], result['layout_b'])
        layout_c = result['layout_c']
        data[layout_ab][size]['wmma'][layout_c] = result['tflops']

    for result in rocblas_results:
        size = (result['m'], result['n'], result['k'])
        layout_ab = (result['layout_a'], result['layout_b'])
        layout_c = result['layout_c']
        data[layout_ab][size]['rocblas'][layout_c] = result['tflops']

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Build title
    main_title = f'ROCm WMMA GEMM FP16 Performance by Layout Configuration'
    if gpu_name:
        subtitle = f'({gpu_name} Performance in TFLOPs)'
    else:
        subtitle = '(Performance in TFLOPs)'

    fig.suptitle(f'{main_title}\n{subtitle}',
                 fontsize=14, fontweight='bold')

    # Layout configurations
    layouts = [
        (('col_major', 'col_major'), 'Column-Major A, Column-Major B'),
        (('row_major', 'col_major'), 'Row-Major A, Column-Major B'),
        (('col_major', 'row_major'), 'Column-Major A, Row-Major B'),
        (('row_major', 'row_major'), 'Row-Major A, Row-Major B')
    ]

    sizes = sorted(set((r['m'], r['n'], r['k']) for r in wmma_results))

    for idx, (layout_key, layout_title) in enumerate(layouts):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(layout_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Matrix Size', fontsize=10)
        ax.set_ylabel('TFLOPs', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Extract data for this layout
        rocblas_perf = []
        wmma_col_perf = []
        wmma_row_perf = []

        for size in sizes:
            rocblas_val = data[layout_key][size]['rocblas'].get('col_major')
            wmma_col_val = data[layout_key][size]['wmma'].get('col_major')
            wmma_row_val = data[layout_key][size]['wmma'].get('row_major')

            rocblas_perf.append(rocblas_val)
            wmma_col_perf.append(wmma_col_val)
            wmma_row_perf.append(wmma_row_val)

        # Plot lines
        x_pos = np.arange(len(sizes))

        if any(v is not None for v in rocblas_perf):
            ax.plot(x_pos, rocblas_perf, 'o-', color='#2ecc71', linewidth=2,
                   markersize=8, label='rocBLAS')

        if any(v is not None for v in wmma_col_perf):
            ax.plot(x_pos, wmma_col_perf, 's-', color='#3498db', linewidth=2,
                   markersize=8, label='rocm_wmma_gemm (col out)')

        if any(v is not None for v in wmma_row_perf):
            ax.plot(x_pos, wmma_row_perf, '^-', color='#e67e22', linewidth=2,
                   markersize=8, label='rocm_wmma_gemm (row out)')

        # Add improvement annotations for significant cases
        for i, size in enumerate(sizes):
            rocblas_val = rocblas_perf[i]
            wmma_row_val = wmma_row_perf[i]

            if rocblas_val and wmma_row_val and wmma_row_val > rocblas_val * 1.15:
                improvement = ((wmma_row_val / rocblas_val - 1) * 100)
                ax.annotate(f'+{improvement:.0f}%',
                           xy=(i, wmma_row_val),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           color='#e67e22',
                           fontweight='bold')

        # Format x-axis labels
        size_labels = []
        for m, n, k in sizes:
            if m == n == k:
                size_labels.append(str(m))
            else:
                size_labels.append(f"{m}×{n}×{k}")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(size_labels, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Performance plots saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate WMMA GEMM benchmark report by running binaries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmarks with default shapes
  python generate_report.py \\
      --wmma-bin ./build/bench/bench_half_half \\
      --rocblas-bin ./build/bench/bench_rocblas_half \\
      --gpu "AMD Radeon RX 7900 GRE" \\
      --os "Ubuntu 24.04.1 LTS" \\
      --rocm-version "6.4.1"

  # Custom square matrix shapes (colon-separated)
  python generate_report.py \\
      --wmma-bin ./build/bench/bench_half_half \\
      --rocblas-bin ./build/bench/bench_rocblas_half \\
      --shapes "512:1024:2048:4096" \\
      --gpu "AMD Radeon RX 7900 GRE"

  # Mix of square and rectangular matrices (colon-separated)
  python generate_report.py \\
      --wmma-bin ./build/bench/bench_half_half \\
      --rocblas-bin ./build/bench/bench_rocblas_half \\
      --shapes "1024:2048:1024,2048,512:2048,1024,1024:4096,4096,2048" \\
      --title "Mixed Matrix FP16 Performance" \\
      --gpu "AMD Radeon RX 7900 GRE"

  # Save benchmark outputs for later analysis
  python generate_report.py \\
      --wmma-bin ./build/bench/bench_half_half \\
      --rocblas-bin ./build/bench/bench_rocblas_half \\
      --shapes "1024:2048:4096:8192" \\
      --save-outputs \\
      --gpu "AMD Radeon RX 7900 GRE"

  # Batch count and verbose output
  python generate_report.py \\
      --wmma-bin ./build/bench/bench_half_half \\
      --rocblas-bin ./build/bench/bench_rocblas_half \\
      --batch-count 4 \\
      --verbose \\
      --gpu "AMD Radeon RX 7900 GRE"
        """)

    # Input binaries
    parser.add_argument('--wmma-bin', required=True,
                       help='Path to rocm_wmma_gemm benchmark binary')
    parser.add_argument('--rocblas-bin', required=True,
                       help='Path to rocBLAS benchmark binary')

    # Benchmark configuration
    parser.add_argument('--shapes',
                       help='Colon-separated list of matrix shapes (e.g., "1024:2048" or "1024:2048:1024,2048,512:4096,4096,2048")')
    parser.add_argument('--batch-count', type=int, default=1,
                       help='Batch count for benchmarks (default: 1)')

    # Report configuration
    parser.add_argument('--title', default='FP16-FP16 WMMA GEMM Performance Benchmarks',
                       help='Report title (default: FP16-FP16 WMMA GEMM Performance Benchmarks)')
    parser.add_argument('--gpu', required=True,
                       help='GPU name (e.g., "AMD Radeon RX 7900 GRE")')
    parser.add_argument('--os',
                       help='Operating system (e.g., "Ubuntu 24.04.1 LTS")')
    parser.add_argument('--rocm-version',
                       help='ROCm version (e.g., "6.4.1")')

    # Output configuration
    parser.add_argument('--markdown-output', default='wmma_performance.md',
                       help='Output markdown file (default: wmma_performance.md)')
    parser.add_argument('--plot-output', default='wmma_performance_plot.png',
                       help='Output plot file (default: wmma_performance_plot.png)')
    parser.add_argument('--save-outputs', action='store_true',
                       help='Save raw benchmark outputs to files')
    parser.add_argument('--wmma-output-file', default='bench_wmma.txt',
                       help='File to save rocm_wmma_gemm output (default: bench_wmma.txt)')
    parser.add_argument('--rocblas-output-file', default='bench_rocblas.txt',
                       help='File to save rocBLAS output (default: bench_rocblas.txt)')

    parser.add_argument('--no-markdown', action='store_true',
                       help='Skip markdown generation')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--verbose', action='store_true',
                       help='Print benchmark output to console')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information for parsing')

    args = parser.parse_args()

    # Run benchmarks
    print("\n" + "="*80)
    print("Running rocm_wmma_gemm benchmark...")
    print("="*80)
    wmma_output = run_benchmark(
        args.wmma_bin,
        shapes=args.shapes,
        batch_count=args.batch_count,
        verbose=args.verbose
    )

    if args.save_outputs:
        with open(args.wmma_output_file, 'w') as f:
            f.write(wmma_output)
        print(f"Saved rocm_wmma_gemm output to {args.wmma_output_file}")

    print("\n" + "="*80)
    print("Running rocBLAS benchmark...")
    print("="*80)
    rocblas_output = run_benchmark(
        args.rocblas_bin,
        shapes=args.shapes,
        batch_count=args.batch_count,
        verbose=args.verbose
    )

    if args.save_outputs:
        with open(args.rocblas_output_file, 'w') as f:
            f.write(rocblas_output)
        print(f"Saved rocBLAS output to {args.rocblas_output_file}")

    # Parse outputs
    print("\nParsing rocm_wmma_gemm benchmark results...")
    wmma_parser = BenchmarkParser(debug=args.debug)
    wmma_results = wmma_parser.parse_output(wmma_output)

    print("Parsing rocBLAS benchmark results...")
    rocblas_parser = BenchmarkParser(debug=args.debug)
    rocblas_results = rocblas_parser.parse_output(rocblas_output)

    print(f"Found {len(wmma_results)} rocm_wmma_gemm benchmark results")
    print(f"Found {len(rocblas_results)} rocBLAS benchmark results")

    if not wmma_results or not rocblas_results:
        print("Error: No benchmark results found")
        sys.exit(1)

    # Generate outputs
    if not args.no_markdown:
        print("\nGenerating markdown table...")
        generate_markdown_table(wmma_results, rocblas_results, args.markdown_output,
                               args.title, args.gpu, args.os, args.rocm_version)

    if not args.no_plot:
        print("\nGenerating performance plots...")
        generate_performance_plots(wmma_results, rocblas_results, args.plot_output,
                                  args.title, args.gpu)

    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
