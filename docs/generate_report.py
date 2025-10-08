#!/usr/bin/env python3

import re
import argparse
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class BenchmarkParser:
    """Parse Google Benchmark output and extract performance metrics."""

    def __init__(self):
        self.results = []

    def parse_line(self, line):
        """Parse a single benchmark result line."""
        # Example line:
        # {A:m_layout::col_major,B:m_layout::col_major,C:m_layout::col_major,m:1024,n:1024,k:1024}/manual_time      0.131 ms        0.204 ms         5481    16.8283      44.7156Gi/s     3.78287    0.107316

        # Extract benchmark name
        match = re.match(r'\{A:m_layout::(\w+),B:m_layout::(\w+),C:m_layout::(\w+),m:(\d+),n:(\d+),k:(\d+)\}', line)
        if not match:
            return None

        layout_a, layout_b, layout_c, m, n, k = match.groups()

        # Extract avg_tflops
        tflops_match = re.search(r'(\d+\.\d+)\s+[\d.]+Gi/s', line)
        if not tflops_match:
            return None

        tflops = float(tflops_match.group(1))

        return {
            'layout_a': layout_a,
            'layout_b': layout_b,
            'layout_c': layout_c,
            'm': int(m),
            'n': int(n),
            'k': int(k),
            'tflops': tflops
        }

    def parse_file(self, filename):
        """Parse a benchmark output file."""
        with open(filename, 'r') as f:
            for line in f:
                result = self.parse_line(line)
                if result:
                    self.results.append(result)
        return self.results

def format_matrix_size(m, n, k):
    """Format matrix size string (e.g., 1024x1024x1024 or 4096x2048x64)."""
    return f"{m}x{n}x{k}"

def generate_markdown_table(wmma_results, rocblas_results, output_file, title, gpu_name, os_info, rocm_version):
    """Generate a markdown table comparing WMMA and rocBLAS performance."""

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
            rocblas_val = data[layout_key][size]['rocblas'].get('col_major', None)
            wmma_col_val = data[layout_key][size]['wmma'].get('col_major', None)
            wmma_row_val = data[layout_key][size]['wmma'].get('row_major', None)

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
                   markersize=8, label='WMMA (col out)')

        if any(v is not None for v in wmma_row_perf):
            ax.plot(x_pos, wmma_row_perf, '^-', color='#e67e22', linewidth=2,
                   markersize=8, label='WMMA (row out)')

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
        description='Generate benchmark report from Google Benchmark output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_report.py --wmma bench_half_half.txt --rocblas bench_rocblas.txt \\
      --gpu "AMD Radeon RX 7900 GRE" \\
      --os "Ubuntu 24.04.1 LTS" \\
      --rocm-version "6.4.1"

  # Square matrices
  python generate_report.py --wmma square_wmma.txt --rocblas square_rocblas.txt \\
      --title "Square Matrix Performance Benchmarks" \\
      --gpu "AMD Radeon RX 7900 GRE" \\
      --os "Ubuntu 24.04.1 LTS" \\
      --rocm-version "6.4.1" \\
      --markdown-output square.md \\
      --plot-output square_plot.png

  # Rectangular matrices
  python generate_report.py --wmma rect_wmma.txt --rocblas rect_rocblas.txt \\
      --title "Rectangular Matrix Performance Benchmarks" \\
      --gpu "AMD Radeon RX 7900 GRE" \\
      --os "Ubuntu 24.04.1 LTS" \\
      --rocm-version "6.4.1" \\
      --markdown-output rectangle.md \\
      --plot-output rectangle_plot.png

  # Minimal info (only GPU)
  python generate_report.py --wmma bench.txt --rocblas rocblas.txt \\
      --gpu "AMD Radeon RX 6800 XT"

  # Generate only markdown
  python generate_report.py --wmma bench.txt --rocblas rocblas.txt \\
      --gpu "AMD Radeon RX 7900 GRE" \\
      --no-plot
        """)

    parser.add_argument('--wmma', required=True,
                       help='WMMA benchmark output file')
    parser.add_argument('--rocblas', required=True,
                       help='rocBLAS benchmark output file')
    parser.add_argument('--title', default='Matrix Performance Benchmarks',
                       help='Report title (default: Matrix Performance Benchmarks)')
    parser.add_argument('--gpu', required=True,
                       help='GPU name (e.g., "AMD Radeon RX 7900 GRE")')
    parser.add_argument('--os',
                       help='Operating system (e.g., "Ubuntu 24.04.1 LTS")')
    parser.add_argument('--rocm-version',
                       help='ROCm version (e.g., "6.4.1")')
    parser.add_argument('--markdown-output', default='performance.md',
                       help='Output markdown file (default: performance.md)')
    parser.add_argument('--plot-output', default='performance_plot.png',
                       help='Output plot file (default: performance_plot.png)')
    parser.add_argument('--no-markdown', action='store_true',
                       help='Skip markdown generation')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plot generation')

    args = parser.parse_args()

    # Parse benchmark files
    print("Parsing WMMA benchmark results...")
    wmma_parser = BenchmarkParser()
    wmma_results = wmma_parser.parse_file(args.wmma)
    print(f"Found {len(wmma_results)} WMMA benchmark results")

    print("Parsing rocBLAS benchmark results...")
    rocblas_parser = BenchmarkParser()
    rocblas_results = rocblas_parser.parse_file(args.rocblas)
    print(f"Found {len(rocblas_results)} rocBLAS benchmark results")

    if not wmma_results or not rocblas_results:
        print("Error: No benchmark results found in input files")
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

    print("\nReport generation complete!")

if __name__ == "__main__":
    main()
