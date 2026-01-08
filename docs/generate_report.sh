#!/usr/bin/env bash
set -euo pipefail

# Temporary directory for benchmark outputs
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Default values
SHAPES=""
BATCH_COUNT=1
TITLE="FP16-FP16 WMMA GEMM Performance Benchmarks"
MARKDOWN_OUTPUT="wmma_performance.md"
PLOT_OUTPUT="wmma_performance_plot.png"
NO_MARKDOWN=0
NO_PLOT=0
VERBOSE=0
DEBUG=0

usage() {
    cat << EOF
Usage: $0 --wmma-bin PATH --rocblas-bin PATH --gpu NAME [OPTIONS]

Required:
  --wmma-bin PATH          Path to rocm_wmma_gemm benchmark binary
  --rocblas-bin PATH       Path to rocBLAS benchmark binary
  --gpu NAME               GPU name (e.g., "AMD Radeon 8060S")

Optional:
  --shapes SHAPES          Colon-separated matrix shapes (e.g., "1024:2048:4096")
  --batch-count N          Batch count for benchmarks (default: 1)
  --title TITLE            Report title
  --os NAME                Operating system (e.g., "Ubuntu 24.04.3 LTS")
  --rocm-version VERSION   ROCm version (e.g., "7.1.1")
  --markdown-output FILE   Output markdown file (default: wmma_performance.md)
  --plot-output FILE       Output plot file (default: wmma_performance_plot.png)
  --no-markdown            Skip markdown generation
  --no-plot                Skip plot generation
  --verbose                Print benchmark output to console
  --debug                  Print debug information for parsing

Examples:
  $0 --wmma-bin ../build/benchmark/bench_half_half \\
     --rocblas-bin ../build/benchmark/bench_rocblas \\
     --gpu "AMD Radeon 8060S" \\
     --os "Ubuntu 24.04.3 LTS" \\
     --rocm-version "7.1.1" \\
     --title "Square Matrix FP16 Performance Benchmarks" \\
     --markdown-output gfx1151_square.md \\
     --plot-output gfx1151_square.png

  $0 --wmma-bin ../build/benchmark/bench_half_half \\
     --rocblas-bin ../build/benchmark/bench_rocblas \\
     --gpu "AMD Radeon 8060S" \\
     --os "Ubuntu 24.04.3 LTS" \\
     --rocm-version "7.1.1" \\
     --title "Rectangle Matrix FP16 Performance Benchmarks" \\
     --markdown-output gfx1151_rectangle.md \\
     --plot-output gfx1151_rectangle.png \\
     --shapes "4096,4096,1024:8192,8192,1024:4096,2048,64:8192,4096,128"
EOF
    exit 1
}

# Parse command line arguments
WMMA_BIN=""
ROCBLAS_BIN=""
GPU=""
OS=""
ROCM_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --wmma-bin)
            WMMA_BIN="$2"
            shift 2
            ;;
        --rocblas-bin)
            ROCBLAS_BIN="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --os)
            OS="$2"
            shift 2
            ;;
        --rocm-version)
            ROCM_VERSION="$2"
            shift 2
            ;;
        --shapes)
            SHAPES="$2"
            shift 2
            ;;
        --batch-count)
            BATCH_COUNT="$2"
            shift 2
            ;;
        --title)
            TITLE="$2"
            shift 2
            ;;
        --markdown-output)
            MARKDOWN_OUTPUT="$2"
            shift 2
            ;;
        --plot-output)
            PLOT_OUTPUT="$2"
            shift 2
            ;;
        --no-markdown)
            NO_MARKDOWN=1
            shift
            ;;
        --no-plot)
            NO_PLOT=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$WMMA_BIN" ]] || [[ -z "$ROCBLAS_BIN" ]] || [[ -z "$GPU" ]]; then
    echo "Error: --wmma-bin, --rocblas-bin, and --gpu are required"
    usage
fi

if [[ ! -f "$WMMA_BIN" ]]; then
    echo "Error: WMMA binary not found: $WMMA_BIN"
    exit 1
fi

if [[ ! -f "$ROCBLAS_BIN" ]]; then
    echo "Error: rocBLAS binary not found: $ROCBLAS_BIN"
    exit 1
fi

# Temporary output files
WMMA_OUTPUT="$TEMP_DIR/wmma_output.txt"
ROCBLAS_OUTPUT="$TEMP_DIR/rocblas_output.txt"

# Build command for benchmark
build_cmd() {
    local binary=$1
    local -a cmd=("$binary")
    
    if [[ -n "$SHAPES" ]]; then
        cmd+=(--shapes "$SHAPES")
    fi
    
    if [[ "$BATCH_COUNT" != "1" ]]; then
        cmd+=(--batch_count "$BATCH_COUNT")
    fi
    
    printf '%q ' "${cmd[@]}"
}

# Run benchmark and save output
run_benchmark() {
    local name=$1
    local output_file=$2
    local cmd=$3
    
    echo "================================================================================"
    echo "Running $name benchmark..."
    echo "================================================================================"
    
    if [[ $VERBOSE -eq 1 ]]; then
        # Show output to console and save to file
        if eval "$cmd" 2>&1 | tee "$output_file"; then
            echo ""
        else
            echo "✗ $name benchmark failed"
            exit 1
        fi
    else
        # Just save to file
        echo "Command: $cmd"
        if eval "$cmd" > "$output_file" 2>&1; then
            echo "✓ $name benchmark completed"
        else
            echo "✗ $name benchmark failed"
            cat "$output_file"
            exit 1
        fi
    fi
    echo ""
}

# Run benchmarks
WMMA_CMD=$(build_cmd "$WMMA_BIN")
run_benchmark "WMMA" "$WMMA_OUTPUT" "$WMMA_CMD"

ROCBLAS_CMD=$(build_cmd "$ROCBLAS_BIN")
run_benchmark "rocBLAS" "$ROCBLAS_OUTPUT" "$ROCBLAS_CMD"

# Build Python command
PYTHON_CMD=(python3 generate_report.py)
PYTHON_CMD+=(--wmma-output "$WMMA_OUTPUT")
PYTHON_CMD+=(--rocblas-output "$ROCBLAS_OUTPUT")
PYTHON_CMD+=(--gpu "$GPU")

if [[ -n "$OS" ]]; then
    PYTHON_CMD+=(--os "$OS")
fi

if [[ -n "$ROCM_VERSION" ]]; then
    PYTHON_CMD+=(--rocm-version "$ROCM_VERSION")
fi

if [[ "$TITLE" != "FP16-FP16 WMMA GEMM Performance Benchmarks" ]]; then
    PYTHON_CMD+=(--title "$TITLE")
fi

if [[ "$MARKDOWN_OUTPUT" != "wmma_performance.md" ]]; then
    PYTHON_CMD+=(--markdown-output "$MARKDOWN_OUTPUT")
fi

if [[ "$PLOT_OUTPUT" != "wmma_performance_plot.png" ]]; then
    PYTHON_CMD+=(--plot-output "$PLOT_OUTPUT")
fi

if [[ $NO_MARKDOWN -eq 1 ]]; then
    PYTHON_CMD+=(--no-markdown)
fi

if [[ $NO_PLOT -eq 1 ]]; then
    PYTHON_CMD+=(--no-plot)
fi

if [[ $DEBUG -eq 1 ]]; then
    PYTHON_CMD+=(--debug)
fi

# Generate report
echo "================================================================================"
echo "Generating report..."
echo "================================================================================"
"${PYTHON_CMD[@]}"

echo ""
echo "================================================================================"
echo "Complete!"
echo "================================================================================"
