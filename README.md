# ROCm WMMA GEMM

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/adelj88/rocm_wmma_gemm)

This repository provides a standalone, high-performance General Matrix Multiplication (GEMM) implementation optimized for AMD GPUs using ROCm's Wave Matrix Multiply-Accumulate (WMMA) intrinsics. It is derived from the fastest half-precision GEMM kernel developed in the `hgemm` sample within the [rocm_wmma_samples](https://github.com/adelj88/rocm_wmma_samples/tree/main/hgemm) project. This new repository refactors the kernel to facilitate exploration of different matrix data layouts and further optimizations.

Take note that the library isn't fully tuned, and has been only tuned for some sizes (if you pass inputs that are calculated as close to the tuned sizes, the right configuration will be selected). The current workflow of this library is to tune for the specific sizes of your use-case before building. This may be improved upon in the future if time permits.

## Purpose
This repository aims to:
- Provide a focused, high-performance GEMM kernel utilizing ROCm WMMA intrinsics.
- Isolate and refine the fastest GEMM implementation derived from the `hgemm` sample in `rocm_wmma_samples`.
- Explore and implement support for various matrix data layouts (e.g., row-major, column-major, potentially tiled formats) beyond the format used in the sample.
- Support `FP16`, `BF16` and `float` accumulators
- Tune the GEMM kernel for different M, N, K sizes

## Overview

This implementation leverages ROCm's Wave Matrix Multiply-Accumulate (WMMA) intrinsics to achieve high-performance GEMM operations across diverse matrix configurations and data layouts.

### Performance Analysis Across Matrix Shapes

Testing on AMD RX 7900 GRE (gfx1100) and 8060S (gfx1151) reveals distinct performance patterns for both square and rectangular matrices:

**Square Matrix Performance by Layout:**

![gfx1100 WMMA Square Performance](docs/gfx1100_square.png)

![gfx1151 WMMA Square Performance](docs/gfx1151_square.png)

**Rectangular Matrix Performance by Layout:**

![gfx1100 WMMA Rectangular Performance](docs/gfx1100_rectangle.png)

![gfx1151 WMMA Rectangular Performance](docs/gfx1151_rectangle.png)

**Key Finding**: `rocm_wmma_gemm` remains competitive with rocBLAS across diverse matrix configurations, demonstrating that WMMA intrinsics can be effectively leveraged for high-performance GEMM implementations.

## Building the Project

### Prerequisites
- AMD ROCm installed with HIP support
- CMake version 3.10 or higher
- Python3 (required for config generation and tuning)
  - Python packages (can be installed with pip or conda)
    - ``numpy``
    - ``optuna``
    - ``matplotlib``
- AMD RDNA3/RDNA3.5 GPU (required for WMMA support)

### Build Steps
1. Clone the repository:
   ```bash
   git https://github.com/adelj88/rocm_wmma_gemm.git
   cd rocm_wmma_gemm
   ```
2. Build:
   ```bash
   mkdir build
   cd build
   # build for gfx1100
   CXX=/opt/rocm/bin/amdclang++ cmake -DGPU_TARGET=gfx1100 ..
   # build for gfx1151
   CXX=/opt/rocm/bin/amdclang++ cmake -DGPU_TARGET=gfx1151 ..
   make
   ```

### Usage
Run the executable after building:
```bash
# Assumes you're currently in /build directory
# To run unit tests
./test/test_float_accum
./test/test_same_prec

# To run unit benchmarks
./benchmark/bench_bf16_bf16
./benchmark/bench_float_bf16
./benchmark/bench_float_half
./benchmark/bench_half_half

# Pass custom sizes
./benchmark/bench_half_half --shapes 4096,4096,2048:4096,5120,5120
./benchmark/bench_half_half --shapes 2048:4096 # passes squares

# To run rocblas equivalent for verification
./test/test_rocblas
./benchmark/bench_rocblas
```

### Automatic Kernel Tuning
The library includes an Optuna-based Tree-structured Parzen Estimator (TPE) tuner that automatically finds optimal kernel configurations for different matrix sizes and data layouts.

#### **Tuning Approach**
The tuner uses **Optuna TPE (Tree-structured Parzen Estimators)** to efficiently explore the discrete parameter space:

- **TPE optimization**: Models the performance landscape using probabilistic distributions to intelligently sample promising regions
- **Smart initialization**: Tests proven baseline configurations first to seed the optimization with known good solutions
- **Multivariate learning**: Understands relationships between parameters (e.g., block sizes and tile configurations)
- **Adaptive sampling**: Balances exploration of uncertain regions with exploitation of high-performing areas
- **Reproducible results**: Uses configurable random seeds for consistent and repeatable tuning runs

To run the tuner:
```bash
cd build

# Default behavior (all sizes and layouts)
python3 tune.py # Results written to gemm_config_tuned.json

# Test specific sizes
python3 tune.py --sizes 1024,1024,1024 2048,2048,2048

# Adjust evaluation budget
python3 tune.py --budget 100

# Test specific layouts
python3 tune.py --layouts r,c c,c

# Reproducible results with specific seed
python3 tune.py --seed 123

# Different GPU architecture
python3 tune.py --gpu-arch gfx1103

# Custom output file
python3 tune.py --output my_config.json

# Custom baseline configurations
python3 tune.py --baselines 4,4,4,4,256,0,0 2,2,4,4,128,1,1 8,2,2,2,64,1,0
```

## Performance Results
Below are benchmark results (in TFLOPs) that compares `rocm_wmma_gemm` against `rocblas` for all layouts and different sizes.

- [View detailed gfx1100 square matrix benchmarks](docs/gfx1100_square.md)
- [View detailed gfx1100 rectangular matrix benchmarks](docs/gfx1100_rectangle.md)
- [View detailed gfx1151 square matrix benchmarks](docs/gfx1151_square.md)
- [View detailed gfx1151 rectangular matrix benchmarks](docs/gfx1151_rectangle.md)

To generate graphs, the following can be run:
```bash
cd docs

# Running standard benchmark sizes
bash generate_report.sh --wmma-bin ../build/benchmark/bench_half_half --rocblas-bin ../build/benchmark/bench_rocblas --gpu "AMD Radeon 8060S" --os "Ubuntu 24.04.3 LTS" --rocm-version "7.1.1" --title "Benchmarks" --markdown-output gfx1151_results.md --plot-output gfx1151_results.png

# Running specified benchmark sizes
bash generate_report.sh --wmma-bin ../build/benchmark/bench_half_half --rocblas-bin ../build/benchmark/bench_rocblas --gpu "AMD Radeon 8060S" --os "Ubuntu 24.04.3 LTS" --rocm-version "7.1.1" --title "Benchmarks" --markdown-output gfx1151_results.md --plot-output gfx1151_results.png --shapes 4096,4096,1024:8192,8192,1024
```

## Future Plans
1. Enable building 2 targets together to allow for dynamic selection based on GPU.
2. Add batched unit tests.
3. Explore any possibility of further optimizations (e.g. Stream-K for smaller M, N, K).s
4. Modify fragments to support RDNA4 WMMA.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
