# ROCm WMMA GEMM

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/adelj88/rocm_wmma_gemm)

This repository provides a standalone, high-performance General Matrix Multiplication (GEMM) implementation optimized for AMD GPUs using ROCm's Wave Matrix Multiply-Accumulate (WMMA) intrinsics. It is derived from the fastest half-precision GEMM kernel developed in the `hgemm` sample within the [rocm_wmma_samples](https://github.com/adelj88/rocm_wmma_samples/tree/main/hgemm) project. This new repository refactors the kernel to facilitate exploration of different matrix data layouts and further optimizations.

## Purpose
This repository aims to:
- Provide a focused, high-performance GEMM kernel utilizing ROCm WMMA intrinsics.
- Isolate and refine the fastest GEMM implementation derived from the `hgemm` sample in `rocm_wmma_samples`.
- Explore and implement support for various matrix data layouts (e.g., row-major, column-major, potentially tiled formats) beyond the format used in the sample.
- Tune the GEMM kernel for different M, N, K sizes

## Building the Project

### Prerequisites
- AMD ROCm installed with HIP support
- CMake version 3.10 or higher
- Python3 (required for config generation and tuning)
  - Python packages (can be installed with pip or conda)
    - ``numpy``
- AMD RDNA3/RDNA3.5/RDNA4 GPU (required for WMMA support)

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
   CXX=/opt/rocm/bin/hipcc cmake ..
   make
   ```

### Usage
Run the executable after building:
```bash
# Assumes you're currently in /build directory
# To run unit tests
./test/gemm_test

# To run unit benchmarks
./benchmark/gemm_bench

# To run rocblas equivalent for verification
./test/rocblas_test
./benchmark/rocblas_bench
```

### Automatic Kernel Tuning
The library includes a Genetic Algorithm-based tuner that automatically finds optimal kernel configurations for different matrix sizes and data layouts.

#### **Tuning Approach**
The tuner uses **Genetic Algorithm (GA)** to efficiently explore the discrete parameter space:

- **Population-based search**: Explores multiple configurations simultaneously through evolutionary generations
- **Smart initialization**: Seeds initial population with proven baseline configurations plus diverse random individuals
- **Tournament selection**: Selects high-performing configurations as parents for the next generation
- **Uniform crossover**: Combines successful parameter combinations from different parent configurations
- **Constraint-aware mutation**: Randomly explores new parameter values while respecting hardware constraints
- **Elitism**: Preserves the best configurations across generations to prevent loss of good solutions
- **Reproducible results**: Uses configurable random seeds for consistent and repeatable tuning runs

To run the tuner:
```bash
cd build
# Default behavior (all sizes and layouts)
python3 tune.py # Results written to gemm_config_tuned.json

# Test specific sizes
python3 tune.py --sizes 1024,1024,1024 2048,2048,2048

# Adjust evaluation budget
python3 tune.py --budget 80

# Test specific layouts
python3 tune.py --layouts r,c c,c

# Reproducible results with specific seed
python3 tune.py --seed 123

# Adjust GA parameters for different exploration
python3 tune.py --pop-size 30 --mutation-rate 0.4

# Different GPU architecture
python3 tune.py --gpu-arch gfx1103

# Custom output file
python3 tune.py --output my_config.json
```

## Performance Results
- [View detailed square matrix benchmarks](docs/square.md)
- [View detailed rectangular matrix benchmarks](docs/rectangle.md)

## Future Plans
1. Add batched implementation for half GEMM
2. Add BF16 implementation
3. Explore any possibility of further optimizations (e.g. Stream-K for smaller M, N, K)
4. Tuning for RDNA3.5 and RDNA4
5. (Maybe) Simplify interface by using leading dimensions, similar to BLAS libraries

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
