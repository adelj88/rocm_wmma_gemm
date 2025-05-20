# ROCm WMMA GEMM

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
- Python3 (required for config generation)
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

### Tuning
The GEMM implementation includes a tuning system to find optimal configurations (warps and tile sizes) for specific matrix dimensions and layouts. The tuner:
- Tests various combinations of warp and tile configurations
- Supports different matrix layouts (row-major and column-major)
- Generates a JSON configuration file used by the library
- Measures actual performance for each configuration

To run the tuner:
```bash
cd build
python3 tune.py  # Results written to gemm_config_tuned.json
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
