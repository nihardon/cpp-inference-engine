# C++ Inference Engine

A high-performance C++ inference engine featuring SIMD-optimized matrix multiplication operations. This project demonstrates the performance benefits of vectorized operations using platform-specific SIMD instructions (ARM NEON for Apple Silicon, AVX2 for Intel/AMD processors).

## Features

- **Tensor Class**: Efficient 2D tensor implementation with intuitive indexing
- **SIMD-Optimized Matrix Multiplication**: Platform-aware implementations using:
  - **ARM NEON** for Apple Silicon (M1/M2/M3 chips)
  - **Intel AVX2** for x86-64 processors
- **Parallel Processing**: OpenMP integration for multi-threaded execution
- **Benchmarking Suite**: Built-in performance comparison between naive and optimized implementations
- **Cross-Platform**: Automatic platform detection and appropriate SIMD instruction selection

## Requirements

- **C++17** compatible compiler (GCC, Clang, or MSVC)
- **CMake** 3.10 or higher
- **OpenMP** library
  - On macOS (Apple Silicon): Install via Homebrew: `brew install libomp`
  - On Linux: Usually available via package manager
  - On Windows: Included with Visual Studio

## Building

### macOS (Apple Silicon)

```bash
# Install OpenMP if not already installed
brew install libomp

# Build the project
mkdir build
cd build
cmake ..
make

# Run the benchmark
./engine
```

### Linux (Intel/AMD)

```bash
# Install OpenMP development libraries (if needed)
sudo apt-get install libomp-dev  # Ubuntu/Debian
# or
sudo yum install libgomp         # CentOS/RHEL

# Build the project
mkdir build
cd build
cmake ..
make

# Run the benchmark
./engine
```

### Windows

```bash
# Using Visual Studio Developer Command Prompt
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release

# Run the benchmark
.\Release\engine.exe
```

## Usage

The main executable runs a benchmark comparing naive and SIMD-optimized matrix multiplication:

```bash
./engine
```

The benchmark:
1. Creates two 1024×1024 tensors filled with test data
2. Performs matrix multiplication using both naive and SIMD implementations
3. Measures execution time for each approach
4. Calculates and displays the speedup factor
5. Verifies correctness by comparing results

### Example Output

```
Initializing 1024x1024 tensors...
Running Naive MatMul...
Naive Time: 2.345 s
Running SIMD MatMul...
SIMD Time:   0.156 s
Speedup: 15.03x
Verifying results...
SUCCESS: All 1048576 elements match!
Benchmark Valid.
```

## Architecture

### Tensor Class

The `Tensor` class provides a simple interface for 2D tensor operations:

```cpp
// Create a tensor
Tensor t({rows, cols});

// Access elements
float value = t(row, col);
t(row, col) = 3.14f;

// Fill with a value
t.fill(1.0f);

// Get shape information
auto shape = t.get_shape();  // Returns {rows, cols}
int size = t.get_size();     // Returns total elements
```

### Matrix Multiplication Implementations

#### Naive Implementation
Standard triple-loop matrix multiplication (O(n³) complexity) used as a baseline for comparison.

#### SIMD Implementation
Optimized matrix multiplication leveraging:
- **Vectorization**: Processes 8 elements at once (AVX2) or 4 elements (NEON)
- **Fused Multiply-Add (FMA)**: Single instruction for multiply and accumulate
- **Parallelization**: OpenMP parallel for loops across rows
- **Memory Efficiency**: Optimized memory access patterns

The SIMD implementation automatically detects the target platform and uses the appropriate instruction set.

## Performance

The SIMD-optimized implementation typically achieves **10-20x speedup** over the naive implementation on modern hardware, depending on:
- CPU architecture and SIMD capabilities
- Matrix dimensions
- Memory bandwidth
- Number of available CPU cores

## Project Structure

```
cpp-inference-engine/
├── CMakeLists.txt          # Build configuration
├── include/
│   ├── tensor.h            # Tensor class declaration
│   └── ops.h               # Matrix multiplication operations
├── src/
│   ├── tensor.cpp          # Tensor class implementation
│   ├── ops.cpp             # SIMD-optimized operations
│   └── main.cpp            # Benchmark and test suite
└── README.md               # This file
```

## Implementation Details

### Platform Detection

The code automatically detects the target platform at compile time:
- **Apple Silicon**: Uses ARM NEON intrinsics (`arm_neon.h`)
- **Intel/AMD**: Uses AVX2 intrinsics (`immintrin.h`)
- **Fallback**: Uses naive implementation if no SIMD support is detected

### SIMD Optimizations

- **AVX2**: 256-bit registers processing 8 single-precision floats simultaneously
- **NEON**: 128-bit registers processing 4 single-precision floats simultaneously
- **FMA Instructions**: `_mm256_fmadd_ps` (AVX2) and `vfmaq_f32` (NEON) for efficient multiply-accumulate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

