## Minimal Deep Learning Autograd Engine (C++ / pybind11 / OpenMP / SIMD)

A small, educational deep learning **autograd engine** implemented in modern C++ and exposed to Python via `pybind11`.  
It provides a `Tensor` type, a `Variable` node with automatic differentiation (reverse‑mode), and a handful of neural‑network style ops (matmul, ReLU, softmax, SGD) accelerated with **SIMD** and **OpenMP**.

This project is designed to be easy to read and hack on while still demonstrating serious performance techniques (arena allocator, vectorization, multithreading).

---

## Features

- **Autograd engine (reverse‑mode)**:
  - `Variable` nodes hold `data` and `grad` tensors
  - Dynamic computation graph with `children` and per‑node `backward_fn`
  - Topological traversal and backprop via `Variable::backward()`
- **Core tensor type**:
  - 2D `Tensor` with contiguous float storage and simple indexing
  - Optional **arena allocator** for fast, bump‑pointer style allocations
  - Move‑only semantics to avoid accidental deep copies
- **Neural network ops** (in namespace `ops` and/or as autograd functions):
  - `matmul` (SIMD‑optimized with NEON/AVX2 + OpenMP)
  - `add`
  - `relu` and `relu_backward`
  - `softmax` (row‑wise)
  - `transpose`
  - `div_scalar`
  - `sgd_step` (in‑place SGD weight update)
- **Python bindings**:
  - Built as a Python extension module named `engine`
  - Thin Python API around C++ `Tensor` / `Variable` / ops
- **Performance techniques**:
  - Platform‑specific SIMD (`ARM NEON` on Apple Silicon, `AVX2` on x86‑64)
  - **OpenMP** parallel loops across tensor elements / rows
  - Manual memory management via an `Arena` for temporary tensors

---

## Project Structure

```text
cpp-inference-engine/
├── CMakeLists.txt          # Build configuration (C++17, pybind11, OpenMP, SIMD flags)
├── include/
│   ├── arena.h             # Simple bump-pointer arena allocator for floats
│   ├── tensor.h            # Tensor class declaration
│   ├── ops.h               # Low-level tensor ops (matmul, relu, softmax, etc.)
│   └── autograd.h          # Variable struct and autograd API
└── src/
    ├── arena.cpp           # Arena implementation
    ├── tensor.cpp          # Tensor implementation
    ├── ops.cpp             # SIMD + OpenMP kernels and tensor ops
    ├── autograd.cpp        # Autograd graph construction and backward passes
    ├── bindings.cpp        # pybind11 Python bindings (module `engine`)
    └── demo.py             # Example Python training script (tiny MLP on XOR)
```

---

## Requirements

- **C++17** compatible compiler (Clang, GCC, or MSVC)
- **CMake** ≥ 3.10
- **Python** ≥ 3.8
- **OpenMP** development libraries
  - macOS (Apple Silicon): `brew install libomp`
  - Linux: usually `libomp-dev` / `libgomp` via your package manager
  - Windows: OpenMP comes with the MSVC toolchain
- No manual `pybind11` install is required – it is fetched automatically via `FetchContent` in `CMakeLists.txt`.

---

## Building the Python Module

From the project root:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

This produces a Python extension module named **`engine`** in the `build/` directory  
(e.g. `engine.cpython-311-darwin.so` or similar, depending on your platform/Python).

To import it from Python, either:

- Add `build/` to `PYTHONPATH`, or
- Append it at runtime:

```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
import engine
```

---

## Python API Overview

The `engine` module exposes the following core pieces:

- **`engine.Arena(size_in_floats)`**
  - Bump‑pointer arena used to back tensors without per‑tensor `new[]/delete[]`.
  - Methods:
    - `reset()` – reset the arena (all allocations become reusable).

- **`engine.Tensor(shape)`**
  - 2‑D tensor with shape `[rows, cols]`.
  - Constructors:
    - `Tensor([rows, cols])`
    - `Tensor([rows, cols], Arena)` – allocate from an `Arena`.
  - Methods:
    - `fill(value)` – fill all elements with a scalar
    - `print()` – debug print
    - `shape()` – returns the underlying shape vector
    - `data()` – returns an integer address of the underlying `float*` (for advanced users)

- **`engine.Variable`**
  - Autograd node wrapping a `Tensor`.
  - Construction:
    - `v = engine.Variable(engine.Tensor([rows, cols]))`
  - Attributes (read‑only from Python):
    - `v.data` – underlying `Tensor`
    - `v.grad` – gradient `Tensor` with same shape
  - Methods:
    - `v.backward()` – run reverse‑mode autodiff from `v` back to all ancestors
    - `v.zero_grad()` – zero out `v.grad`

- **Autograd ops (return new `Variable` nodes):**
  - `engine.add(a, b)` – elementwise add
  - `engine.matmul(a, b)` – matrix multiplication
  - `engine.relu(a)` – ReLU activation

- **Tensor‑level / helper ops:**
  - `engine.matmul_tensor(A, B)` – `Tensor × Tensor -> Tensor` (no autograd)
  - `engine.sgd_step(v, lr)` – in‑place SGD update: `v.data -= lr * v.grad`

---

## Quickstart: Minimal Autograd Example

```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
import engine

# Create parameters (weights) as Variables
w1 = engine.Variable(engine.Tensor([2, 4]))
w2 = engine.Variable(engine.Tensor([4, 1]))
w1.data.fill(0.5)
w2.data.fill(0.5)

# Single input example: shape [1, 2]
x = engine.Variable(engine.Tensor([1, 2]))
x.data.fill(1.0)  # simplified; library focuses on engine mechanics, not data loaders

# Forward pass: x -> matmul -> relu -> matmul
h = engine.matmul(x, w1)
h_relu = engine.relu(h)
y = engine.matmul(h_relu, w2)

# Treat y as "loss" just to exercise backprop
w1.zero_grad()
w2.zero_grad()
y.backward()

# Gradient step
lr = 0.01
engine.sgd_step(w1, lr)
engine.sgd_step(w2, lr)

print("Updated weights:")
w1.data.print()
w2.data.print()
```

For a more complete example (a tiny MLP trained on the XOR problem), see `src/demo.py`.

---

## Autograd Design

- **`Variable` graph**
  - Each `Variable` stores:
    - `Tensor data`
    - `Tensor grad`
    - `std::vector<std::shared_ptr<Variable>> children`
    - `std::function<void()> backward_fn`
  - Operations like `add`, `matmul`, and `relu`:
    - Build a new `Variable` with output `data`
    - Record input `Variable` pointers in `children`
    - Attach a `backward_fn` closure that:
      - Reads `out.grad`
      - Accumulates into each input’s `grad` using the appropriate math

- **Backward pass**
  - `Variable::backward()`:
    - Builds a **topological ordering** of the graph (children before parents)
    - Seeds the starting node’s `grad` with `1.0` (dL/dL = 1)
    - Walks the topo order in reverse and calls each node’s `backward_fn`

- **Examples of local gradients**
  - `add(a, b)`:
    - `dL/da += dL/dout`
    - `dL/db += dL/dout`
  - `matmul(a, b)`:
    - `dL/da += dL/dout * b^T`
    - `dL/db += a^T * dL/dout`
  - `relu(a)`:
    - Uses `ops::relu_backward`, which masks gradients where `a.data <= 0`

---

## Performance & Implementation Notes

- **SIMD & OpenMP**
  - `ops::matmul` uses:
    - `ARM NEON` intrinsics on Apple Silicon (`float32x4_t`, `vfmaq_f32`, etc.)
    - `AVX2` intrinsics on x86‑64 (`__m256`, `_mm256_fmadd_ps`, etc.)
  - OpenMP `#pragma omp parallel for` is used to parallelize:
    - Rows in matmul
    - Elementwise ops like ReLU, add, div, SGD, etc.

- **Arena allocator**
  - `Arena` is a simple bump‑pointer allocator for `float`:
    - `alloc(count)` returns a `float*` slice from a pre‑allocated buffer
    - `reset()` rewinds the offset to reuse memory
  - `Tensor` can either:
    - Own its own heap buffer, or
    - Borrow memory from an `Arena` (non‑owning) for faster temporary allocations.

---

## Running the Demo

After building the module:

```bash
cd src
python demo.py
```

`demo.py` constructs a tiny 2‑layer MLP and trains (in a deliberately simplified way) on the XOR dataset, mainly to exercise the autograd engine and verify gradient flow.

---

## Contributing

This is intentionally a small, hackable project.  
Issues, suggestions, and pull requests to add new ops, layers, or visualizations of the computation graph are welcome.

## High-Performance Tensor Library (C++, CMake, OpenMP, SIMD)

A high-performance tensor and linear algebra library in modern C++ with SIMD-optimized matrix multiplication and activation functions. It showcases platform-specific SIMD (ARM NEON on Apple Silicon, AVX2 on x86-64) combined with OpenMP-based parallelism to accelerate core numerical kernels.

## Features

- **Tensor Class**: Efficient 2D tensor implementation with intuitive indexing and move semantics
- **SIMD-Optimized Matrix Multiplication**: Platform-aware implementations using:
  - **ARM NEON** for Apple Silicon (M1/M2/M3 chips)
  - **Intel AVX2** for x86-64 processors
- **Activation Functions**: ReLU (Rectified Linear Unit) with parallel processing
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

// Access raw data pointer (useful for SIMD operations)
float* data = t.data();

// Move semantics (copy constructor/assignment deleted for performance)
Tensor moved = std::move(t);
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

### Activation Functions

#### ReLU (Rectified Linear Unit)
The `relu()` function applies the ReLU activation function element-wise: `Z = max(0, X)`. It's parallelized using OpenMP for efficient processing of large tensors.

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
│   └── ops.h               # Matrix multiplication and activation functions
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

### Design Decisions

- **Move Semantics**: Tensor class uses move semantics and deleted copy constructor/assignment to prevent expensive deep copies and encourage efficient memory management
- **Raw Data Access**: The `data()` method provides direct access to the underlying memory buffer for advanced optimizations and SIMD operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

