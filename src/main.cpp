#include <iostream>
#include <chrono>  
#include "tensor.h"
#include "ops.h"
#include <cmath>


bool verify_tensors(const Tensor& T1, const Tensor& T2, float epsilon = 1e-4) {
    if (T1.get_shape() != T2.get_shape()) {
        std::cout << "FAILURE: Shapes differ!" << std::endl;
        return false;
    }

    int rows = T1.get_shape()[0];
    int cols = T1.get_shape()[1];
    int mismatch_count = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = std::abs(T1(i, j) - T2(i, j));
            if (diff > epsilon) {
                if (mismatch_count < 5) {
                    std::cout << "Mismatch at (" << i << "," << j << "): " 
                              << T1(i, j) << " vs " << T2(i, j) << std::endl;
                }
                mismatch_count++;
            }
        }
    }

    if (mismatch_count > 0) {
        std::cout << "FAILURE: Found " << mismatch_count << " mismatches." << std::endl;
        return false;
    }
    
    std::cout << "SUCCESS: All " << (rows * cols) << " elements match!" << std::endl;
    return true;
}


int main() {
    int size = 1024; 
    std::cout << "Initializing " << size << "x" << size << " tensors...\n";
    
    Tensor A({size, size});
    Tensor B({size, size});
    Tensor C_naive({size, size});
    Tensor C_avx({size, size});

    // Fill with dummy data (all 1.0f)
    A.fill(1.0f);
    B.fill(1.0f);

    // Benchmark Naive
    std::cout << "Running Naive MatMul..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    matmul_naive(A, B, C_naive);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_naive = end - start;
    std::cout << "Naive Time: " << diff_naive.count() << " s\n";

    // Benchmark SIMD
    std::cout << "Running SIMD MatMul..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    matmul_simd(A, B, C_avx); 
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_simd = end - start;
    std::cout << "SIMD Time:   " << diff_simd.count() << " s\n";

    // Calculate Speedup
    double speedup = diff_naive.count() / diff_simd.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    std::cout << "Verifying results..." << std::endl;
    if (verify_tensors(C_naive, C_avx)) {
        std::cout << "Benchmark Valid." << std::endl;
    } else {
        std::cout << "Benchmark Invalid: Logic Error in SIMD implementation." << std::endl;
    }

    return 0;
}