#include "ops.h"
#include <stdexcept>

void matmul_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.get_shape()[0]; // A rows
    int K = A.get_shape()[1]; // A cols 
    int N = B.get_shape()[1]; // B cols

    // Validate shapes
    if (B.get_shape()[0] != K) {
        throw std::runtime_error("Dimension mismatch: A cols must equal B rows");
    }
    if (C.get_shape()[0] != M || C.get_shape()[1] != N) {
        throw std::runtime_error("Output tensor C has wrong shape");
    }

    // The Triple Loop
    for (int i = 0; i < M; i++) {           
        for (int j = 0; j < N; j++) {       
            
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // C[i, j] += A[i, k] * B[k, j]
                sum += A(i, k) * B(k, j);
            }
            
            // Store result
            C(i, j) = sum;
        }
    }
}