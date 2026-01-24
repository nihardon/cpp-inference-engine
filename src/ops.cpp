#include "ops.h"
#include <stdexcept>

// Detect 
#if defined(__aarch64__) || defined(__ARM_NEON)
    #include <arm_neon.h>
    #define PLATFORM_NAME "Apple/ARM NEON"
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define PLATFORM_NAME "Intel AVX2"
#else
    #define PLATFORM_NAME "Generic CPU"
#endif

void matmul_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.get_shape()[0];
    int K = A.get_shape()[1];
    int N = B.get_shape()[1];
    
    // Quick error check
    if (B.get_shape()[0] != K) throw std::runtime_error("Dimension mismatch");

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

void matmul_simd(const Tensor& A, const Tensor& B, Tensor& C){

    int M = A.get_shape()[0];
    int K = A.get_shape()[1];
    int N = B.get_shape()[1];
    C.fill(0.0f);

#if defined(__aarch64__) || defined(__ARM_NEON)
    
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {

            // "vdupq_n_f32" = Vector Duplicate Quad (4) Float 32
            float32x4_t a_vec = vdupq_n_f32(A(i, k));

            int j = 0;

            // Loop stride is 4 because NEON holds 4 floats
            for (; j <= N - 4; j += 4) {

                // "vld1q_f32" = Vector Load 1 Quad Float 32
                float32x4_t b_vec = vld1q_f32(&B(k, j));
                float32x4_t c_vec = vld1q_f32(&C(i, j));

                // "vfmaq_f32" = Vector Fused Multiply Add Quad (C + A * B)
                float32x4_t result = vfmaq_f32(c_vec, a_vec, b_vec);

                vst1q_f32(&C(i, j), result);
            }

            // Cleanup loop for remaining columns
            for (; j < N; j++) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

#elif defined(__AVX2__)

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            __m256 a_vec = _mm256_set1_ps(A(i, k));

            int j = 0;
            for (; j <= N - 8; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&B(k, j));
                __m256 c_vec = _mm256_loadu_ps(&C(i, j));
                __m256 result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C(i, j), result);
            }
            
            for (; j < N; j++) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

#else
    // Fallback if no SIMD available
    matmul_naive(A, B, C);
#endif
}