#pragma once
#include "tensor.h"

void matmul_naive(const Tensor& A, const Tensor& B, Tensor& C);

// Optimized Matrix Multiplication (SIMD)
void matmul_simd(const Tensor& A, const Tensor& B, Tensor& C);

// Activation Function: ReLU (Rectified Linear Unit)
// Z = max(0, X)
void relu(Tensor& Z);