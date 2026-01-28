#pragma once
#include "tensor.h"
namespace ops {
    void matmul_naive(const Tensor& A, const Tensor& B, Tensor& C);

    // Optimized Matrix Multiplication (SIMD)
    void matmul(const Tensor& A, const Tensor& B, Tensor& C);

    // Activation Function: ReLU (Rectified Linear Unit)
    // Z = max(0, X)
    void relu(Tensor& Z);

    void relu_backward(const Tensor& Grad_Out, const Tensor& Input, Tensor& Grad_In);

    // Computes B = A^T (A transposed)
    void transpose(const Tensor& A, Tensor& B);

    // Computes Softmax in-place (exp(x) / sum(exp(x)))
    void softmax(Tensor& Z);

    // In-place scalar division: A = A / s
    void div_scalar(Tensor& A, float scalar);

    // C = A + B
    void add(const Tensor& A, const Tensor& B, Tensor& C);

    // SGD Step: param = param - lr * grad
    void sgd_step(Tensor& param, const Tensor& grad, float learning_rate);
}