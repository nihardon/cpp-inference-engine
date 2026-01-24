#include <iostream>
#include "tensor.h"
#include "ops.h"

int main() {
    // Create A (2x3)
    // 1 2 3
    // 4 5 6
    Tensor A({2, 3});
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;

    // Create B (3x2)
    // 7 8
    // 9 1
    // 2 3
    Tensor B({3, 2});
    B(0,0)=7; B(0,1)=8;
    B(1,0)=9; B(1,1)=1;
    B(2,0)=2; B(2,1)=3;

    // Create C (2x2) to hold the result
    Tensor C({2, 2});

    // Run MatMul
    std::cout << "Running Naive Matrix Multiplication...\n";
    matmul_naive(A, B, C);

    // Print Result
    // Expected:
    // [ (1*7 + 2*9 + 3*2)  (1*8 + 2*1 + 3*3) ]  -> [ 31  19 ]
    // [ (4*7 + 5*9 + 6*2)  (4*8 + 5*1 + 6*3) ]  -> [ 85  55 ]
    C.print();

    return 0;
}