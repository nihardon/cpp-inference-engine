#include "tensor.h"
#include <stdexcept>

// Constructor
Tensor::Tensor(std::vector<int> shape) {
    shape_ = shape;
    
    size_ = 1;
    for (int dim : shape) {
        size_ *= dim;
    }

    data_ = new float[size_];
}

// Destructor
Tensor::~Tensor() {
    delete[] data_;
}

// Operator Overloading for 2D Indexing
// This maps (row, col) -> 1D index
// Index = row * (number_of_cols) + col
float& Tensor::operator()(int row, int col) {
    // Simple 2D stride math
    int index = row * shape_[1] + col;
    
    // Safety check
    if (index >= size_) {
        throw std::out_of_range("Index out of bounds!");
    }
    
    return data_[index];
}

// Read-only version
const float& Tensor::operator()(int row, int col) const {
    int index = row * shape_[1] + col;
    return data_[index];
}

void Tensor::fill(float value) {
    for (int i = 0; i < size_; i++) {
        data_[i] = value;
    }
}

void Tensor::print() const {
    int rows = shape_[0];
    int cols = shape_[1];

    std::cout << "Tensor(" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        std::cout << "[ ";
        for (int j = 0; j < cols; j++) {
            // Using our own operator() to get values
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << "]\n";
    }
}