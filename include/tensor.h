#pragma once
#include <vector>
#include <iostream>

class Tensor {
public:
    // Constructor
    Tensor(std::vector<int> shape);

    // Destructor
    ~Tensor();

    // Data Access
    float& operator()(int row, int col);
    
    const float& operator()(int row, int col) const;

    void fill(float value);

    void print() const;

    // Getters
    const std::vector<int>& get_shape() const { return shape_; }
    int get_size() const { return size_; }

private:
    std::vector<int> shape_;
    float* data_;
    int size_;               
};
