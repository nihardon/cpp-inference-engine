#pragma once
#include "tensor.h"
#include <vector>
#include <functional>
#include <memory>

struct Variable;

struct Variable : public std::enable_shared_from_this<Variable> {
    Tensor data; 
    Tensor grad; 

    std::vector<std::shared_ptr<Variable>> children;

    std::function<void()> backward_fn;

    // Constructor
    Variable(Tensor v_data) 
        : data(std::move(v_data)),
          grad(data.get_shape())   
    {
        grad.fill(0.0f); 
    }

    // Triggers backpropagation engine
    void backward();

    // Helper to zero out gradients
    void zero_grad() {
        grad.fill(0.0f);
    }
};

std::shared_ptr<Variable> add(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
std::shared_ptr<Variable> relu(std::shared_ptr<Variable> a);