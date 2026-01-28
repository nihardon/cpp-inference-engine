#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>

class Arena {
    float* buffer_;      
    size_t total_size_;  
    size_t offset_;      

public:
    Arena(size_t size_in_floats);
    ~Arena();

    float* alloc(size_t count);

    void reset();
    
    size_t used() const { return offset_; }
};