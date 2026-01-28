#include "arena.h"

Arena::Arena(size_t size_in_floats) 
    : total_size_(size_in_floats), offset_(0) {
    buffer_ = new float[total_size_];
}

Arena::~Arena() {
    delete[] buffer_;
}

float* Arena::alloc(size_t count) {
    if (offset_ + count > total_size_) {
        throw std::runtime_error("Arena Out of Memory!");
    }
    
    float* ptr = buffer_ + offset_;
    
    offset_ += count;
    
    return ptr;
}

void Arena::reset() {
    offset_ = 0;
}