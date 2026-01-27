#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Auto-converts std::vector to Python list
#include "tensor.h"
#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(engine, m) {
    m.doc() = "High-Performance C++ Inference Engine";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>>())
        .def("fill", &Tensor::fill)
        .def("print", &Tensor::print)
        .def("data_ptr", [](const Tensor& t) {
            return (uintptr_t)t.data();
        });

    m.def("matmul", [](const Tensor& A, const Tensor& B) {
        
        int M = A.get_shape()[0];
        int N = B.get_shape()[1];
        Tensor C({M, N});
        
        matmul_simd(A, B, C);
        
        return C;
    }, "Perform Matrix Multiplication using SIMD");
}