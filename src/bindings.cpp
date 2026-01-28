#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "arena.h"
#include "ops.h"
#include "autograd.h"

namespace py = pybind11;

PYBIND11_MODULE(engine, m) {
    m.doc() = "High-Performance Autograd Engine";

    py::class_<Arena>(m, "Arena")
        .def(py::init<size_t>())
        .def("reset", &Arena::reset);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>>())
        .def(py::init<std::vector<int>, Arena&>())
        .def("fill", &Tensor::fill)
        .def("print", &Tensor::print)
        .def("shape", &Tensor::get_shape)
        .def("data", [](Tensor& t) { return (uintptr_t)t.data(); });

    py::class_<Variable, std::shared_ptr<Variable>>(m, "Variable")
        .def(py::init([](const Tensor& t) {
            return std::make_shared<Variable>(t.clone());
        }))
        .def("backward", &Variable::backward)
        .def("zero_grad", &Variable::zero_grad)
        .def_readonly("data", &Variable::data)
        .def_readonly("grad", &Variable::grad);

        m.def("matmul_tensor", [](const Tensor& A, const Tensor& B) {
        Tensor C({A.get_shape()[0], B.get_shape()[1]});
        ops::matmul(A, B, C);
        return C;
    });

    m.def("add", [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
        return add(a, b);
    });

    m.def("matmul", [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
        return matmul(a, b);
    });

    m.def("relu", [](std::shared_ptr<Variable> a) {
        return relu(a);
    });

    m.def("sgd_step", [](std::shared_ptr<Variable> v, float lr) {
        ops::sgd_step(v->data, v->grad, lr);
    });
}