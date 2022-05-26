#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> double_numpy_2d(py::array_t<double> input_arr){
    
    //Gets a buffer_info struct that summarises the data
    py::buffer_info buf_info = input_arr.request();

    //Ensure input is of specific dimension, so that we can demonstrate the convenient indexing functions at() and mutable_at()
    if (buf_info.ndim != 2)
        throw std::runtime_error("Number of dimensions must be 2");

    //Get shape of input so we can define new output array
    auto input_shape = buf_info.shape;

    //Create a new numpy array of the same shape as the input
    auto new_arr = py::array_t<double>(input_shape);

    for (int i=0; i < input_shape[0]; i++){
        for (int j=0; j < input_shape[1]; j++)
            new_arr.mutable_at(i,j) = 2*input_arr.at(i,j); //at() and mutable_at() allow simple indexing for reading and writing.
    }

    return new_arr;
}

PYBIND11_MODULE(example_module_2, m) {
    m.def("double_numpy_2d", &double_numpy_2d, "A function that returns a new numpy array whose elements are double those in the inputted array.");
}
