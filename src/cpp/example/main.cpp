#include <iostream>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // Special Header for numpy operations

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

int parallel_test(int max) {
    #pragma omp parallel
    {
        int nloops = 0;

        #pragma omp for
        for (int i=0; i < max; ++i)
        {
            ++nloops;
        }

        int thread_id = omp_get_thread_num();

        std::cout << "Thread " << thread_id << " performed "
                  << nloops << " iterations of the loop.\n";
    }

    return 0;
}

// More details on dealing with numpy can be 
// found in https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
// py::array_t<T> is a special type which corresponds to a numpy array of type T
py::array_t<double> add_numpyc(py::array_t<double> array_1, py::array_t<double> array_2) {
    // Get array info by accessing buffer view of arrays
    py::buffer_info buf_1 = array_1.request();
    py::buffer_info buf_2 = array_2.request();

    // Can now read off info about corresponding array
    int n = buf_1.shape[0];

    // Assign our outgoing array
    auto out = py::array_t<double>(n);
    py::buffer_info buf_out = out.request();

    // Check that the dimensions are as requireds
    if (buf_1.shape[0] != buf_2.shape[0])
        throw std::runtime_error("Input shapes must match");
    
    // Get pointers to our arrays (we have to do this to avoid python GIL)
    double *ptr_1 = static_cast<double *>(buf_1.ptr);
    double *ptr_2 = static_cast<double *>(buf_2.ptr);
    double *ptr_out = static_cast<double *>(buf_out.ptr);

    // Perform computation (this could be generic C++ function)
    for (int i=0; i < n; ++i) {
        ptr_out[i] = ptr_1[i] + ptr_2[i];
    }

    return out;
}

PYBIND11_MODULE(example, m) {
    m.def("addc", &add, "A function that adds two numbers");
    m.def("set_num_threads", &omp_set_num_threads, "Set number of threads"); // omp_set_num_threads directly from omp library
    m.def("parallel_test", &parallel_test, "A function that performs parallel loops");
    m.def("add_numpyc", &add_numpyc, "A function to add two numpy arrays");
}