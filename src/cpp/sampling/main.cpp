#include <stdarg.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // Special Header for numpy operations
#include <stdexcept>

using namespace pybind11::literals; // to bring in the `_a` literal
namespace py = pybind11;

typedef double (*Acceptance_Function)(double *, double *, int); // function pointer type
typedef std::unordered_map<std::string, Acceptance_Function> acceptance_function_map;

acceptance_function_map = 

double normal_ratio(double *current, double *proposal, int dim) {
    // va_arg( acceptance_args, int );
    std::cout << "acceptance 1" << std::endl;
    double x = 0.1;
    return x;
}


py::array_t<double> run_mh(py::array_t<double> initial, std::string proposal, std::string acceptance_ratio, int num_samples, int burn_in, py::dict acceptance_kwargs) {
    // Convert numpy arrays to pointers
    py::buffer_info buf_initial = initial.request();
    double *initial_ptr = static_cast<double *>(buf_initial.ptr);
    int dim = buf_initial.shape[0];

    // Setup Acceptance function
    // That is, a function which accepts two double arrays (last, proposal) and the dimension of those
    double (*acceptance_fcn)(double *, double *, int);
    switch (acceptance_ratio) {
        case "normal_ratio":
            acceptance_fcn = &normal_ratio;
        break;

        default:
            throw std::invalid_argument("Acceptance Ratio not implemented!");
        break;
    }

    acceptance_fcn(initial_ptr, initial_ptr, dim);

    // Set up proposal function
    // Proposals take current + random seed and advances to proposal 
    // *double (*proposal_fcn)(*double, int):
    

    // // Get reference to acceptance ratio fcn
    // if (acceptance_ratio == 'normal') {
    //     acceptance_fcn = &
    // }

    // // Burn in in serial
    // for (int i=0; i < burn_in; ++i) {
    //     ptr_out[i] = ptr_1[i] + ptr_2[i];
    // }

    // Fan out over cores to finish sampling
    auto out = py::array_t<double>(10);
    py::buffer_info buf_out = out.request();

    double l = py::getattr(acceptance_kwargs, "l").cast<double>();
    std::cout << l << std::endl;

    return out;
}


// This macro exposes our C++ functions to PYBIND11
PYBIND11_MODULE(sampling, m) {
    m.def("run_mh", &run_mh, "Function for running a number of MH algorithms");
}