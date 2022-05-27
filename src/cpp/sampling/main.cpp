#include <stdarg.h>
#include <iostream>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // Special Header for numpy operations
#include <stdexcept>

#include "mkl.h"

using namespace pybind11::literals; // to bring in the `_a` literal
namespace py = pybind11;

// Acceptance function setup
// acceptance functions are stored in map from string identifier to a pointer to a function
// which returns a double
typedef double (*Acceptance_Fcn)(double *, double *, int, py::dict& acceptance_kwargs);
typedef std::map<std::string, Acceptance_Fcn> acceptance_function_map;

acceptance_function_map a_map;

double normal_ratio(double *current, double *proposal, int dim, py::dict& acceptance_kwargs) {
    // Convert kwargs into C++ types
    float l = py::float_(acceptance_kwargs["l"]).cast<float>();

    // va_arg( acceptance_args, int );
    std::cout << "| normal_ratio |" << l << std::endl;
    double x = 0.1;
 
    return x;
}

const auto success_nr = a_map.insert({"normal_ratio", &normal_ratio}); // We are throwing away returned value

// Proposal function setup
// Proposals take current + random seed and advances to proposal 
typedef void (*Proposal_Fcn) (double *, int, py::dict& acceptance_kwargs);
typedef std::map<std::string, Proposal_Fcn> proposal_function_map;

proposal_function_map p_map;

void independent_normal_proposal(double *current, int seed, py::dict& proposal_kwargs) {
    // Convert kwargs into C++ types
    
    // va_arg( acceptance_args, int );
    std::cout << "| ind norm |" << std::endl;
    double x = 0.1;
 
    return x;
}

const auto success_inp = p_map.insert({"independent_normal", &independent_normal_proposal}); // We are throwing away returned value

// Run a single chain given a random state
void run_chain(
    VSLStreamStatePtr stream, 
    float * out, 
    int num_samples,
    int sample_dimension,
    Proposal_Fcn &proposal_fcn, 
    py::dict& proposal_kwargs,
    Acceptance_Fcn &acceptance_fcn,
    py::dict& acceptance_kwargs
) {
    // We need one uniform sample for each step
    float uniform_samples[num_samples]; 
    float *uniform_samples_ptr = &uniform_samples[0];
    vsRngUniform(
        VSL_RNG_METHOD_UNIFORM_STD, // gen method
        stream, // pointer to stream
        num_samples, // number of samples
        uniform_samples_ptr, // out buffer
        0., 1. // min/max
    );

    for (int i=0; i < num_samples; i++) {
        proposal_fcn(&out[i], sample_dimension, proposal_kwargs);
    }

    // vsRngGaussian(
    //     VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, // gen method
    //     stream, // stream pointer
    //     num_samples, // num samples
    //     out, // out buffer
    //     5.0f, // mean 
    //     2.0f // std
    // );


}

// Run MH
py::array_t<double> run_mh(
    py::array_t<double> initial,
    std::string proposal,
    std::string acceptance_ratio,
    int samples_per_chain,
    int sample_dimension,
    int burn_in,
    int chains,
    py::dict& acceptance_kwargs, 
    py::dict& proposal_kwargs
) {
    // Convert intial array to a pointer
    py::buffer_info buf_initial = initial.request();
    double *initial_ptr = static_cast<double *>(buf_initial.ptr);
    int dim = buf_initial.shape[0];

    // Setup Acceptance function
    // That is, a function which accepts two double arrays (last, proposal) and the dimension of those
    auto acceptance_search = a_map.find(acceptance_ratio);
    if (acceptance_search == a_map.end()) {
        throw std::invalid_argument("Acceptance Ratio not implemented!");
    }
    auto acceptance_fcn = acceptance_search->second;

    acceptance_fcn(initial_ptr, initial_ptr, dim, acceptance_kwargs);

    // Setup proposal function
    auto proposal_search = p_map.find(proposal);
    if (proposal_search == p_map.end()) {
        throw std::invalid_argument("Proposal not implemented!");
    }
    auto proposal_fcn = proposal_search->second;

    // Set up the out array and get pointer
    auto out = py::array_t<float>(samples_per_chain * chains * sample_dimension);
    py::buffer_info buf_out = out.request();
    float *out_ptr = static_cast<float *>(buf_out.ptr);

    int seed = 10;
   
    #pragma omp parallel for
    for (int chain_ix=0; chain_ix < chains; ++chain_ix) {
        // Setup RNG for this chain
        int chain_start = chain_ix * samples_per_chain * sample_dimension;
        VSLStreamStatePtr stream;
        vslNewStream( &stream, VSL_BRNG_PHILOX4X32X10, seed );
        vslSkipAheadStream(stream, chain_start);

        // Run the chain
        run_chain(
            stream, // Random state for this stream
            out_ptr + chain_start, // Advance pointer to correct part of array
            samples_per_chain, // Number to take in this chain 
            proposal_fcn, proposal_kwargs, 
            acceptance_fcn, acceptance_kwargs // The chain functions
        );

        vslDeleteStream( &stream ); // Make sure stream is tidied
    }

    // reshape to correspond to expected shape
    out.resize({chains, samples_per_chain, sample_dimension});

    return out;
}


// This macro exposes our C++ functions to PYBIND11
PYBIND11_MODULE(sampling, m) {
    m.def("run_mh", &run_mh, "Function for running a number of MH algorithms");
}