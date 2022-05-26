#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <random>
#include "normal.h"

namespace py = pybind11;

double probit_L(py::array_t<double> f, py::array_t<double> y){
    int N = y.size();

    double total = 0;
    for (int i=0; i < N; i++){
        total += logcdf(f.at(i)*y.at(i));
    }

    return total;
}

void slice_sample(py::array_t<double> f_dash, py::array_t<double> f, py::array_t<double> y, py::array_t<double> nu, double bracket_min, double bracket_max, double log_y){

    std::default_random_engine generator;

    //Gets a buffer_info struct that summarises the data
    py::buffer_info f_dash_buf_info = f_dash.request();
    py::buffer_info f_buf_info = f.request();
    py::buffer_info y_buf_info = y.request();
    py::buffer_info nu_buf_info = nu.request();

    bool not_all_1D = f_dash_buf_info.ndim != 1 || f_buf_info.ndim != 1 || y_buf_info.ndim != 1 || nu_buf_info.ndim != 1;
    bool not_all_size_N = f_dash_buf_info.size != f_buf_info.size || f_buf_info.size != y_buf_info.size || y_buf_info.size != nu_buf_info.size;

    if (not_all_1D || not_all_size_N)
        throw std::runtime_error("f, f_dash, y, and nu must all be 1D numpy arrays with N elements");

    int N = y_buf_info.size;

    //Slice sampling loop
    double angle = bracket_max;
    bool valid_sample_found = false;
    while (not valid_sample_found){

        //propose new f_dash
        for (int i=0; i<N; i++){
          f_dash.mutable_at(i) = f.at(i)*cos(angle) + nu.at(i)*sin(angle);
        }

        if (probit_L(f_dash, y) > log_y){ valid_sample_found = true;}

        else{
          if (angle < 0){
            bracket_min = angle;
          }
          else{
            bracket_max = angle;
          }
          std::uniform_real_distribution<double> distribution(bracket_min , bracket_max);
          angle = distribution(generator);
        }

    }
}

PYBIND11_MODULE(ellss, m) {
    m.def("slice_sample", &slice_sample, "Performs the slice sampling step of the ELL-SS algorithm.");
}
