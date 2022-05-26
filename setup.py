from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

# Name eg example have to match name of directory
ext_modules = [
    Pybind11Extension(
        "gproc.example",
        sorted(glob("src/cpp/example/*.cpp")),  # Sort source files for reproducibility
        libraries=["m"],
        extra_compile_args = ["-O3", "-ffast-math", "-fopenmp"],  # include the -fopenmp flag to give access to omp
        extra_link_args=["-fopenmp"],
    ),
    Pybind11Extension(
        "gproc.ellss",
        sorted(glob("src/cpp/ellss/*.cpp")),  # Sort source files for reproducibility
        libraries=["m"],
        extra_compile_args = ["-O3", "-ffast-math", "-fopenmp"],  # include the -fopenmp flag to give access to omp
        extra_link_args=["-fopenmp"],
    ),
]

setup(ext_modules=ext_modules)
