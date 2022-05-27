from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
from sysconfig import get_paths

ENV_ROOT = get_paths()['data']

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
        "gproc.sampling",
        sorted(glob("src/cpp/sampling/*.cpp")),  # Sort source files for reproducibility
        extra_compile_args = ["-DMKL_ILP64", "-fopenmp", "-O3", "-ffast-math"],  # include the -fopenmp flag to give access to omp
        # extra_link_args=["-Wl,--no-as-needed"],
        library_dirs=[f"{ENV_ROOT}/lib"],
        libraries=[
            "mkl_rt",
            # "mkl_intel_ilp64",
            # "mkl_intel_thread",
            # "mkl_core",
            "iomp5",
            "pthread",
            "m",
            "dl",
        ],
        include_dirs=[f"{ENV_ROOT}/include"]
    ),
]

setup(ext_modules=ext_modules)