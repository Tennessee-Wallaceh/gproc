# gproc
Guassian Process Classifiction

## Envrionment
Python environment is being managed by conda, if needed, can install via (https://docs.conda.io/en/latest/miniconda.html).
Or just execute 
```
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Configuration is stored in the `ENV.yml` file.
To create your environment just use `conda env create --file ENV.yml` from the base of the repo.
Then execute `conda activate gproc` to activate the environment.
To update using a new config use `conda env update --file ENV.yml --prune`.
For other conda tasks check out the cheat sheet @ https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf .

## Install
To install in dev mode run `pip install -e .` from the root of the project repo.

C++ dependencies are built by `pybind11`.
This package is installed by `pip` in the `ENV.yml`.
To use this we add `"pybind11~=2.6.1"` to our required build systems in `pyproject.toml`.
Additionally, we need to add a build step to our `setup.py` file, which will build `.cpp` files in the `cpp/` directory and map to conigured python module structure.
For more details look at https://pybind11.readthedocs.io/en/latest/basics.html .
Static configuration is managed in `setup.cfg`.

## Data
The data folder for the reported experiments, if required, can be downloaded from https://drive.google.com/drive/folders/17rLJQekKVYesQjRptC9g2GDKa5yv7qV2?usp=sharing . The experiment script may need to be edited to point at the data folder.

## Documentation

Documented in NumPy style.
Documentation is built automatically by sphinx.
To build run `make html` from the `docs` directory.
The corresponding docs are then found in `docs/_build/index.html`.
On merge to main, the docs are rebuilt and deployed automatically to `https://tennessee-wallaceh.github.io/gproc/`.
For the automatic documentation to be generated the configuration needs to be added to `docs/index.rst`.
