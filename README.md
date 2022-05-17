# gproc
Guassian Process Classifiction

## Envrionment
Python environment is being managed by conda.
Configuration is stored in the `ENV.yml` file.
To create your environment just use `conda env create --file ENV.yml` from the base of the repo.
Then execute `conda activate gproc` to activate the environment.
To update using a new config use `conda env update --file ENV.yml --prune`.
For other conda tasks check out the cheat sheet @ https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf .

## Install
To install in dev mode run `pip install -e .` from the root of the project repo.