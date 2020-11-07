# Deep Learning Tutorials with Lightning and Hydra

### Setup
If you already have a python environment and have [`poetry`](https://python-poetry.org) installed, you can skip ahead
to the [Installing Dependencies](#installing-dependencies) section. Otherwise, it is recommended to first go through the
[Virtual Environment](#virtual-environment) section to setup an environment with all the required tools.

#### Virtual Environment
If you don't operate inside a virtual environment, or only have access to an incompatible python version, it is
recommended you create a virtual environment using [`conda`](https://docs.conda.io/en/latest/):
```shell script
conda env create -f environment.yml
conda activate deep-learning-tutorials
```

#### Installing Dependencies
Once you have a python interpreter and poetry setup, simply install the project's dependencies:
```shell script
poetry install
```

#### Development Setup
If you want to contribute to the project, then you have one to perform an additional setup step: installing the
pre-commit hooks. This is done to ensure that any code committed to the repository meets the project's format and
quality standards.
