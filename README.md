# Deep Learning Tutorials with Lightning and Hydra

## Setup
If you already have a python environment and have [`poetry`](https://python-poetry.org) installed, you can skip ahead
to the [Installing Dependencies](#installing-dependencies) section. Otherwise, it is recommended to first go through the
[Virtual Environment](#virtual-environment) section to setup an environment with all the required tools.

### Virtual Environment
If you don't operate inside a virtual environment, or only have access to an incompatible python version (<3.8), it is
recommended you create a virtual environment using [`conda`](https://docs.conda.io/en/latest/):
```shell script
conda env create -f environment.yml
conda activate deep-learning-tutorials
```

### Installing Dependencies
Once you have a python interpreter and poetry setup, simply install the project's dependencies:
```shell script
poetry install
```

## How to Contribute
If you want to contribute to the project, then you have to install the pre-commit hooks, on top of the basic setup for
using the project, detailed [above](#setup). The pre-commit hooks are there to ensure that any code committed to the
repository meets the project's format and quality standards.
```shell script
pre-commit install
```
> Be aware that for some pre-commit hooks to run correctly, poetry MUST be installed globally on the machine you'll be
committing from. If you don't already have it installed globally, the instructions to do so are available
[here](https://python-poetry.org/docs/#installation).
