# Deep Learning Project Template for Lightning and Hydra

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/nathanpainchaud/deep-learning-template/blob/develop/LICENSE)
[![CI: pre-commit](https://github.com/nathanpainchaud/deep-learning-template/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/nathanpainchaud/deep-learning-template/actions?query=workflow%3A%22pre-commit%22)

## Setup
If you already have a python environment and have [`poetry`](https://python-poetry.org) installed, you can skip ahead
to the [Installing Dependencies](#installing-dependencies) section. Otherwise, it is recommended to first go through the
[Virtual Environment](#virtual-environment) section to setup an environment with all the required tools.

### Virtual Environment
If you don't operate inside a virtual environment, or only have access to an incompatible python version (<3.8), it is
recommended you create a virtual environment using [`conda`](https://docs.conda.io/en/latest/):
```shell script
conda env create -f environment.yml
conda activate deep-learning-template
```

### Installing Dependencies
Once you have a python interpreter and poetry setup, simply install the project's dependencies:
```shell script
poetry install
```

## Principal Command
```shell script
python run.py task=TASK model=MODEL datamodule=DATAMODULE
```

- `task` str: Task configuration to use for the run.
- `model` str: Model configuration to use for the run.
- `datamodule` str: Datamodule configuration to use for the run.

Once you've provided the required parameters described above, you can see detailed help on how to customize any other
aspect of the configuration by running:
```shell script
python run.py task=TASK model=MODEL datamodule=DATAMODULE --help
```

## Basic Example Command
```shell script
python run.py task=pixel_wise_autoencoding model=simple_ae datamodule=mnist
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
