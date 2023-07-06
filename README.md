# remin: Residual Minimizer - Physics Informed Neural Networks

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0.0%2B-orange.svg)](https://pytorch.org/)

**remin** is a Python module that provides a framework for developing Physics Informed Neural Networks (PINNs) using PyTorch. This module integrates PyTorch for model creation and training, Latin-Hypercube sampling for geometry creation, and also includes a small, user-friendly module for creating geometries.

## Key Features

- Implementation of Physics Informed Neural Networks (PINNs) using PyTorch.
- Support for creating geometries using Latin-Hypercube sampling.
- Intuitive module for defining and creating complex geometries.
- Command-line interface (CLI) for simultaneous training of multiple models.

## Installation

### Prerequisites

Before using `remin`, you will need to install the following packages:

* PyTorch (version 2.0.0 or later)
* NumPy
* pyDOE (version 0.3.8 or later)

You can install all the requirements using pip, the Python package installer. To install PyTorch, run the following command or follow the instructions on their [website](https://pytorch.org/get-started/locally/) to install with GPU support:
```
pip install torch>=2.0.0
```
To install NumPy, run the following command:
```
pip install numpy
```
To install pyDOE, run the following command:
```
pip install pyDOE>=0.3.8
```
Alternatively, you can install these packages using conda or any other package manager of your choice.

Once you have installed these packages, `remin` can be installed using pip([PyPi](https://pypi.org/project/remin/)):
```
pip install remin
```
---
## Usage

To use **remin** in your Python project, import the necessary modules as follows:
```
import remin.geometry as rd
import remin.func as rf
import remin.solver.residual_loss as rl
from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
```
*Will be extended in future.*

---
## Contributing

If you want to contribute to **remin**, feel free to submit a pull request or open an issue on [GitHub](https://github.com/SalihTasdelen/remin).

## License

This project is licensed under the [MIT License](https://github.com/SalihTasdelen/remin/blob/main/LICENSE).

## Contact

For any questions, suggestions, or feedback, please feel free to contact the maintainer at [salih.tasdelen@metu.edu.tr].

Thank you for using **remin**!