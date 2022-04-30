## Monitoring system

This project may be used as a monitoring system for quantum processors 
that allows estimating their parameters based on the set of executed quantum circuits.
Currently it is in a work-in-progress state with lots of drawbacks -
typical embodiment of a "research code".   
However, it was enough to produce satisfactory results,
as it will be shown in the article, which is going to be finished soon and then published (hopefully).

### Requirements 

The main requirements are:
* python >= 3.9
* tensorflow >= 2.0
* [tensornetwork library](https://github.com/google/TensorNetwork)
* [QGOpt library](https://github.com/LuchnikovI/QGOpt)

[Qiskit](https://qiskit.org/) library is nost used in monitoring system itself, but 
is required for post-processing and working with IBMQ.
For a full list of requirements, see `requirements.txt`.

### Installation

This package is not wrapped into a Python library yet.   
Right now, as a workaround, one may clone the repository and run the test Jupyter notebooks. 
They will import .py files directly from the `solver` folder instead of an installed library.
Mind the required libraries, since there's no wheel package to resolve dependencies automatically.
After cloning, you may install essential libraries by running shell command
```shell
pip install -r requirements.txt
```

### Basic usage

Right now there is a tomography notebook which can be used as tutorial. 
It is heavily recommended to look it first;
the notebook will walk you through the most important parts of conducting experiments.

### Self-testing

Since this project is in a work-in-progress state and is not very well documented (yet),
it may also be useful to see the folder `solver/unit_tests`. 
They can provide some insights about the program.
To run the tests, one should install pytest (it is included in requirements) 
and run the simple command
```
pytest solver/unit_tests
```






