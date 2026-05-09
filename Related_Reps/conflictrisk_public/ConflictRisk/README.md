# ConflictRisk Python Package

This is the Python implementation of the conflict risk optimization algorithms described in the paper "Quantifying and minimizing risk of conflict in social networks".

## Installation

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

There are several ways to use this package:

### 1. Using the run_demo.py script

The simplest way to run the demo:

```bash
python run_demo.py [input_matrix.mat]
```

If no input file is specified, it will look for 'A.mat' or create a random network.

### 2. Using the demo_optimization module directly

```bash
python run_direct_demo.py
```

### 3. Using the package in your own code

```python
from ConflictRisk import conflict_risk_optimization

# Create or load your adjacency matrix
import numpy as np
A = np.random.random((10, 10)) < 0.3
A = (A + A.T) / 2
np.fill_diagonal(A, 0)

# Run the optimization
OptA, acr, wcr, conflicts = conflict_risk_optimization(
    A, 
    ms=2,  # External conflict
    grad=True,  # Use projected gradient descent
    avg_case=True,  # Optimize for average case
    iter_count=50,
    k=6,
    step_size=1,
    dim=10
)
```

## API Reference

### Main Functions

- `conflict_risk_optimization(A, ms, grad, avg_case, iter_count, k, step_size, dim=10)`: Main optimization function
- `para4measure(m, L)`: Parameters for different conflict measures
- `actual_conflict(m, L)`: Calculate actual conflict for three internal opinion vectors
- `worst_case_risk_gradient(m, L, S)`: Calculate worst-case risk gradient

You can install all requirements using:

```bash
pip install -r requirements.txt
```

## Usage

The main demo file is `demo_optimization.py`. It requires an adjacency matrix of an undirected network to be loaded from a `.mat` file. You can set your own parameters in this file.

```python
from ConflictRisk import demo_optimization

# Run the demo
OptA, acr, wcr, conflicts = demo_optimization()
```

## Parameters

- `m`: Conflict measure to optimize
  - 1: internal conflict
  - 2: external conflict
  - 3: controversy
  - 4: resistance

- `gradient`: Optimization method
  - 1: projected gradient descent
  - 0: coordinate descent

- `avg_case`: Internal opinion vector case
  - 1: average case s
  - 0: worst case s

- Other parameters:
  - `iter_count`: Number of iterations
  - `k`: Budget for edge modifications (k for k/2 edges)
  - `step_size`: Step size for coordinate descent
  - `dim`: Number of worst case opinions

## Description

Detailed descriptions are provided in the docstrings of each function in the respective files.

This is a Python implementation of the original MATLAB code for conflict risk optimization in social networks.
