# Conflict Risk Optimization

Codes for paper "Quantifying and minimizing risk of conflict in social networks".

## Overview

This repository contains both MATLAB and Python implementations of the conflict risk optimization algorithms described in the paper. The algorithms optimize network structures to minimize conflict risk using different conflict measures.

## Repository Structure

This repository contains:

- Original MATLAB implementation in the root directory
- Python implementation in the `ConflictRisk` package
- Consolidated demo script (`demo.py`)

## MATLAB Implementation

The MATLAB implementation consists of:

- `demo_Optimization.m`: Demo file showing how to run the optimization
- `ConflictRiskOptimization.m`: Main optimization function
- `para4Measure.m`: Parameters for different conflict measures
- `actualConflict.m`: Calculate actual conflict for three internal opinion vectors
- `worstCaseRiskGradient.m`: Calculate worst-case risk gradient

### Requirements

- MATLAB
- CVX (for convex optimization)

### Usage

Run the `demo_Optimization.m` file in MATLAB after loading an adjacency matrix of an undirected network.

## Python Implementation

The Python implementation is contained within the `ConflictRisk` package:

- `demo_optimization.py`: Demo file showing how to run the optimization
- `conflict_risk_optimization.py`: Main optimization function
- `para4measure.py`: Parameters for different conflict measures
- `actual_conflict.py`: Calculate actual conflict for three internal opinion vectors
- `worst_case_risk_gradient.py`: Calculate worst-case risk gradient

### Requirements

The Python implementation requires:

- numpy
- scipy
- matplotlib
- cvxpy

You can install all requirements using:

```bash
pip install -r ConflictRisk/requirements.txt
```

### Usage

You can run the Python demo in multiple ways:

1. Use the consolidated demo script:

```bash
# Run standard demo with default settings
python demo.py

# Create a sample network first, then run demo
python demo.py --create-sample

# Run comparison of all measures
python demo.py --mode comparison

# Run specific measure with custom parameters
python demo.py --measure 2 --method 1 --case 1 --iterations 40
```

2. Import and use the package in your own Python code:

```python
from ConflictRisk import conflict_risk_optimization, create_random_network

# Create a random network
A = create_random_network(n=20, p=0.3)

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

## Parameters

Both implementations support the same parameters:

- Conflict measures (m):
  - 1: internal conflict
  - 2: external conflict
  - 3: controversy
  - 4: resistance

- Optimization methods:
  - Projected gradient descent
  - Coordinate descent

- Internal opinion vector cases:
  - Average case
  - Worst case

## References

A short introduction video: https://www.youtube.com/watch?v=LT2ALCeG0S8&t=3s
