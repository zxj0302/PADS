# Development Guide

This document provides an overview of the repository structure and guide for further development.

## Repository Structure

```
conflictrisk-public/
├── A.mat                      # Sample adjacency matrix
├── cleanup.sh                 # Script to clean up redundant files
├── demo.py                    # Consolidated demo script
├── README.md                  # Main README file
├── setup.py                   # Python package setup script
├── ConflictRisk/              # Python implementation package
│   ├── __init__.py            # Package initialization
│   ├── actual_conflict.py     # Actual conflict computation
│   ├── conflict_risk_optimization.py # Main optimization algorithm
│   ├── demo_optimization.py   # Demo functionality
│   ├── para4measure.py        # Parameters for different measures
│   ├── README.md              # Python package README
│   ├── requirements.txt       # Python dependencies
│   ├── utils.py               # Utility functions
│   └── worst_case_risk_gradient.py # Gradient computation for WCR
└── matlab_original/           # Original MATLAB implementation
    ├── actualConflict.m       # Actual conflict computation
    ├── ConflictRiskOptimization.m # Main optimization algorithm
    ├── demo_Optimization.m    # Demo script
    ├── para4Measure.m         # Parameters for different measures
    ├── README.md              # MATLAB documentation
    └── worstCaseRiskGradient.m # Gradient computation for WCR
```

## Development Workflow

### Python Implementation

1. The core functionality is in the `ConflictRisk` package
2. For new features, add them to the appropriate module or create a new one
3. Update `__init__.py` to expose new functionality
4. Update the demo script if needed

### Testing

Currently, there are no formal tests. Consider adding:

1. Unit tests for individual functions
2. Integration tests for the optimization workflow
3. Comparison tests between MATLAB and Python implementations

### Documentation

1. Each function has docstrings detailing parameters and return values
2. The README files provide usage instructions
3. Consider adding more extensive documentation with examples

## Contributing

When contributing to this repository:

1. Keep code well-documented with proper docstrings
2. Follow the existing code style (PEP 8 for Python)
3. Add tests for new functionality
4. Update documentation as needed
