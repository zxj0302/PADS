# Original MATLAB Implementation

This directory contains the original MATLAB implementation of the conflict risk optimization algorithms described in the paper "Quantifying and minimizing risk of conflict in social networks".

## Files

- `demo_Optimization.m`: Demo file showing how to run the optimization
- `ConflictRiskOptimization.m`: Main optimization function
- `para4Measure.m`: Parameters for different conflict measures
- `actualConflict.m`: Calculate actual conflict for three internal opinion vectors
- `worstCaseRiskGradient.m`: Calculate worst-case risk gradient

## Requirements

- MATLAB
- CVX (for convex optimization)

## Usage

Run the `demo_Optimization.m` file in MATLAB after loading an adjacency matrix of an undirected network.

## Notes

This is the original implementation. A Python version is available in the parent directory under the `ConflictRisk` package.
