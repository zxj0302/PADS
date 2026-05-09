#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Handle imports both when run as a module and as a script
if __name__ == "__main__" and __package__ is None:
    # Add parent directory to path to allow relative imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ConflictRisk.conflict_risk_optimization import conflict_risk_optimization
    from ConflictRisk.utils import load_matlab_matrix, save_results_to_matlab, visualize_network
else:
    from .conflict_risk_optimization import conflict_risk_optimization
    from .utils import load_matlab_matrix, save_results_to_matlab, visualize_network

def demo_optimization(mat_file='A.mat'):
    """
    Demo script for conflict risk optimization
    
    Parameters:
    -----------
    mat_file : str
        Path to the MATLAB .mat file containing the adjacency matrix
        
    Returns:
    --------
    OptA : numpy.ndarray
        Optimized adjacency matrix
    acr : numpy.ndarray
        Average-case conflict risks
    wcr : numpy.ndarray
        Worst-case conflict risks
    conflicts : numpy.ndarray
        Actual conflicts for three internal opinion vectors
    """
    # Load adjacency matrix (it can be any undirected positive-weighted network with 0<=w_ij<=1)
    try:
        A = load_matlab_matrix(mat_file)
        print(f"Successfully loaded network with {A.shape[0]} nodes from {mat_file}")
    except Exception as e:
        print(f"Error loading matrix from {mat_file}: {str(e)}")
        print("Using a random test network instead...")
        # Generate a random test network if file loading fails
        n = 10  # Number of nodes
        A = np.random.rand(n, n)
        A = (A + A.T) / 2  # Make it symmetric
        A = A * (A > 0.5)  # Threshold to make it sparse
        np.fill_diagonal(A, 0)  # No self-loops
    
    # Display the network
    plt.figure()
    plt.imshow(A)
    plt.colorbar()
    plt.title('Original Network')
    plt.show()
    
    # Options for the four conflict measures:
    m = 2  # [1 for the internal conflict - ic;
           #  2 for the external conflict - ec;
           #  3 for the controversy - c;
           #  4 for the resistance - r]
    
    # Options for the two methods
    gradient = 1  # [1 for projected gradient descent;
                  #  0 for coordinate descent]
    
    # Options for both case internal opinions
    avg_case = 1  # [1 for average case s;
                  #  0 for the worst case s]
    
    # Other parameters
    iter_count = 50
    k = 6  # k for k/2 edges
    step_size = 1
    dim = 10
    
    # Run optimization
    OptA, acr, wcr, conflicts = conflict_risk_optimization(
        A, m, gradient, avg_case, iter_count, k, step_size, dim
    )
    
    # Plot results
    plt.figure()
    plt.imshow(OptA)
    plt.colorbar()
    plt.title('Optimized Network')
    plt.show()
    
    # Plot conflict risk changes
    plt.figure()
    plt.plot(acr, label='Average Case Risk')
    plt.plot(wcr, label='Worst Case Risk')
    plt.xlabel('Iteration')
    plt.ylabel('Risk')
    plt.legend()
    plt.title('Conflict Risk Over Iterations')
    plt.show()
    
    # Plot actual conflicts for the three internal opinion vectors
    plt.figure()
    plt.plot(conflicts[:, 0], label='Random opinion')
    plt.plot(conflicts[:, 1], label='Low-frequency opinion')
    plt.plot(conflicts[:, 2], label='High-frequency opinion')
    plt.xlabel('Iteration')
    plt.ylabel('Conflict')
    plt.legend()
    plt.title('Actual Conflict Over Iterations')
    plt.show()
    
    # Visualize the original and optimized networks
    try:
        visualize_network(A, title="Original Network")
        visualize_network(OptA, title="Optimized Network")
    except Exception as e:
        print(f"Network visualization skipped: {str(e)}")
    
    # Save results to a MATLAB file for comparison with original code
    try:
        output_file = 'python_optimization_results.mat'
        save_results_to_matlab(output_file, OptA, acr, wcr, conflicts)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    return OptA, acr, wcr, conflicts

def main():
    """Entry point for the package when used as a script"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Conflict Risk Optimization Demo')
    parser.add_argument('matrix_file', nargs='?', default='A.mat',
                        help='Path to MATLAB .mat file containing adjacency matrix (default: A.mat)')
    parser.add_argument('--measure', type=int, choices=[1, 2, 3, 4], default=2,
                        help='Conflict measure: 1=internal conflict, 2=external conflict, ' +
                             '3=controversy, 4=resistance (default: 2)')
    parser.add_argument('--method', type=int, choices=[0, 1], default=1,
                        help='Optimization method: 0=coordinate descent, 1=projected gradient descent (default: 1)')
    parser.add_argument('--case', type=int, choices=[0, 1], default=1,
                        help='Internal opinion vector case: 0=worst case, 1=average case (default: 1)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations (default: 50)')
    parser.add_argument('--budget', type=int, default=6,
                        help='Budget for edge modifications (default: 6)')
    
    args = parser.parse_args()
    
    # Run the optimization demo with specified parameters
    OptA, acr, wcr, conflicts = demo_optimization(args.matrix_file)
    
    print("\nOptimization complete!")
    print(f"Initial average case risk: {acr[0]:.6f}")
    print(f"Final average case risk: {acr[-1]:.6f}")
    print(f"Improvement: {(acr[0] - acr[-1])/acr[0]*100:.2f}%")
    
    return OptA, acr, wcr, conflicts

if __name__ == "__main__":
    main()
