#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consolidated demo script for the ConflictRisk package.

This script combines functionality from:
- run_demo.py
- run_direct_demo.py
- example_custom_usage.py
- create_sample_network.py

Usage:
  python demo.py [--matrix MATRIX_FILE] [--mode {standard,comparison}] 
                 [--measure {1,2,3,4}] [--method {0,1}] [--case {0,1}]
                 [--iterations ITER] [--create-sample] [--k K]
                 [--step-size STEP_SIZE] [--dim DIM]

Examples:
  # Run standard demo with default settings
  python demo.py
  
  # Create a sample network first, then run demo
  python demo.py --create-sample
  
  # Run comparison of all measures
  python demo.py --mode comparison
  
  # Run specific measure with custom parameters
  python demo.py --measure 2 --method 1 --case 1 --iterations 40 --k 8 --step-size 0.5 --dim 15
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Set environment variables to limit parallelism
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1" 
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ConflictRisk import (
    conflict_risk_optimization, 
    demo_optimization,
    load_matlab_matrix, 
    visualize_network,
    create_random_network,
    create_and_save_sample_network
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='ConflictRisk Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--matrix', 
        default='graph.mat',
        help='Path to MATLAB .mat file containing adjacency matrix'
    )
    parser.add_argument(
        '--mode', 
        choices=['standard', 'comparison'],
        default='standard',
        help='Demo mode: standard runs a single optimization, comparison runs multiple'
    )
    parser.add_argument(
        '--measure', 
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
        help='Conflict measure: 1=internal, 2=external, 3=controversy, 4=resistance'
    )
    parser.add_argument(
        '--method', 
        type=int,
        choices=[0, 1],
        default=1,
        help='Optimization method: 0=coordinate descent, 1=projected gradient descent'
    )
    parser.add_argument(
        '--case', 
        type=int,
        choices=[0, 1],
        default=0,
        help='Opinion case: 0=worst case, 1=average case'
    )
    parser.add_argument(
        '--iterations', 
        type=int,
        default=1,
        help='Number of iterations'
    )
    parser.add_argument(
        '--create-sample', 
        action='store_true',
        help='Create a sample network before running the demo'
    )
    parser.add_argument(
        '--k', 
        type=int,
        default=2000,
        help='Number of eigenvectors to consider'
    )
    parser.add_argument(
        '--step-size', 
        type=float,
        default=1.0,
        help='Step size for optimization'
    )
    parser.add_argument(
        '--dim', 
        type=int,
        default=50,
        help='Dimension for optimization'
    )
    parser.add_argument(
        '--min-eig', 
        type=float,
        default=1e-6,
        help='Minimum eigenvalue for optimization'
    )
    parser.add_argument(
        '--plot', 
        action='store_true',
        help='Plot results'
    )
    
    return parser.parse_args()

def run_standard_demo(args):
    """
    Run the standard optimization demo with a single measure
    """
    print("\n" + "="*50)
    print(f"Running optimization with measure {args.measure}")
    print("="*50)
    
    # Create sample network if requested
    if args.create_sample:
        print("Creating sample network...")
        create_and_save_sample_network(args.matrix)
        print(f"Sample network saved to {args.matrix}")
    
    # Load the adjacency matrix
    try:
        A = load_matlab_matrix(args.matrix)
        print(f"Loaded network with {A.shape[0]} nodes from {args.matrix}")
    except Exception as e:
        print(f"Error loading matrix from {args.matrix}: {str(e)}")
        print("Creating a random test network instead...")
        A = create_random_network(n=15, p=0.3)
    
    # Run the optimization
    try:
        OptA, acr, wcr, conflicts = conflict_risk_optimization(
            A, 
            args.measure, 
            bool(args.method), 
            bool(args.case),
            args.iterations, 
            k=args.k, 
            step_size=args.step_size, 
            dim=args.dim,
            min_eig=args.min_eig,
            plot=args.plot
        )
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        print("Try with different parameters or a different network.")
        return None, None, None, None
    
    # Display results
    print("\nOptimization complete!")
    print(f"Initial average case risk: {acr[0]:.6f}")
    print(f"Final average case risk: {acr[-1]:.6f}")
    print(f"Improvement: {(acr[0] - acr[-1])/acr[0]*100:.2f}%")
    
    if len(wcr) > 0:
        print(f"\nInitial worst case risk: {wcr[0]:.6f}")
        print(f"Final worst case risk: {wcr[-1]:.6f}")
        print(f"Improvement: {(wcr[0] - wcr[-1])/wcr[0]*100:.2f}%")
    
    if args.plot:
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(A)
        plt.colorbar()
        plt.title('Original Network')
        
        plt.subplot(1, 2, 2)
        plt.imshow(OptA)
        plt.colorbar()
        plt.title('Optimized Network')
        plt.tight_layout()
        plt.show()
        
        # Plot risk changes
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(acr, label='Average Case Risk')
        plt.plot(wcr, label='Worst Case Risk')
        plt.xlabel('Iteration')
        plt.ylabel('Risk')
        plt.legend()
        plt.title('Risk Over Iterations')
        
        plt.subplot(1, 2, 2)
        plt.plot(conflicts[:, 0], label='Random opinion')
        plt.plot(conflicts[:, 1], label='Low-frequency opinion')
        plt.plot(conflicts[:, 2], label='High-frequency opinion')
        plt.xlabel('Iteration')
        plt.ylabel('Conflict')
        plt.legend()
        plt.title('Actual Conflict Over Iterations')
        plt.tight_layout()
        plt.show()
        
        # Visualize networks
        try:
            visualize_network(A, title="Original Network")
            visualize_network(OptA, title="Optimized Network")
        except Exception as e:
            print(f"Network visualization error: {str(e)}")

    # compute how many elements are different(tolerance < 1e-5)
    diff = np.sum(np.abs(A - OptA) > 1e-5)
    print(f"Number of different elements between original and optimized network: {diff}")

def run_comparison_demo(args):
    """
    Run comparison of all conflict measures
    """
    print("\n" + "="*50)
    print("Running comparison of all conflict measures")
    print("="*50)
    
    # Create sample network if requested
    if args.create_sample:
        print("Creating sample network...")
        create_and_save_sample_network(args.matrix)
        print(f"Sample network saved to {args.matrix}")
    
    # Load the adjacency matrix
    try:
        A = load_matlab_matrix(args.matrix)
        print(f"Loaded network with {A.shape[0]} nodes from {args.matrix}")
    except Exception as e:
        print(f"Error loading matrix from {args.matrix}: {str(e)}")
        print("Creating a random test network instead...")
        A = create_random_network(n=15, p=0.3)
    
    # Parameters for optimization
    params = {
        'grad': bool(args.method),
        'avg_case': bool(args.case),
        'iter_count': args.iterations,
        'k': args.k,
        'step_size': args.step_size,
        'dim': args.dim
    }
    
    # Run optimization for each measure
    measures = {
        1: 'Internal Conflict',
        2: 'External Conflict',
        3: 'Controversy', 
        4: 'Resistance'
    }
    
    results = {}
    
    for m, name in measures.items():
        print(f"\nRunning optimization for {name}...")
        OptA, acr, wcr, conflicts = conflict_risk_optimization(
            A.copy(), m, **params
        )
        results[m] = {
            'OptA': OptA,
            'acr': acr,
            'wcr': wcr,
            'conflicts': conflicts,
            'name': name
        }
        
        print(f"  Initial risk: {acr[0]:.6f}")
        print(f"  Final risk: {acr[-1]:.6f}")
        print(f"  Improvement: {(acr[0] - acr[-1])/acr[0]*100:.2f}%")
    
    # Plot comparison of risk reduction
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    for m, res in results.items():
        plt.plot(res['acr'] / res['acr'][0], label=f"{res['name']}")
    
    plt.xlabel('Iteration')
    plt.ylabel('Normalized ACR')
    plt.legend()
    plt.title('Comparison of ACR Reduction')
    
    plt.subplot(2, 2, 3)
    plt.imshow(A)
    plt.title('Original Network')
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.imshow(results[args.measure]['OptA'])
    plt.title(f'Network Optimized for {results[args.measure]["name"]}')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    args = parse_arguments()
    
    if args.mode == 'standard':
        results = run_standard_demo(args)
    else:  # comparison mode
        results = run_comparison_demo(args)
    
    return results

if __name__ == "__main__":
    main()
