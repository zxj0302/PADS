#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# Set environment variables to limit parallelism
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import sys
from tqdm import tqdm
import networkx as nx

# Handle imports both when run as a module and as a script
if __name__ == "__main__" and __package__ is None:
    # Add parent directory to path to allow relative imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ConflictRisk.para4measure import para4measure
    from ConflictRisk.actual_conflict import actual_conflict
    from ConflictRisk.worst_case_risk_gradient import worst_case_risk_gradient
else:
    from .para4measure import para4measure
    from .actual_conflict import actual_conflict
    from .worst_case_risk_gradient import worst_case_risk_gradient

def run_wcr(graph_file, ms=3, grad=True, avg_case=False, iter_count=20, k=10, step_size=1, dim=10, min_eig=1e-8, plot=False):
    G = nx.read_gml(graph_file)
    A = nx.to_numpy_array(G)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return conflict_risk_optimization(
        A, ms=ms, grad=grad, avg_case=avg_case, iter_count=iter_count, k=k, step_size=step_size, dim=dim, min_eig=min_eig, plot=plot)


def conflict_risk_optimization(A, ms, grad, avg_case, iter_count, k, step_size, dim=10, min_eig=1e-8, plot=False):
    # Set thread limiting environment variables at function level
    import os
    original_env = {}
    thread_vars = [
        "OMP_NUM_THREADS", 
        "OPENBLAS_NUM_THREADS", 
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", 
        "NUMEXPR_NUM_THREADS"
    ]
    
    # Save original environment variables and set to "1"
    for var in thread_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
        os.environ[var] = "1"
    
    try:
        # Initialize network and variables
        L = np.diag(np.sum(A, axis=1)) - A  # Laplacian matrix
        n = A.shape[0]
        I = np.eye(n)
        e = np.ones(n)
        J = np.outer(e, e)  # J is the all-ones matrix
        
        # Get initial measurements
        acr0, _, _ = para4measure(ms, L)  # risk for measure m for the original network
        conflict0 = actual_conflict(ms, L)  # for the three internal opinion vectors
        
        # Arrays to store optimization results
        values = []  # stores the worst case conflict values Tr(s'*M*s)
        acrs = np.zeros(iter_count)
        conflicts = np.zeros((iter_count, 3))
        
        # Helper functions for cleaner code
        def solve_sdp_relaxation(M, n, min_eig=1e-8):
            """Helper function to solve the SDP relaxation for finding worst-case opinion vectors"""
            X = cp.Variable((n, n), symmetric=True)
            objective = cp.Maximize(cp.sum(cp.multiply(X, M)))
            constraints = [
                cp.diag(X) == 1,
                X - min_eig * I >> 0  # X is positive semidefinite with min eigenvalue
            ]
            # Add MOSEK-specific parameters to limit threading
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS)
            
            # Get matrix value and generate random vectors
            X_val = X.value
            print("X_val computed")
            
            # Use single-threaded SVD instead of Cholesky for better stability
            try:
                C = np.linalg.cholesky(X_val).T
            except np.linalg.LinAlgError:
                # Fallback to SVD if Cholesky fails
                U, s, _ = np.linalg.svd(X_val)
                C = U @ np.diag(np.sqrt(s))
            print("Cholesky/SVD computed")    
            
            # Set fixed random seed for reproducibility
            rng = np.random.RandomState(42)  
            V = np.sign(C @ rng.randn(C.shape[1], dim))  # Relaxation
            
            return prob.value, V, X_val
        
        def optimize_with_gradient_descent(A, gradient_matrix, k):
            """Optimize using projected gradient descent"""
            step = cp.Variable((n, n), symmetric=True)
            diag_sum_step = cp.diag(cp.sum(step, axis=1))
            
            objective = cp.Maximize(cp.sum(cp.multiply(gradient_matrix, diag_sum_step - step)))
            constraints = [
                cp.diag(step) == 0,
                A - step >= 0,
                A - step <= 1,
                cp.sum(cp.abs(step)) <= k
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.MOSEK, mosek_params={
                'MSK_IPAR_NUM_THREADS': 1,
            })
            
            return step.value
    
        def optimize_acr_with_coordinate_descent(A, gradient_matrix, step_size):
            """Optimize ACR using coordinate descent"""
            n = A.shape[0]
            Incr = 1000 * np.ones(A.shape)
            Decr = 1000 * np.ones(A.shape)
            
            # Calculate potential changes for each edge
            for ii in range(n):
                for jj in range(ii+1, n):
                    delta = gradient_matrix[ii, ii] + gradient_matrix[jj, jj] - gradient_matrix[ii, jj] - gradient_matrix[jj, ii]
                    Decr[ii, jj] = -delta
                    Incr[ii, jj] = delta
                    
                    # Ensure the weights are within the range [0,1]
                    if A[ii, jj] > 1 - step_size:
                        Incr[ii, jj] = 1000
                    elif A[ii, jj] < step_size:
                        Decr[ii, jj] = 1000
            
            modified = False
            
            # Find the best modification
            if np.min(Decr) < np.min(Incr) and np.min(Decr) <= 0:
                imin, jmin = np.where(Decr == np.min(Decr))
                imin, jmin = imin[0], jmin[0]
                A[imin, jmin] -= step_size
                A[jmin, imin] = A[imin, jmin]
                modified = True
            elif np.min(Incr) <= 0 or np.round(np.min(Incr), 15) == 0:
                imin, jmin = np.where(Incr == np.min(Incr))
                imin, jmin = imin[0], jmin[0]
                A[imin, jmin] += step_size
                A[jmin, imin] = A[imin, jmin]
                modified = True
                
            return A, modified
        
        # Main optimization loop
        for i in tqdm(range(iter_count), desc="Optimizing network", unit="iteration"):
            # Get the middle matrix for measure m and project to zero mean
            _, _, M = para4measure(ms, L)
            M_proj = (I - J/n) @ M @ (I - J/n)  # to make the opinion vector have 0 mean
            
            # 1. Find the current WCR when optimizing the ACR
            # 1. Find the worst-case binary opinion vector
            print("Solving SDP relaxation")
            prob_value, V, _ = solve_sdp_relaxation(M_proj, n, min_eig=min_eig)
            values.append([prob_value, np.max(np.diag(V.T @ M_proj @ V))])
            print("SDP relaxation solved")

            if avg_case:
                # ==== Optimize Average Case Risk (ACR) ====
                # Get gradient for ACR optimization
                _, AG_m, _ = para4measure(ms, L)
                
                # 2. Find the step to take
                if grad:
                    # Optimize ACR using projected gradient descent
                    step_val = optimize_with_gradient_descent(A, AG_m, k)
                    
                    # In case all the edges are deleted - then stop
                    if np.sum(np.round(A - step_val, 5)) == 0:
                        break
                    
                    A = A - step_val  # Network update
                else:
                    # Optimize ACR using coordinate descent
                    A, modified = optimize_acr_with_coordinate_descent(A, AG_m, step_size)
                    if not modified:
                        break
            else:
                # ==== Optimize Worst Case Risk (WCR) ====
                # Project V to have zero mean
                V = (I - J/n) @ V
                
                # 2. Optimize the network based on worst-case vectors
                if grad:
                    # Optimize WCR using projected gradient descent
                    print("Gradient descent for WCR")
                    WG_m = worst_case_risk_gradient(ms, L, V)
                    print("Gradient matrix calculated")
                    step_val = optimize_with_gradient_descent(A, WG_m, k)
                    print("Step value calculated")
                    A = A - step_val
                else:
                    # Optimize WCR using coordinate descent
                    XX = worst_case_risk_gradient(ms, L, V)
                    XX_new = np.zeros((n, n))
                    
                    # Calculate edge changes impact
                    for k1 in range(n):
                        for k2 in range(n):
                            XX_new[k1, k2] = XX[k1, k1] + XX[k2, k2] - XX[k1, k2] - XX[k2, k1]
                    
                    XX = XX_new
                    
                    # Check 0<=A+step<=1
                    for k1 in range(n):
                        for k2 in range(n):
                            if XX[k1, k2] > 0 and A[k1, k2] == 0:
                                XX[k1, k2] = 0
                            elif XX[k1, k2] < 0 and A[k1, k2] == 1:
                                XX[k1, k2] = 0
                    
                    # Find maximum absolute value in upper triangular part
                    triu_indices = np.triu_indices(n, 1)
                    XX_triu = XX[triu_indices]
                    max_idx = np.argmax(np.abs(XX_triu))
                    iii, jjj = triu_indices[0][max_idx], triu_indices[1][max_idx]
                    
                    if XX[iii, jjj] > 0:
                        # Delete edge
                        A[iii, jjj] = 0
                        A[jjj, iii] = 0
                    else:
                        # Add edge
                        A[iii, jjj] = 1
                        A[jjj, iii] = 1
            
            # Update Laplacian and store metrics
            L = np.diag(np.sum(A, axis=1)) - A
            acrs[i], _, _ = para4measure(ms, L)
            conflicts[i, :] = actual_conflict(ms, L)
            
            if plot:
                # Display the current network
                plt.figure()
                plt.imshow(A)
                plt.colorbar()
                plt.title(f"Iteration {i+1}")
                plt.pause(0.01)
                plt.close()

        # Add the final wcr value to the list
        _, _, M = para4measure(ms, L)
        M_proj = (I - J/n) @ M @ (I - J/n)
        prob_value, V, _ = solve_sdp_relaxation(M_proj, n, min_eig=min_eig)
        values.append([prob_value, np.max(np.diag(V.T @ M_proj @ V))])

        # Prepare return values
        Opt_A = A
        acr = np.concatenate(([acr0], acrs[:i+1]))
        wcr = np.array(values)[:, 1]
        all_conflicts = np.vstack((conflict0, conflicts[:i+1]))
        
        return Opt_A, acr, wcr, all_conflicts
    
    finally:
        # Restore original environment variables
        for var in thread_vars:
            if var in original_env:
                os.environ[var] = original_env[var]
            elif var in os.environ:
                del os.environ[var]
