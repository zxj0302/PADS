#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys

# Handle imports both when run as a module and as a script
if __name__ == "__main__" and __package__ is None:
    # Add parent directory to path to allow relative imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ConflictRisk.para4measure import para4measure
else:
    from .para4measure import para4measure

def actual_conflict(m, L):
    """
    Description: Compute the actual conflict for the three different internal opinion vectors:
                 s1 - the random one consists of -1 and 1
                 s2 - corresponds to the 10th smallest eigenvalue (low-frequency on the graph)
                 s3 - corresponds to the 10th largest eigenvalue (high-frequency on the graph)
    
    Parameters:
    -----------
    m : int
        Represents the measure
    L : numpy.ndarray
        The Laplacian matrix of the current network
    
    Returns:
    --------
    conflict : numpy.ndarray
        The real conflict for the three internal opinion vectors
    """
    n = L.shape[0]
    I = np.eye(n)
    e = np.ones(n)
    J = np.outer(e, e)
    
    conflict = np.zeros(3)
    _, _, M_conf = para4measure(m, L)
    M_conf = (I - J/n) @ M_conf @ (I - J/n)
    
    # Random vector of -1 and 1
    s1 = np.sign(np.random.randn(n))
    
    # Eigenvectors for low and high frequency
    val, vec = np.linalg.eig(L)
    # Sort eigenvalues and eigenvectors
    idx = val.argsort()
    val = val[idx]
    vec = vec[:, idx]
    
    s2 = np.sign(vec[:, 9])  # 10th smallest eigenvalue (0-indexed)
    s3 = np.sign(vec[:, n-11])  # 10th largest eigenvalue (0-indexed)
    
    conflict[0] = s1.T @ M_conf @ s1
    conflict[1] = s2.T @ M_conf @ s2
    conflict[2] = s3.T @ M_conf @ s3
    
    return conflict
