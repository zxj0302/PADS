#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv

def worst_case_risk_gradient(m, L, S):
    """
    Description: Used for getting the gradients for the WCR for current
    network L w.r.t different conflict measures
    
    Parameters:
    -----------
    m : int
        Represents the measure
    L : numpy.ndarray
        The Laplacian matrix of the current network
    S : numpy.ndarray
        A set of worst case internal opinions (size = dim) for the 
        current network, which is the result of the max-cut problem in the previous step
    
    Returns:
    --------
    WGm : numpy.ndarray
        The Gradient of the WCR for current network
    """
    n = L.shape[0]
    I = np.eye(n)
    
    if m == 1:
        # Internal conflict
        WGm = L @ inv(L + I) @ inv(L + I) @ S @ S.T @ inv(L + I) + \
              inv(L + I) @ S @ S.T @ inv(L + I) @ inv(L + I) @ L
    elif m == 2:
        # External conflict
        WGm = inv(L + I) @ inv(L + I) @ S @ S.T @ inv(L + I) @ inv(L + I) - \
              L @ inv(L + I) @ inv(L + I) @ S @ S.T @ inv(L + I) @ inv(L + I) @ L
    elif m == 3:
        # Controversy
        WGm = -inv(L + I) @ S @ S.T @ inv(L + I) @ inv(L + I) - \
               inv(L + I) @ inv(L + I) @ S @ S.T @ inv(L + I)
    else:
        # Resistance
        WGm = -inv(L + I) @ S @ S.T @ inv(L + I)
    
    return WGm
