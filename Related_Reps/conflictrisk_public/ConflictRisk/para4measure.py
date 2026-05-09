#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv

def para4measure(m, L):
    """
    Description: Used for getting parameters for different measures at different
    time steps for average case internal opinion vector s
    
    Parameters:
    -----------
    m : int
        Represents the measure
        1: internal conflict
        2: external conflict
        3: controversy
        4: resistance
    L : numpy.ndarray
        The Laplacian matrix of the current network
    
    Returns:
    --------
    risk_m : float
        Current ACR w.r.t. measure m
    ag_m : numpy.ndarray
        The Gradient of the ACR for current network
    M : numpy.ndarray
        The current middle matrix for measure m
    """
    n = L.shape[0]
    I = np.eye(n)
    
    if m == 1:
        # internal conflict
        risk_m = np.trace(inv(L + I) @ L @ L @ inv(L + I))
        ag_m = 2 * inv(L + I) @ inv(L + I) - 2 * inv(L + I) @ inv(L + I) @ inv(L + I)
        M = inv(L + I) @ L @ L @ inv(L + I)
    elif m == 2:
        # external conflict
        risk_m = np.trace(inv(L + I) @ L @ inv(L + I))
        ag_m = -inv(L + I) @ inv(L + I) + 2 * inv(L + I) @ inv(L + I) @ inv(L + I)
        M = inv(L + I) @ L @ inv(L + I)
    elif m == 3:
        # controversy
        risk_m = np.trace(inv(L + I) @ inv(L + I))
        ag_m = -2 * inv(L + I) @ inv(L + I) @ inv(L + I)
        M = inv(L + I) @ inv(L + I)
    else:
        # resistance
        risk_m = np.trace(inv(L + I))
        ag_m = -inv(L + I) @ inv(L + I)
        M = inv(L + I)
    
    return risk_m, ag_m, M
