#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for ConflictRisk package
"""

import numpy as np
import scipy.io as sio
import os

def load_matlab_matrix(filepath):
    """
    Load a MATLAB matrix from a .mat file
    
    Parameters:
    -----------
    filepath : str
        Path to the .mat file
        
    Returns:
    --------
    matrix : numpy.ndarray
        The loaded matrix
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        mat_contents = sio.loadmat(filepath)
        # Try to find the first matrix in the file
        for key, value in mat_contents.items():
            if key.startswith('__'):  # Skip metadata
                continue
            if isinstance(value, np.ndarray) and len(value.shape) == 2:
                return value
        
        raise ValueError(f"No suitable matrix found in file: {filepath}")
    except Exception as e:
        raise IOError(f"Error loading matrix from {filepath}: {str(e)}")

def save_results_to_matlab(filepath, OptA, acr, wcr, conflicts):
    """
    Save optimization results to a MATLAB .mat file
    
    Parameters:
    -----------
    filepath : str
        Path to save the .mat file
    OptA : numpy.ndarray
        Optimized adjacency matrix
    acr : numpy.ndarray
        Average-case conflict risks
    wcr : numpy.ndarray
        Worst-case conflict risks
    conflicts : numpy.ndarray
        Actual conflicts for three internal opinion vectors
    """
    save_dict = {
        'OptA': OptA,
        'acr': acr,
        'wcr': wcr,
        'conflicts': conflicts
    }
    
    sio.savemat(filepath, save_dict)
    
def convert_network_to_nx(A):
    """
    Convert adjacency matrix to networkx graph
    
    Parameters:
    -----------
    A : numpy.ndarray
        Adjacency matrix
        
    Returns:
    --------
    G : networkx.Graph
        NetworkX graph object
    """
    try:
        import networkx as nx
        G = nx.from_numpy_array(A)
        return G
    except ImportError:
        print("NetworkX not installed. Install with 'pip install networkx'")
        return None

def visualize_network(A, title="Network"):
    """
    Visualize network using networkx and matplotlib
    
    Parameters:
    -----------
    A : numpy.ndarray
        Adjacency matrix
    title : str
        Title for the plot
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.from_numpy_array(A)
        plt.figure(figsize=(10, 8))
        
        # Get position layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')
        
        # Use edge weights for thickness
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        # Draw each edge with its own width
        for (u, v, w) in G.edges(data='weight'):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w*2, alpha=0.7)
        
        # Add node labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title(title)
        plt.axis('off')
        plt.show()
    except ImportError:
        print("NetworkX not installed. Install with 'pip install networkx'")

def create_random_network(n=20, p=0.3, min_weight=0, stable=False):
    """
    Create a random network with n nodes and edge probability p
    
    Parameters:
    -----------
    n : int
        Number of nodes
    p : float
        Edge probability
    min_weight : float
        Minimum edge weight (default is 0)
    stable : bool
        If True, creates a more numerically stable network for optimization
        
    Returns:
    --------
    A : numpy.ndarray
        Adjacency matrix of the random network
    """
    if stable:
        # Create a smaller, denser, more stable network
        n = min(n, 10)  # Limit size for stability
        p = max(p, 0.5)  # Ensure sufficient density
        min_weight = 0.1  # Add minimum weight to all edges
    
    # Create a random adjacency matrix with probability p
    A = np.random.random((n, n)) < p
    A = A.astype(float)
    
    # Make it symmetric (undirected)
    A = (A + A.T) / 2
    A[A > 0] = 1
    
    # Add minimum edge weight if specified
    if min_weight > 0:
        A[A > 0] = A[A > 0] + min_weight
        A[A > 1] = 1  # Cap at 1
    
    # No self-loops
    np.fill_diagonal(A, 0)
    
    # Ensure the network is connected (for stability)
    if stable:
        # Make sure all nodes have at least one connection
        for i in range(n):
            if np.sum(A[i, :]) == 0:
                # Connect to a random node
                j = np.random.randint(0, n)
                if i != j:  # Avoid self-loops
                    A[i, j] = A[j, i] = 1
    
    return A

def create_and_save_sample_network(output_file='A.mat', n=500, p=0.01):
    """
    Create a sample random network and save it to a .mat file
    
    Parameters:
    -----------
    output_file : str
        Path to save the .mat file
    n : int
        Number of nodes
    p : float
        Edge probability
        
    Returns:
    --------
    A : numpy.ndarray
        The created network
    """
    # Create a sample network
    A = create_random_network(n, p)
    
    # Save it as a .mat file
    sio.savemat(output_file, {'A': A})
    
    return A
