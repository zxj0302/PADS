# Note: this is not the original file, but a optimized(efficiency and readability) version.
import os
# import sys
# import copy
# import random
import argparse
import itertools
# from time import time
# from collections import defaultdict

import pickle
import numpy as np
import networkx as nx
import multiprocessing as mp
from scipy.sparse import dia_matrix, diags
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
import psutil

# import parallel_walks 
# import parallel_addition 
# import parallel_centrality
# from algo import Algorithms
try:
    from .load_data import LoadData
except ImportError:
    from RepBublik.load_data import LoadData
# from random_walks import chunk_random_walk
# from utils import load_bubble_diameter, get_bad_and_good_nodes, get_centralities, sort_edges
from tqdm import tqdm

def get_tok_nodes(G, node_id_matrix, nodes, perc_k):    
    red_degrees = {}
    for i in nodes:
        red_degrees[i] = G.degree(i)
    k = int(len(red_degrees)*perc_k)
    red_topk = [node_id_matrix[i] for i,j in sorted(red_degrees.items(), key=lambda x: x[1], reverse=True)][:k]

    return red_topk

def get_candidate_edges(G, red_topk, blue_topk):
    candidate_edges = list(itertools.product(red_topk, blue_topk))
    # Add reverse edges if directed
    if G.is_directed():
        # Add reverse edges
        candidate_edges = candidate_edges + [(j,i) for i,j in candidate_edges]
    return candidate_edges

def color_chain(A, node_id_matrix, nodes_subset, alpha=0.95, delta_col_u=None, u_modified=None):
    """
    Computes the stationary distribution for a given subset of nodes using power iteration.
    Can incorporate a modification to a single column 'u_modified' of A represented by 'delta_col_u'.
    """
    if not isinstance(A, csc_matrix):
        A = csc_matrix(A) # Ensure A is CSC for efficient column operations if needed later

    num_nodes = A.shape[0]
    subset_idx = [node_id_matrix[i] for i in nodes_subset]

    r_subset = np.zeros((num_nodes, 1))
    if len(subset_idx) > 0:
        r_subset[subset_idx] = 1.0 / len(subset_idx)
    r_subset = csc_matrix(r_subset)

    e_subset = np.zeros((num_nodes, 1))
    if len(subset_idx) > 0:
        e_subset[subset_idx] = 1.0 / len(subset_idx)
    e_subset = csc_matrix(e_subset)

    old = r_subset
    diff = 1.0
    i = 0
    # Increased max iterations slightly and using a tolerance based approach
    while diff > 1e-5 and i < 250:
        # Calculate A.dot(old) or A'.dot(old)
        mat_dot_old = A.dot(old)
        if delta_col_u is not None and u_modified is not None:
            # Add the effect of the modification: delta_col_u * old[u_modified]
            # Ensure old[u_modified, 0] is treated as a scalar
            old_u_val = old[u_modified, 0]
            if isinstance(old_u_val, csc_matrix): # Extract scalar if it's sparse
                 old_u_val = old_u_val.toarray()[0,0]
            # Perform sparse vector * scalar + sparse vector addition
            mat_dot_old = mat_dot_old + (delta_col_u * old_u_val)

        # Perform the PageRank update step
        new = alpha * mat_dot_old + (1 - alpha) * e_subset

        # Normalize the vector
        sum_new = new.sum()
        if sum_new > 1e-9: # Avoid division by zero
             new = new / sum_new
        else: # Handle case of zero vector (e.g., disconnected components)
             new = e_subset # Reset to teleportation vector

        # Calculate difference (using sum of absolute differences)
        # Ensure subtraction works correctly between sparse matrices
        diff = np.sum(np.abs((new - old).toarray()))

        old = new
        i += 1

    # print(f"Color chain iterations: {i}, final diff: {diff}") # Optional: for debugging convergence
    return old # Return the final stationary distribution


def compute_controversy_score(A, node_id_matrix, topic_obj, red_topk, blue_topk, perc=10, alpha=0.95, delta_col_u=None, u_modified=None):
    """
    Computes the Random Walk Controversy (RWC) score.
    Can compute the score for a modified matrix A' by passing delta_col_u and u_modified.
    """
    # print('Computing RWC...')
    # Ensure A is CSC for color_chain, although color_chain checks internally now
    A_csc = csc_matrix(A)

    # Pass the modification details down to color_chain
    r_red_final = color_chain(A_csc, node_id_matrix, topic_obj.red_nodes, alpha=alpha, delta_col_u=delta_col_u, u_modified=u_modified)
    r_blue_final = color_chain(A_csc, node_id_matrix, topic_obj.blue_nodes, alpha=alpha, delta_col_u=delta_col_u, u_modified=u_modified)

    num_nodes = A.shape[0]
    c_red = np.zeros((num_nodes, 1))
    if len(red_topk) > 0:
        c_red[red_topk] = 1 # No normalization needed here based on original code
    c_blue = np.zeros((num_nodes, 1))
    if len(blue_topk) > 0:
        c_blue[blue_topk] = 1 # No normalization needed here

    # Difference vector c_red - c_blue (dense is fine here)
    c_diff = c_red - c_blue
    # Convert to sparse row vector for dot product: (c_red - c_blue)^T
    diff_c_sparse_row = csr_matrix(c_diff.T)

    # Calculate RWC = (c_red - c_blue)^T . (r_red - r_blue)
    # Ensure r_red_final and r_blue_final are compatible for subtraction (e.g., both CSC)
    r_diff = r_red_final - r_blue_final

    RWC = diff_c_sparse_row.dot(r_diff)

    # RWC should be a 1x1 sparse matrix, extract the scalar value
    if RWC.shape == (1, 1):
        return RWC[0, 0]
    else:
        # Handle unexpected shape, maybe return NaN or raise error
        print("Warning: RWC calculation resulted in unexpected shape:", RWC.shape)
        return np.nan


def update_rwc(chunk, RWC, A, node_id_matrix, topic_obj, red_topk, blue_topk, perc=20, alpha=0.95):
    """
    Calculates the change in RWC for a chunk of candidate edges (u, v)
    by calculating the effect of adding the edge without copying A.
    """
    initial_RWC = RWC
    diff_RWC = {}

    # Ensure A is CSC for efficient column slicing and consistent data access
    if not isinstance(A, csc_matrix):
        A = csc_matrix(A)

    num_nodes = A.shape[0]

    for u, v in tqdm(chunk, desc=f'Updating RWC', total=len(chunk)):
        # Calculate the change vector delta_col_u = A'[:, u] - A[:, u]
        # where A' is the matrix *if* edge (u, v) were added and column u re-normalized.

        # Get original neighbors and probabilities for column u
        start_ptr = A.indptr[u]
        end_ptr = A.indptr[u+1]
        original_indices = A.indices[start_ptr:end_ptr]
        original_data = A.data[start_ptr:end_ptr]
        num_original_neighbors = len(original_indices)

        # Calculate the new uniform probability after adding edge (u,v)
        new_prob = 1.0 / (num_original_neighbors + 1)

        # Calculate the difference vector data (dense first for simplicity)
        delta_data_dense = np.zeros(num_nodes)
        # Change for original neighbors: new_prob - old_prob
        delta_data_dense[original_indices] = new_prob - original_data
        # Change for the new neighbor v: new_prob (assuming A[v, u] was 0)
        # If v was already a neighbor, its entry in delta_data_dense is updated correctly:
        # (new_prob - old_prob_v) + new_prob. This seems wrong.
        # Let's rethink:
        # A'[i, u] = new_prob for i in original_indices
        # A'[v, u] = new_prob
        # delta[i, u] = A'[i, u] - A[i, u]
        delta_data_dense = np.zeros(num_nodes)
        delta_data_dense[original_indices] = new_prob - original_data
        # If v was NOT an original neighbor, A[v, u] = 0, so delta[v, u] = new_prob - 0 = new_prob
        # If v WAS an original neighbor, A[v, u] = old_prob_v, so delta[v, u] = new_prob - old_prob_v.
        # The line `delta_data_dense[original_indices] = new_prob - original_data` already handles the case where v is in original_indices.
        # We only need to add the `new_prob` if v was *not* in original_indices.
        if v not in original_indices:
             delta_data_dense[v] += new_prob # Use += just in case, though it should be 0

        # Convert the dense difference vector to a sparse column matrix
        delta_col_u = csc_matrix(delta_data_dense.reshape(-1, 1))

        # Compute the new RWC score using the original A and the calculated delta
        new_rwc = compute_controversy_score(A, node_id_matrix, topic_obj, red_topk, blue_topk, perc, alpha, delta_col_u=delta_col_u, u_modified=u)

        # Store the difference
        diff_RWC[(u, v)] = initial_RWC - new_rwc

    return diff_RWC

def _get_chunks(num_nodes, list_nodes, parallel=False):
    """
    """
    
    n_proc = mp.cpu_count()
    print('Number of cores: ', n_proc)
    if not parallel:
        # p = psutil.Process(os.getpid())
        # p.cpu_affinity([0, 1])
        n_proc = 1
    max_nodes = num_nodes
    nodes = list_nodes[:max_nodes]
    if len(nodes) <= 40:
        n_proc = int(len(nodes)/1)
        n = int(len(nodes)/n_proc)
        return n_proc, [np.array(nodes[i:i + n]) for i in range(0, len(nodes), n)]
    
    n = int(len(nodes)/n_proc)
    
    chunks = [np.array(nodes[i:i + n]) for i in range(0, len(nodes), n)]
    
    return n_proc, chunks


def parallelization(RWC, A, node_id_matrix, topic_obj, red_topk, blue_topk, candidate_edges, perc=20, alpha=0.95, parallel=False):
        """
        """
        
        n_proc, chunks = _get_chunks(len(candidate_edges), candidate_edges, parallel)
        print('Number of cores used: ', n_proc)
        # print('Number of chunks: ', len(chunks))

        if n_proc > 1:
            print('Parallelizing...')
            with mp.Pool(processes=n_proc) as pool:
                proc_results = [pool.apply_async(update_rwc,
                                                args=(chunk, RWC, A, node_id_matrix, topic_obj, red_topk, blue_topk, perc, alpha, ))
                                for index_chunk, chunk in enumerate(chunks)]
            result_chunks = [r.get() for r in proc_results]#
        else:
            print('Not parallelizing...')
            result_chunks = [update_rwc(chunk, RWC, A, node_id_matrix, topic_obj, red_topk, blue_topk, perc, alpha) for chunk in chunks]
        
        return result_chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the entire/partial pipeline of FairRandomWalks experiments.')
    parser.add_argument('-proc', type=str, default='radius', 
                        help='Part of pipeline to execute: diameter to compute the bubble diameter of all partitions, addition to compute new bubble diameter')
    #parser.add_argument('-algo', type=str, default='baseline', help='If -proc is addition, choose the algo for addition')
    parser.add_argument('-topic', type=str, help='Graph to analyze')
    parser.add_argument('-unweighted', type=str, default='false', help='Weighted or unweighted graph')

    #parser.add_argument('-topic', type=str, help='Graph to analyze')
    args = parser.parse_args()

    topic = args.topic

    if args.unweighted == 'false':
        # Recall the graph of interest
        politics = LoadData(args.topic, uniform=False)
    elif args.unweighted == 'true':
        politics = LoadData(args.topic, uniform=True)
    id_color = politics.id_color
    dictionary_weights = politics.dictionary_weights
    id_name = politics.id_name
    G = politics.G
    edges_updated = set(list(G.edges()))

    # Indeces for the adj matrix
    node_id_matrix = {n:i for i,n in enumerate(list(G.nodes()))}
    id_matrix_node = {j:i for i,j in node_id_matrix.items()}

    # Define adj matrix
    adj_mat = nx.adjacency_matrix(G)
    # Calculate out-degrees (sum of rows in adjacency matrix)
    out_degrees = np.array(adj_mat.sum(axis=1)).flatten()
    # Avoid division by zero for isolated nodes
    inv_out_degrees = np.reciprocal(out_degrees.astype(float), where=out_degrees > 0)
    # Create diagonal matrix of inverse out-degrees
    d_inv = diags(inv_out_degrees)
    # Calculate column-stochastic transition matrix A = D^-1 @ Adj
    # Note: The original code did A = Adj.T @ D^-1 which results in A[i,j] = prob(j->i)
    # Let's stick to the original calculation for consistency:
    A_csr = nx.adjacency_matrix(G) # Usually CSR
    row_sums = np.array(A_csr.sum(axis=1)).flatten()
    inv_row_sums = np.reciprocal(row_sums.astype(float), where=row_sums > 0)
    d_inv_row = diags(inv_row_sums)
    # A = A.T.dot(d) -> Transition matrix where A[i, j] = prob(j -> i)
    A_csc = A_csr.T.dot(d_inv_row).tocsc() # Ensure CSC format

    # Order nodes labels and colors
    labels = np.array([j for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])
    color_nodes = np.array([id_color[j] for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])


    red_topk = get_tok_nodes(G, node_id_matrix, politics.red_nodes, perc_k=10)
    blue_topk = get_tok_nodes(G, node_id_matrix, politics.blue_nodes, perc_k=10)
    RWC = compute_controversy_score(A_csc, node_id_matrix, politics, red_topk, blue_topk, perc=10, alpha=0.95)


    candidate_edges = get_candidate_edges(G, politics.red_nodes, politics.blue_nodes, node_id_matrix, perc_k=10)#[:10]
    candidate_edges = [(i,j) for i,j in candidate_edges if (id_matrix_node[i], id_matrix_node[j]) not in G.edges()]
    #candidate_edges = [(i,j) for i,j in candidate_edges if id_color[id_matrix_node[i]] == c]
    #print((len(candidate_edges)))

    # Changed by Xiangju
    # candidate_edges = random.sample(candidate_edges, min(10000, len(candidate_edges)))
    print('LEN CANDIDATES: ', len(candidate_edges))


    candidate_to_save = []
    result_chunks = parallelization(RWC, A_csc, node_id_matrix, politics, red_topk, blue_topk, candidate_edges, perc=10, alpha=0.95)

    tops = []
    for i in range(len(result_chunks)):
        tops += sorted(result_chunks[i].items(), key=lambda x: x[1], reverse=True)
    candidate_edges_ = sorted(tops, key=lambda x: x[1], reverse=True)[:min(len(tops), 2000)]#[0]


    candidate_edges_ = [(i[0][0], i[0][1],1) for i in candidate_edges_]
    with open('rov_candidate_edges.pickle', 'wb') as f:
        pickle.dump(candidate_edges_, f)
    print(candidate_edges_)