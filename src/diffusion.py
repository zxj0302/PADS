import os

import numpy as np
import networkx as nx
from numba import njit, prange
from joblib import Parallel, delayed
from tqdm import tqdm
import math
import json


def preprocess_graph(G, attr_name='polarity'):
    num_nodes = G.number_of_nodes()
    node_indices = {node: idx for idx, node in enumerate(G.nodes())}
    inverse_node_indices = {idx: node for node, idx in node_indices.items()}

    # Extract node attributes
    attr_values = np.zeros(num_nodes, dtype=np.float64)
    for node, idx in node_indices.items():
        attr_values[idx] = G.nodes[node][attr_name]

    # Create adjacency list
    adjacency_list = [[] for _ in range(num_nodes)]
    degrees = np.zeros(num_nodes, dtype=np.int32)
    for node in G.nodes():
        idx = node_indices[node]
        neighbors = list(G.neighbors(node))
        adjacency_list[idx] = [node_indices[neigh] for neigh in neighbors]
        degrees[idx] = len(neighbors)

    # Convert adjacency_list to a flattened array and an index pointer array
    max_degree = max(len(neighs) for neighs in adjacency_list)
    adjacency_array = np.full((num_nodes, max_degree), -1, dtype=np.int32)
    for idx, neighbors in enumerate(adjacency_list):
        adjacency_array[idx, :len(neighbors)] = neighbors

    return {
        'num_nodes': num_nodes,
        'node_indices': node_indices,
        'inverse_node_indices': inverse_node_indices,
        'attr_values': attr_values,
        'adjacency_array': adjacency_array,
        'degrees': degrees,
        'max_degree': max_degree
    }


@njit(parallel=True)
def compute_reach_probabilities_numba(
        source_idx, num_nodes, adjacency_array, max_degree, degrees, attr_values,
        info_stance, theta, epsilon, deg_enc, max_iterations, tol
):
    reach_probs = np.zeros(num_nodes, dtype=np.float64)
    reach_probs[source_idx] = 1.0
    prev_probs = np.zeros(num_nodes, dtype=np.float64)

    for iteration in range(max_iterations):
        # Copy current reach_probs to prev_probs
        prev_probs[:] = reach_probs[:]

        # Update reach_probs for all nodes except source
        for v in prange(num_nodes):
            if v == source_idx:
                continue
            accumulated_prob = 0.0
            deg_sum = 0.0
            nei_prob_sum = 0.0
            for i in range(max_degree):
                u = adjacency_array[v, i]
                if u == -1:
                    break
                p_uv = (theta * (2.0 - math.fabs(info_stance - attr_values[v])) +
                        (1.0 - theta) * (2.0 - math.fabs(attr_values[u] - attr_values[v]))) / 2.0 + epsilon
                accumulated_prob += prev_probs[u] * p_uv * degrees[u]
                deg_sum += degrees[u]
                nei_prob_sum += prev_probs[u]
            if deg_sum > 0.0:
                p_uv_final = (accumulated_prob / deg_sum) * (2 * deg_enc / (1.0 + math.exp(-nei_prob_sum)))
                p_uv_final = min(p_uv_final, 1.0)
                reach_probs[v] = p_uv_final
            else:
                reach_probs[v] = 0.0  # No neighbors

        # Check for convergence - compute changes in a separate array
        changes = np.zeros(num_nodes, dtype=np.float64)
        for v in prange(num_nodes):
            changes[v] = math.fabs(reach_probs[v] - prev_probs[v])

        # Find maximum change using a single-threaded operation
        max_change = 0.0
        for v in range(num_nodes):
            if changes[v] > max_change:
                max_change = changes[v]
        if max_change < tol:
            break

    return reach_probs


def diffusion_optimized(
        G, attr_name='polarity', theta=0.95, epsilon=0, deg_enc=1.2,
        max_iterations=1000, tol=1e-4, n_jobs=-1
):
    # Preprocess the graph
    graph_data = preprocess_graph(G, attr_name=attr_name)
    num_nodes = graph_data['num_nodes']
    node_indices = graph_data['node_indices']
    inverse_node_indices = graph_data['inverse_node_indices']
    attr_values = graph_data['attr_values']
    adjacency_array = graph_data['adjacency_array']
    degrees = graph_data['degrees']
    max_degree = graph_data['max_degree']

    # Identify positive and negative nodes
    pos_nodes = [node for node in G.nodes if G.nodes[node]['pos_com']]
    neg_nodes = [node for node in G.nodes if G.nodes[node]['neg_com']]
    pos_indices = [node_indices[node] for node in pos_nodes]
    neg_indices = [node_indices[node] for node in neg_nodes]

    # Define a wrapper function for Numba-accelerated computation
    def compute_rp(source_idx):
        info_stance = attr_values[source_idx]
        rp = compute_reach_probabilities_numba(
            source_idx, num_nodes, adjacency_array, max_degree, degrees, attr_values,
            info_stance, theta, epsilon, deg_enc, max_iterations, tol
        )
        return rp

    # Compute reach probabilities in parallel
    pos_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_rp)(source_idx) for source_idx in tqdm(pos_indices, desc="Processing positive nodes")
    )
    neg_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_rp)(source_idx) for source_idx in tqdm(neg_indices, desc="Processing negative nodes")
    )

    # Convert results to NumPy arrays for efficient processing
    pos_rps = np.array(pos_results)  # Shape: (num_pos, num_nodes)
    neg_rps = np.array(neg_results)  # Shape: (num_neg, num_nodes)

    # map back to nodes
    pos_rps_nodes = {node: [0 for _ in range(num_nodes)] for node in pos_nodes}
    neg_rps_nodes = {node: [0 for _ in range(num_nodes)] for node in neg_nodes}
    for i, node in enumerate(pos_nodes):
        for j, p in enumerate(pos_rps[i]):
            pos_rps_nodes[node][inverse_node_indices[j]] = p
    for i, node in enumerate(neg_nodes):
        for j, p in enumerate(neg_rps[i]):
            neg_rps_nodes[node][inverse_node_indices[j]] = p

    return pos_rps_nodes, neg_rps_nodes


def my_diffusion(graph_path, save_path,
                 attr_name='polarity',
                 theta=0.95,
                 epsilon=0,
                 deg_enc=0.7,
                 max_iterations=10000,
                 tol=1e-8,
                 n_jobs=-1):
    G = nx.read_gml(graph_path)
    # change node labels to integers
    G = nx.convert_node_labels_to_integers(G)
    methods = ['maxflow_cpp_udsp', 'maxflow_cpp_wdsp', 'node2vec_gin', 'pads_cpp']
    for node in G.nodes:
        G.nodes[node]['pos_com'] = False
        G.nodes[node]['neg_com'] = False
        for m in methods:
            G.nodes[node]['pos_com'] = G.nodes[node]['pos_com'] or G.nodes[node][m] == 1
            G.nodes[node]['neg_com'] = G.nodes[node]['neg_com'] or G.nodes[node][m] == -1

    pos_reach, neg_reach = diffusion_optimized(
        G,
        attr_name=attr_name,
        theta=theta,
        epsilon=epsilon,
        deg_enc=deg_enc,
        max_iterations=max_iterations,
        tol=tol,
        n_jobs=n_jobs
    )

    # save the results into a json file
    if save_path is not None:
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(save_path, 'w') as f:
            json.dump({'pos': pos_reach, 'neg': neg_reach}, f)
    return pos_reach, neg_reach


def mean_diffusion(graph_path, diffusion_path, method_name, filter=False):
    G = nx.read_gml(graph_path)
    diffusion = json.load(open(diffusion_path))
    pos_nodes = [node for node in G.nodes if ((G.nodes[node][method_name] == 1) and (not filter or G.nodes[node][
        'polarity'] >= 0))]
    neg_nodes = [node for node in G.nodes if ((G.nodes[node][method_name] == -1) and (not filter or G.nodes[node][
        'polarity'] <= 0))]
    pos_reach = diffusion['pos']
    neg_reach = diffusion['neg']
    pos_reach_pos = {}
    pos_reach_neg = {}
    neg_reach_pos = {}
    neg_reach_neg = {}
    for pos in pos_nodes:
        pos_reach_pos[pos] = sum([pos_reach[pos][int(p)] for p in pos_nodes]) / len(pos_nodes)
        pos_reach_neg[pos] = sum([pos_reach[pos][int(n)] for n in neg_nodes]) / len(neg_nodes)
    for neg in neg_nodes:
        neg_reach_pos[neg] = sum([neg_reach[neg][int(p)] for p in pos_nodes]) / len(pos_nodes)
        neg_reach_neg[neg] = sum([neg_reach[neg][int(n)] for n in neg_nodes]) / len(neg_nodes)
    popm = sum(pos_reach_pos.values()) / len(pos_reach_pos)
    ponm = sum(pos_reach_neg.values()) / len(pos_reach_neg)
    nopm = sum(neg_reach_pos.values()) / len(neg_reach_pos)
    nonm = sum(neg_reach_neg.values()) / len(neg_reach_neg)
    print(f'{method_name}: popm({popm}), ponm({ponm}), nonpm({nopm}), nonpm({nonm})')
    return popm, ponm, nopm, nonm
