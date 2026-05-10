# The CD algorithm from 'Towards Consensus:  Reducing Polarization by Perturbing Social Networks'
# Code from their repository, while collated for convenience
# Changes made to support weighted graphs
import numpy as np
import networkx as nx
import itertools
from tqdm import tqdm

class CD:
    def __init__(self, G, k, s=None, innate_opinion_attr='polarity', weight_attr='weight', max_weight=1.0):
        # Store original node labels and create mapping
        self.original_nodes = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.original_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        # Create a copy of G with continuous node labels
        self.G = nx.relabel_nodes(G.copy(), self.node_to_idx)
        self.k = k
        if s is None:
            if innate_opinion_attr is None:
                raise ValueError("Either s or attr must be provided.")
            self.s = np.array([G.nodes[node][innate_opinion_attr] for node in self.original_nodes])
        else:
            self.s = s
        self.n = len(self.G.nodes())
        self.weight_attr = weight_attr
        self.max_weight = max_weight
        self.compute_weight()

    def compute_weight(self):
        # Compute the weight of each edge as the 'polarity' difference of the two endpoints
        for u, v in self.G.edges():
            if self.weight_attr not in self.G[u][v]:
                self.G[u][v][self.weight_attr] = (2-np.abs(self.s[u] - self.s[v])) / 2

    def coordinate_descent(self):
        modified_edges = []
        for _ in tqdm(range(self.k), desc="Coordinate Descent"):
            modified_edge = self.opt_max_grad_fast()
            # Convert edge indices back to original node labels
            original_edge = (self.idx_to_node[modified_edge[0]], self.idx_to_node[modified_edge[1]])
            modified_edges.append(original_edge)
        
        # Convert the graph back to original node labels
        result_graph = nx.relabel_nodes(self.G, self.idx_to_node)
        return result_graph, modified_edges

    def opt_max_grad_fast(self):
        # Calculate expressed opinions z = (I + L)^-1 s
        I_plus_L = np.identity(self.n) + nx.laplacian_matrix(self.G, weight=self.weight_attr).todense()
        inv_IpL = np.linalg.inv(I_plus_L)
        z = np.dot(inv_IpL, self.s)
        z_tilde = z - z.mean()
        
        # Pre-compute gradient matrix
        grad_matrix = 2 * np.outer(inv_IpL @ z_tilde, z_tilde)
        
        best_edge = (0, 0)  # Initialize with valid indices
        best_score = -np.inf
        
        # Process all possible pairs (both existing edges and non-edges)
        for i, j in itertools.combinations(range(self.n), 2):
            # Get current weight (0 if non-edge)
            current_weight = 0
            if self.G.has_edge(i, j):
                current_weight = self.G[i][j].get(self.weight_attr, 0)
            
            # Only consider if weight can be increased
            if current_weight < self.max_weight:
                # Efficient computation of gradient value
                grad_val = grad_matrix[i,i] + grad_matrix[j,j] - grad_matrix[i,j] - grad_matrix[j,i]
                score = (self.max_weight - current_weight) * grad_val
                
                if score > best_score:
                    best_score = score
                    best_edge = (i, j)
        
        # Add or update the best edge
        if not self.G.has_edge(best_edge[0], best_edge[1]):
            self.G.add_edge(best_edge[0], best_edge[1], **{self.weight_attr: self.max_weight})
        else:
            self.G[best_edge[0]][best_edge[1]][self.weight_attr] = self.max_weight
        
        return best_edge
