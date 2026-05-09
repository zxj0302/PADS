from .controversy_for_edges import get_tok_nodes, compute_controversy_score, get_candidate_edges, parallelization
import pickle
import networkx as nx
from scipy.sparse import diags
import math
import random

class MyGraphData:
    def __init__(self, G, uniform=True, partition=True):
        self.G = G.copy()
        if uniform:
            # assign uniform weights to edges
            for u, v in self.G.edges():
                self.G[u][v]['weight'] = 1.0
        else:
            raise NotImplementedError("Non-uniform weights not implemented")
        
        if partition:
            self.partition()
            self.make_id_color()

    def partition(self):
        # if most of node v's neighbors has polarity larger than 0, then v is red
        # if most of node v's neighbors has polarity smaller than 0, then v is blue
        # If the number of red and blue neighbors are equal, then v is red if its polarity is larger than 0
        # and blue if its polarity is smaller than 0

        self.red_nodes = []
        self.blue_nodes = []
        
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            red_count = sum(1 for n in neighbors if self.G.nodes[n]['polarity'] > 0)
            blue_count = sum(1 for n in neighbors if self.G.nodes[n]['polarity'] <= 0)
            
            if red_count > blue_count:
                self.red_nodes.append(node)
            elif blue_count > red_count:
                self.blue_nodes.append(node)
            else:
                if self.G.nodes[node]['polarity'] > 0:
                    self.red_nodes.append(node)
                else:
                    self.blue_nodes.append(node)

    def compute_perc(self, k, simple=True, perc_rb=0.2):
        if simple:
            return perc_rb, perc_rb
        r_num = len(self.red_nodes)
        b_num = len(self.blue_nodes)

        # First approach: try to satisfy all constraints
        # We want r_num * perc_r * b_num * perc_b between 4k and 8k
        # and perc_r / perc_b ≈ r_num / b_num
        
        # Target product in the middle of the range
        target_product = 6 * k
        
        # If we maintain the ratio perc_r / perc_b = r_num / b_num
        # Then we can derive x where perc_r = x / r_num and perc_b = x / b_num
        x = math.sqrt(target_product)
        
        # Calculate initial percentages
        perc_r = min(1.0, x / r_num)
        perc_b = min(1.0, x / b_num)
        
        # If either percentage is capped at 1.0, recalculate to meet target product
        if perc_r == 1.0:
            # Need to increase perc_b to compensate
            perc_b = min(1.0, target_product / (r_num * b_num))
        elif perc_b == 1.0:
            # Need to increase perc_r to compensate
            perc_r = min(1.0, target_product / (r_num * b_num))
        
        # Final check to ensure we're in the valid range
        final_product = r_num * perc_r * b_num * perc_b
        if final_product < 4 * k or final_product > 8 * k:
            print(f"Warning: Could not satisfy all constraints. Product: {final_product}, Target: between {2*k} and {5*k}")
        
        return perc_r, perc_b
    
    def make_id_color(self):
        # Create a dictionary to map node IDs to their colors
        id_color = {}
        for node in self.G.nodes():
            if node in self.red_nodes and node not in self.blue_nodes:
                id_color[node] = 'red'
            elif node in self.blue_nodes and node not in self.red_nodes:
                id_color[node] = 'blue'
            else:
                raise ValueError(f"Node {node} is not in either red or blue nodes list.")
        self.id_color = id_color


def compute_rwc(G, unweighted=True, k=1000, perc_rb=0.2, add_edges=[]):
    politics = MyGraphData(G=G, uniform=unweighted)
    G = politics.G
    if add_edges:
        G.add_edges_from(add_edges)
    node_id_matrix = {n:i for i,n in enumerate(list(G.nodes()))}
    A = nx.adjacency_matrix(G)
    d = diags(1/A.sum(axis=1).ravel())
    A = A.T.dot(d)
    perc_r, perc_b = politics.compute_perc(k, simple=True, perc_rb=perc_rb)
    red_topk = get_tok_nodes(G, node_id_matrix, politics.red_nodes, perc_k=perc_r)
    blue_topk = get_tok_nodes(G, node_id_matrix, politics.blue_nodes, perc_k=perc_b)
    score = compute_controversy_score(A, node_id_matrix, politics, red_topk, blue_topk, perc=10, alpha=0.95)
    return score


def RWC_reduction(G, unweighted=True, k=1000, perc_rb=0.2, ratio=10):
    politics = MyGraphData(G=G, uniform=unweighted)
    G = politics.G

    # Indeces for the adj matrix
    node_id_matrix = {n:i for i,n in enumerate(list(G.nodes()))}
    id_matrix_node = {j:i for i,j in node_id_matrix.items()}

    # Define adj matrix
    A = nx.adjacency_matrix(G)
    d = diags(1/A.sum(axis=1).ravel())
    # Column stochastic
    A = A.T.dot(d)#.T

    perc_r, perc_b = politics.compute_perc(k, perc_rb=perc_rb)
    red_topk = get_tok_nodes(G, node_id_matrix, politics.red_nodes, perc_k=perc_r)
    blue_topk = get_tok_nodes(G, node_id_matrix, politics.blue_nodes, perc_k=perc_b)
    print('RED TOPK: ', len(red_topk))
    print('BLUE TOPK: ', len(blue_topk))
    RWC = compute_controversy_score(A, node_id_matrix, politics, red_topk, blue_topk, perc=10, alpha=0.95)
    print('RWC computed: ', RWC)

    candidate_edges = get_candidate_edges(G, red_topk, blue_topk)
    candidate_edges = [(i,j) for i,j in candidate_edges if (id_matrix_node[i], id_matrix_node[j]) not in G.edges()]

    # Changed by Xiangju
    candidate_edges = random.sample(candidate_edges, min(ratio*k, len(candidate_edges)))
    print('LEN CANDIDATES: ', len(candidate_edges))

    result_chunks = parallelization(RWC, A, node_id_matrix, politics, red_topk, blue_topk, candidate_edges, perc=10, alpha=0.95, parallel=False)
    # print(result_chunks)
    tops = []
    for i in range(len(result_chunks)):
        tops += sorted(result_chunks[i].items(), key=lambda x: x[1], reverse=True)
    candidate_edges_ = sorted(tops, key=lambda x: x[1], reverse=True)[:min(len(tops), k)]
    print('FINAL SELECTED LEN CANDIDATES: ', len(candidate_edges_))

    # convert the candidate edges to the original node labels
    candidate_edges_ = [(id_matrix_node[i[0][0]], id_matrix_node[i[0][1]]) for i in candidate_edges_]
    # G.add_edges_from(candidate_edges_)
    new_rwc_score = compute_rwc(G, perc_rb=perc_rb, add_edges=candidate_edges_)
    print('NEW RWC SCORE: ', new_rwc_score)
    # with open('rov_candidate_edges.pickle', 'wb') as f:
    #     pickle.dump(candidate_edges_, f)
    
    return candidate_edges_


if __name__ == "__main__":
    G = nx.read_gml('../../../output/results-theta=0.5/Gun/graph.gml')
    RWC_reduction(G, unweighted=True)