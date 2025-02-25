# Implement Friedkin-Johnsen Model, Altafini Model, Hegselmann-Krause Model, and Deffuant Model in this file
import networkx as nx
import numpy as np
from typing import Dict
import random


class OpinionDynamics:
    def __init__(self, G: nx.Graph, attri_name: str):
        self.G = G
        self.s = attri_name

    def friedkin_johnsen(self, opt=3, lb=0.5, ub: float = 0.9, max_iter: int = 20, reweight=[]) -> list:
        # Get initial opinions
        opinions = nx.get_node_attributes(self.G, self.s)
        vars = [np.var(list(opinions.values()))]
        initial_opinions = opinions.copy()
        max_deg = max([len(list(self.G.neighbors(node))) for node in self.G.nodes()])

        for _ in range(max_iter):
            old_opinions = opinions.copy()

            # Update each node's opinion
            for node in self.G.nodes():
                neighbors = list(self.G.neighbors(node))
                if opt == 1:
                # ==== Option 1: Use similarity as weights ====
                    similarities = 2 - abs(np.array([old_opinions[neigh] for neigh in neighbors if neigh != node]) -
                    old_opinions[node])
                    # weighted sum of neighbors' opinions
                    neighbor_influence = sum([similarity * old_opinions[neigh] for similarity, neigh in zip(similarities, neighbors)]) / sum(similarities)
                elif opt == 2:
                    # ==== Option 2: Use similarity * degree as weights ====
                    degrees = [len(list(self.G.neighbors(neigh))) for neigh in neighbors if neigh != node]
                    similarities = 2 - abs(np.array([old_opinions[neigh] for neigh in neighbors if neigh != node]) -
                    old_opinions[node])
                    for i in range(len(similarities)):
                        similarities[i] = similarities[i] * degrees[i]
                    if sum(similarities) == 0:
                        continue
                    neighbor_influence = (sum([sim * old_opinions[neigh] for sim, neigh in zip(similarities, neighbors)])
                                          / sum(similarities))
                elif opt == 3:
                    # ==== Option 3: PADS nodes have less weights ====
                    degrees = [len(list(self.G.neighbors(neigh))) for neigh in neighbors if neigh != node]
                    similarities = 2 - abs(np.array([old_opinions[neigh] for neigh in neighbors if neigh != node]) -
                                           old_opinions[node])
                    reweight_label = [neigh in reweight for neigh in neighbors if neigh != node]
                    for i in range(len(similarities)):
                        similarities[i] = similarities[i] * degrees[i]
                        if reweight_label[i]:
                            similarities[i] = similarities[i] * 0.3
                    if sum(similarities) == 0:
                        continue
                    neighbor_influence = (sum([sim * old_opinions[neigh] for sim, neigh in zip(similarities, neighbors)])
                                          / sum(similarities))


                # # Update opinion
                # stubbornness = lb + (1 - (len(neighbors) - 1) / max_deg) * (ub - lb)
                stubbornness = abs(old_opinions[node])
                opinions[node] = (1 - stubbornness) * initial_opinions[node] + stubbornness * neighbor_influence

            vars.append(np.var(list(opinions.values())))
        return vars

    def altafini(self, alpha: float = 0.1, max_iter: int = 100, tol: float = 1e-6) -> Dict:
        opinions = nx.get_node_attributes(self.G, self.s)

        # Get edge signs (assuming edges have 'sign' attribute: +1 or -1)
        signs = nx.get_edge_attributes(self.G, 'sign')

        for _ in range(max_iter):
            old_opinions = opinions.copy()

            for node in self.G.nodes():
                update = 0
                for neighbor in self.G.neighbors(node):
                    edge = (node, neighbor) if (node, neighbor) in signs else (neighbor, node)
                    sign = signs.get(edge, 1)  # Default to positive if no sign specified
                    update += sign * (opinions[neighbor] - opinions[node])

                opinions[node] += alpha * update

            # Check convergence
            if all(abs(opinions[node] - old_opinions[node]) < tol for node in self.G.nodes()):
                break

        return opinions

    def hegselmann_krause(self, confidence: float = 0.5, max_iter: int = 8, tol: float = 1e-6, reweight=[]) -> list:
        opinions = nx.get_node_attributes(self.G, self.s)
        vars = [np.var(list(opinions.values()))]

        for _ in range(max_iter):
            old_opinions = opinions.copy()

            for node in self.G.nodes():
                # Find neighbors within confidence bound
                confident_neighbors = [neigh for neigh in self.G.neighbors(node)
                    if abs(old_opinions[neigh] - old_opinions[node]) <= confidence and neigh not in reweight]

                if confident_neighbors:
                    # Include self in averaging
                    confident_neighbors.append(node)
                    # Update opinion as average of confident neighbors
                    opinions[node] = sum(old_opinions[neigh] for neigh in confident_neighbors) / len(confident_neighbors)

            vars.append(np.var(list(opinions.values())))
            # Check convergence
            if all(abs(opinions[node] - old_opinions[node]) < tol for node in self.G.nodes()):
                break

        return vars

    def deffuant(self, mu: float = 0.5, confidence: float = 0.2, max_iter: int = 100) -> Dict:
        opinions = nx.get_node_attributes(self.G, self.s)
        edges = list(self.G.edges())

        for _ in range(max_iter):
            # Randomly select an edge
            edge = edges[np.random.randint(len(edges))]
            i, j = edge

            # Check if opinions are within confidence bound
            if abs(opinions[i] - opinions[j]) <= confidence:
                # Update opinions
                old_i = opinions[i]
                old_j = opinions[j]
                opinions[i] += mu * (old_j - old_i)
                opinions[j] += mu * (old_i - old_j)

        return opinions


def opinion_dynamics_connections(d='Brexit', num_edges=2000, cascade_pos=3, cascade_neg=6, ax=None):
    print(f"===Dataset {d}===")
    # Read the original graph
    G = nx.read_gml(f'Output/{d}/graph.gml')

    # --- Simulation 1: Original Graph ---
    od1 = OpinionDynamics(G, 'polarity')
    vars1 = od1.friedkin_johnsen()

    # --- Simulation 2: Add edges within pads ---
    nodes_pads_pos = [node for node in G.nodes if G.nodes[node]['pads_cpp'] == 1]
    nodes_pads_neg = [node for node in G.nodes if G.nodes[node]['pads_cpp'] == -1]
    node_pairs_pads = [(pos, neg) for pos in nodes_pads_pos for neg in nodes_pads_neg]
    G2 = G.copy()
    for pos, neg in random.sample(node_pairs_pads, num_edges):
        G2.add_edge(pos, neg)
    vars2 = OpinionDynamics(G2, 'polarity').friedkin_johnsen()

    # --- Simulation 3: Add edges within cascade ---
    nodes_cascade_pos = [node for node in G.nodes if G.nodes[node]['cascade'] == cascade_pos]
    nodes_cascade_neg = [node for node in G.nodes if G.nodes[node]['cascade'] == cascade_neg]
    node_pairs_cascade = [(pos, neg) for pos in nodes_cascade_pos for neg in nodes_cascade_neg]
    G3 = G.copy()
    for pos, neg in random.sample(node_pairs_cascade, num_edges):
        G3.add_edge(pos, neg)
    vars3 = OpinionDynamics(G3, 'polarity').friedkin_johnsen()

    # --- Simulation 4: Add edges within cascade while not within pads ---
    cascade_minus_pads_pos = [node for node in G.nodes if G.nodes[node]['cascade'] == cascade_pos and G.nodes[node]['pads_cpp'] != 1]
    cascade_minus_pads_neg = [node for node in G.nodes if G.nodes[node]['cascade'] == cascade_neg and G.nodes[node]['pads_cpp'] != -1]
    node_pairs_cascade_minus_pads = [(pos, neg) for pos in cascade_minus_pads_pos for neg in cascade_minus_pads_neg]
    G4 = G.copy()
    for pos, neg in random.sample(node_pairs_cascade_minus_pads, num_edges):
        G4.add_edge(pos, neg)
    vars4 = OpinionDynamics(G4, 'polarity').friedkin_johnsen()

    # --- Simulation 5: Add edges within high-degree nodes with opposing polarity ---
    # select top 50 nodes with the largest degree and positive polarity
    nodes_pos = [node for node in G.nodes if G.nodes[node]['polarity'] > 0]
    nodes_neg = [node for node in G.nodes if G.nodes[node]['polarity'] < 0]
    nodes_pos.sort(key=lambda x: G.degree(x), reverse=True)
    nodes_neg.sort(key=lambda x: G.degree(x), reverse=True)
    node_pairs_high_degree = [(pos, neg) for pos in nodes_pos[:50] for neg in nodes_neg[:50]]
    G5 = G.copy()
    for pos, neg in random.sample(node_pairs_high_degree, num_edges):
        G5.add_edge(pos, neg)
    vars5 = OpinionDynamics(G5, 'polarity').friedkin_johnsen()


    # --- Plot the curves ---
    time_steps = range(len(vars1))

    ax.plot(time_steps, vars1, label='Original Graph')
    ax.plot(time_steps, vars2, label='PADS')
    ax.plot(time_steps, vars3, label='Cascade')
    ax.plot(time_steps, vars4, label='Cascade - PADS')
    ax.plot(time_steps, vars5, label='High Degree')

def opinion_dynamics_reweight(d='Brexit', ax=None):
    print(f"===Dataset {d}===")
    G = nx.read_gml(f'Output/{d}/graph.gml')
    od = OpinionDynamics(G, 'polarity')
    vars = od.friedkin_johnsen()
    time_steps = range(len(vars))
    ax.plot(time_steps, vars, marker='o', linewidth=1, label='No reweight', ms=3)

    border_nodes_pads = [node for node in G.nodes if G.nodes[node]['pads_cpp'] != 0 and any([G.nodes[neigh]['pads_cpp'] == 0 for neigh in G.neighbors(node)])]
    var = od.friedkin_johnsen(reweight=random.sample(list(G.nodes), len(border_nodes_pads)))
    ax.plot(time_steps, var, marker='s', linewidth=1, label='Random', ms=3)

    border_nodes_maxflow_u = [node for node in G.nodes if G.nodes[node]['maxflow_cpp_udsp'] != 0 and any([G.nodes[neigh]['maxflow_cpp_udsp'] == 0 for neigh in G.neighbors(node)])]
    var = od.friedkin_johnsen(reweight=border_nodes_maxflow_u)
    ax.plot(time_steps, var, marker='^', linewidth=1, label='MaxFlow-U border', ms=3)

    border_nodes_maxflow_w = [node for node in G.nodes if G.nodes[node]['maxflow_cpp_wdsp'] != 0 and any([G.nodes[neigh]['maxflow_cpp_wdsp'] == 0 for neigh in G.neighbors(node)])]
    var = od.friedkin_johnsen(reweight=border_nodes_maxflow_w)
    ax.plot(time_steps, var, marker='v', linewidth=1, label='MaxFlow-W border', ms=3)

    var = od.friedkin_johnsen(reweight=border_nodes_pads)
    ax.plot(time_steps, var, marker='D', linewidth=1, label='PADS border', ms=3)
