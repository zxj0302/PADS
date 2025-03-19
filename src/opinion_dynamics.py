# Implement Friedkin-Johnsen Model, Altafini Model, Hegselmann-Krause Model, and Deffuant Model in this file
import networkx as nx
import numpy as np
from typing import Dict
import random


class OpinionDynamics:
    def __init__(self, G: nx.Graph, attri_name: str, ratio: float):
        self.G = G
        self.s = attri_name
        self.ratio = ratio
        self.non_core = [node for node in self.G.nodes if self.G.nodes[node]['pads_cpp'] == 0]

    def friedkin_johnsen(self, opt=3, lb=0.5, ub: float = 0.9, max_iter: int = 20, reweight=[]):
        # Get initial opinions
        opinions = nx.get_node_attributes(self.G, self.s)
        # get the average positive opinions and average negative opinions in each iteration and return the list
        avg_pos = []
        avg_neg = []
        vars = [np.var(list(opinions.values()))]
        # Calculate initial average positive and negative opinions
        pos_opinions = [op for op in opinions.values() if op > 0]
        neg_opinions = [op for op in opinions.values() if op < 0]
        avg_pos.append(np.mean(pos_opinions) if pos_opinions else 0)
        avg_neg.append(np.mean(neg_opinions) if neg_opinions else 0)
        initial_opinions = opinions.copy()
        max_deg = max([len(list(self.G.neighbors(node))) for node in self.G.nodes()])

        for _ in range(max_iter):
            old_opinions = opinions.copy()
            # Update each node's opinion
            for node in self.G.nodes():
                neighbors = list(self.G.neighbors(node))
                if opt == 1:
                # ==== Option 1: Use similarity as weights ====
                    similarities = 2 - abs(np.array([old_opinions[neigh] for neigh in neighbors if neigh != node]) - old_opinions[node])
                    # weighted sum of neighbors' opinions
                    neighbor_influence = sum([similarity * old_opinions[neigh] for similarity, neigh in zip(similarities, neighbors)]) / sum(similarities)
                elif opt == 2:
                    # ==== Option 2: Use similarity * degree as weights ====
                    degrees = [len(list(self.G.neighbors(neigh))) for neigh in neighbors if neigh != node]
                    similarities = 2 - abs(np.array([old_opinions[neigh] for neigh in neighbors if neigh != node]) - old_opinions[node])
                    for i in range(len(similarities)):
                        similarities[i] = similarities[i] * degrees[i]
                    if sum(similarities) == 0:
                        continue
                    neighbor_influence = (sum([sim * old_opinions[neigh] for sim, neigh in zip(similarities, neighbors)]) / sum(similarities))
                elif opt == 3:
                    # ==== Option 3: PADS nodes have less weights ====
                    degrees = [len(list(self.G.neighbors(neigh))) for neigh in neighbors if neigh != node]
                    similarities = 2 - abs(np.array([old_opinions[neigh] for neigh in neighbors if neigh != node]) - old_opinions[node])
                    reweight_label = [neigh in reweight for neigh in neighbors if neigh != node]
                    for i in range(len(similarities)):
                        similarities[i] = similarities[i] * degrees[i]
                        if reweight_label[i]:
                            similarities[i] = similarities[i] * self.ratio
                    if sum(similarities) == 0:
                        continue
                    neighbor_influence = (sum([sim * old_opinions[neigh] for sim, neigh in zip(similarities, neighbors)]) / sum(similarities))
                    
                # # Update opinion
                # stubbornness = lb + (1 - (len(neighbors) - 1) / max_deg) * (ub - lb)
                stubbornness = abs(old_opinions[node])
                opinions[node] = stubbornness * initial_opinions[node] + (1 - stubbornness) * neighbor_influence
            # Calculate averages after all nodes are updated in this iteration
            pos_opinions = [op for op in opinions.values() if op > 0]
            neg_opinions = [op for op in opinions.values() if op < 0]
            avg_pos.append(np.mean(pos_opinions) if pos_opinions else 0)
            avg_neg.append(np.mean(neg_opinions) if neg_opinions else 0)

            # caluculate variance for nodes that are non-core
            # vars.append(np.var(list([opinions[node] for node in self.non_core])))
            vars.append(np.var(list(opinions.values())))
        return vars, avg_pos, avg_neg
        
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


def opinion_dynamics_connections(d='Brexit', num_edges=2000, ax_var=None, ax_diff=None, ratio=0.3, it=20):
    print(f"===Dataset {d}===")
    # Read the original graph
    G = nx.read_gml(f'Output/{d}/graph.gml')

    # --- Simulation 1: Original Graph ---
    od1 = OpinionDynamics(G, 'polarity', ratio)
    vars1, avg_pos1, avg_neg1 = od1.friedkin_johnsen(max_iter=it)

    # --- Simulation 2: Add random edges between positive and negative nodes ---
    pos_nodes = [node for node in G.nodes if G.nodes[node]['polarity'] > 0]
    neg_nodes = [node for node in G.nodes if G.nodes[node]['polarity'] < 0]
    
    G2 = G.copy()
    # Prevent adding repeated edges
    added_edges = 0
    attempts = 0
    max_attempts = num_edges * 10  # Limit attempts to avoid infinite loop
    
    while added_edges < num_edges and attempts < max_attempts:
        pos_node = random.choice(pos_nodes)
        neg_node = random.choice(neg_nodes)
        
        # Check if edge already exists
        if not G2.has_edge(pos_node, neg_node):
            G2.add_edge(pos_node, neg_node)
            added_edges += 1
        
        attempts += 1
    
    od2 = OpinionDynamics(G2, 'polarity', ratio)
    vars2, avg_pos2, avg_neg2 = od2.friedkin_johnsen(max_iter=it)

    # --- Simulation 3: Add edges between high degree positive and negative nodes ---
    # Get the counts of positive and negative nodes in PADS
    num_pads_pos = len([node for node in G.nodes if G.nodes[node]['pads_cpp'] == 1])
    num_pads_neg = len([node for node in G.nodes if G.nodes[node]['pads_cpp'] == -1])
    
    # Sort nodes by degree
    pos_nodes_sorted = sorted(pos_nodes, key=lambda x: G.degree(x), reverse=True)
    neg_nodes_sorted = sorted(neg_nodes, key=lambda x: G.degree(x), reverse=True)
    
    # Truncate to match PADS counts
    high_degree_pos = pos_nodes_sorted[:num_pads_pos]
    high_degree_neg = neg_nodes_sorted[:num_pads_neg]
    
    G3 = G.copy()
    # Prevent adding repeated edges
    added_edges = 0
    attempts = 0
    
    while added_edges < num_edges and attempts < max_attempts:
        pos_node = random.choice(high_degree_pos)
        neg_node = random.choice(high_degree_neg)
        
        # Check if edge already exists
        if not G3.has_edge(pos_node, neg_node):
            G3.add_edge(pos_node, neg_node)
            added_edges += 1
        
        attempts += 1
    
    od3 = OpinionDynamics(G3, 'polarity', ratio)
    vars3, avg_pos3, avg_neg3 = od3.friedkin_johnsen(max_iter=it)

    # --- Simulation 4: Add edges within PADS communities ---
    nodes_pads_pos = [node for node in G.nodes if G.nodes[node]['pads_cpp'] == 1]
    nodes_pads_neg = [node for node in G.nodes if G.nodes[node]['pads_cpp'] == -1]
    
    G4 = G.copy()
    # Prevent adding repeated edges
    added_edges = 0
    attempts = 0
    
    while added_edges < num_edges and attempts < max_attempts:
        pos_node = random.choice(nodes_pads_pos)
        neg_node = random.choice(nodes_pads_neg)
        
        # Check if edge already exists
        if not G4.has_edge(pos_node, neg_node):
            G4.add_edge(pos_node, neg_node)
            added_edges += 1
        
        attempts += 1
    
    od4 = OpinionDynamics(G4, 'polarity', ratio)
    vars4, avg_pos4, avg_neg4 = od4.friedkin_johnsen(max_iter=it)

    # --- Simulation 5: Add edges between (positive but not in PADS) and (negative but not in PADS) nodes ---
    non_pads_pos = [node for node in pos_nodes if G.nodes[node]['pads_cpp'] != 1]
    non_pads_neg = [node for node in neg_nodes if G.nodes[node]['pads_cpp'] != -1]
    
    G5 = G.copy()
    # Prevent adding repeated edges
    added_edges = 0
    attempts = 0
    
    while added_edges < num_edges and attempts < max_attempts:
        pos_node = random.choice(non_pads_pos)
        neg_node = random.choice(non_pads_neg)
        
        # Check if edge already exists
        if not G5.has_edge(pos_node, neg_node):
            G5.add_edge(pos_node, neg_node)
            added_edges += 1
        
        attempts += 1
    
    od5 = OpinionDynamics(G5, 'polarity', ratio)
    vars5, avg_pos5, avg_neg5 = od5.friedkin_johnsen(max_iter=it)

    # --- Simulation 6: Add edges between PADS nodes with higher degree and lower polarity ---
    # Calculate weights for each node based on degree and polarity
    pos_weights = {}
    for node in nodes_pads_pos:
        # Higher degree and lower polarity gets higher weight
        pos_weights[node] = G.degree(node) * (1 / (G.nodes[node]['polarity'] + 0.1))  # Add 0.1 to avoid division by zero
    
    neg_weights = {}
    for node in nodes_pads_neg:
        # Higher degree and lower absolute polarity gets higher weight
        neg_weights[node] = G.degree(node) * (1 / (abs(G.nodes[node]['polarity']) + 0.1))
    
    # Normalize weights
    pos_nodes_list = list(pos_weights.keys())
    pos_weights_list = [pos_weights[node] for node in pos_nodes_list]
    pos_weights_sum = sum(pos_weights_list)
    pos_weights_normalized = [w/pos_weights_sum for w in pos_weights_list]
    
    neg_nodes_list = list(neg_weights.keys())
    neg_weights_list = [neg_weights[node] for node in neg_nodes_list]
    neg_weights_sum = sum(neg_weights_list)
    neg_weights_normalized = [w/neg_weights_sum for w in neg_weights_list]
    
    # Create a copy of the graph
    G6 = G.copy()
    
    # Add edges by sampling pairs based on weights
    added_edges = 0
    attempts = 0
    max_attempts = num_edges * 10  # Limit attempts to avoid infinite loop
    
    while added_edges < num_edges and attempts < max_attempts:
        # Sample one node from each community based on weights
        pos_node = np.random.choice(pos_nodes_list, p=pos_weights_normalized)
        neg_node = np.random.choice(neg_nodes_list, p=neg_weights_normalized)
        
        # Check if edge already exists
        if not G6.has_edge(pos_node, neg_node):
            G6.add_edge(pos_node, neg_node)
            added_edges += 1
        
        attempts += 1
    
    od6 = OpinionDynamics(G6, 'polarity', ratio)
    vars6, avg_pos6, avg_neg6 = od6.friedkin_johnsen(max_iter=it)

    # --- Simulation 7: Add edges between all positive and negative nodes with higher degree and lower polarity ---
    # Calculate weights for all positive and negative nodes
    all_pos_weights = {}
    for node in pos_nodes:  # Using all positive nodes, not just PADS
        # Higher degree and lower polarity gets higher weight
        all_pos_weights[node] = G.degree(node) * (1 / (G.nodes[node]['polarity'] + 0.1))
    
    all_neg_weights = {}
    for node in neg_nodes:  # Using all negative nodes, not just PADS
        # Higher degree and lower absolute polarity gets higher weight
        all_neg_weights[node] = G.degree(node) * (1 / (abs(G.nodes[node]['polarity']) + 0.1))
    
    # Normalize weights
    all_pos_nodes_list = list(all_pos_weights.keys())
    all_pos_weights_list = [all_pos_weights[node] for node in all_pos_nodes_list]
    all_pos_weights_sum = sum(all_pos_weights_list)
    all_pos_weights_normalized = [w/all_pos_weights_sum for w in all_pos_weights_list]
    
    all_neg_nodes_list = list(all_neg_weights.keys())
    all_neg_weights_list = [all_neg_weights[node] for node in all_neg_nodes_list]
    all_neg_weights_sum = sum(all_neg_weights_list)
    all_neg_weights_normalized = [w/all_neg_weights_sum for w in all_neg_weights_list]
    
    # Create a copy of the graph
    G7 = G.copy()
    
    # Add edges by sampling pairs based on weights
    added_edges = 0
    attempts = 0
    max_attempts = num_edges * 10  # Limit attempts to avoid infinite loop
    
    while added_edges < num_edges and attempts < max_attempts:
        # Sample one node from each community based on weights
        pos_node = np.random.choice(all_pos_nodes_list, p=all_pos_weights_normalized)
        neg_node = np.random.choice(all_neg_nodes_list, p=all_neg_weights_normalized)
        
        # Check if edge already exists
        if not G7.has_edge(pos_node, neg_node):
            G7.add_edge(pos_node, neg_node)
            added_edges += 1
        
        attempts += 1
    
    od7 = OpinionDynamics(G7, 'polarity', ratio)
    vars7, avg_pos7, avg_neg7 = od7.friedkin_johnsen(max_iter=it)

    # --- Plot the curves ---
    time_steps = range(len(vars1))
    
    # Define colors for better visualization - using muted, low contrast palette like in reweight
    colors = {
        'Original': '#8da0cb',     # Muted blue
        'Random': '#66c2a5',       # Soft teal
        'High Degree': '#e5c494',  # Soft yellow/gold
        'PADS': '#a6d854',         # Soft green
        'Non-PADS': '#fc8d62',     # Muted orange
        'Weighted PADS': '#e78ac3', # Soft purple/pink
        'Weighted All': '#b3b3b3'   # Soft gray
    }
    
    # Define markers for each simulation
    markers = {
        'Original': 'o',
        'Random': 's',
        'High Degree': '*',
        'PADS': 'D',
        'Non-PADS': '^',
        'Weighted PADS': 'v',
        'Weighted All': 'p'
    }

    # Plot variance with consistent style
    if ax_var is not None:
        ax_var.plot(time_steps, vars1, marker=markers['Original'], linewidth=1, 
                   label='Original Graph', ms=3, color=colors['Original'])
        ax_var.plot(time_steps, vars2, marker=markers['Random'], linewidth=1, 
                   label='Random Edges', ms=3, color=colors['Random'])
        ax_var.plot(time_steps, vars3, marker=markers['High Degree'], linewidth=1, 
                   label='High Degree', ms=3, color=colors['High Degree'])
        ax_var.plot(time_steps, vars4, marker=markers['PADS'], linewidth=1, 
                   label='PADS', ms=3, color=colors['PADS'])
        ax_var.plot(time_steps, vars5, marker=markers['Non-PADS'], linewidth=1, 
                   label='Non-PADS', ms=3, color=colors['Non-PADS'])
        ax_var.plot(time_steps, vars6, marker=markers['Weighted PADS'], linewidth=1, 
                   label='Weighted PADS', ms=3, color=colors['Weighted PADS'])
        ax_var.plot(time_steps, vars7, marker=markers['Weighted All'], linewidth=1, 
                   label='Weighted All', ms=3, color=colors['Weighted All'])
        
        ax_var.set_xlabel('Time Steps')
        ax_var.set_ylabel('Opinion Variance')
        ax_var.grid(True, linestyle='--', alpha=0.3)
        ax_var.legend(loc='best', fontsize=9)
    
    # Plot opinion gap with consistent style
    if ax_diff is not None:
        # Calculate opinion gaps
        diff1 = [p - n for p, n in zip(avg_pos1, avg_neg1)]
        diff2 = [p - n for p, n in zip(avg_pos2, avg_neg2)]
        diff3 = [p - n for p, n in zip(avg_pos3, avg_neg3)]
        diff4 = [p - n for p, n in zip(avg_pos4, avg_neg4)]
        diff5 = [p - n for p, n in zip(avg_pos5, avg_neg5)]
        diff6 = [p - n for p, n in zip(avg_pos6, avg_neg6)]
        diff7 = [p - n for p, n in zip(avg_pos7, avg_neg7)]
        
        ax_diff.plot(time_steps, diff1, marker=markers['Original'], linewidth=1, 
                    label='Original Graph', ms=3, color=colors['Original'])
        ax_diff.plot(time_steps, diff2, marker=markers['Random'], linewidth=1, 
                    label='Random Edges', ms=3, color=colors['Random'])
        ax_diff.plot(time_steps, diff3, marker=markers['High Degree'], linewidth=1, 
                    label='High Degree', ms=3, color=colors['High Degree'])
        ax_diff.plot(time_steps, diff4, marker=markers['PADS'], linewidth=1, 
                    label='PADS', ms=3, color=colors['PADS'])
        ax_diff.plot(time_steps, diff5, marker=markers['Non-PADS'], linewidth=1, 
                    label='Non-PADS', ms=3, color=colors['Non-PADS'])
        ax_diff.plot(time_steps, diff6, marker=markers['Weighted PADS'], linewidth=1, 
                    label='Weighted PADS', ms=3, color=colors['Weighted PADS'])
        ax_diff.plot(time_steps, diff7, marker=markers['Weighted All'], linewidth=1, 
                    label='Weighted All', ms=3, color=colors['Weighted All'])
        
        ax_diff.set_xlabel('Time Steps')
        ax_diff.set_ylabel('Opinion Gap (Avg Pos - Avg Neg)')
        ax_diff.grid(True, linestyle='--', alpha=0.3)
        ax_diff.legend(loc='best', fontsize=9)
    
    # Return the data for further analysis if needed
    # return {
    #     'Variance': {
    #         'Original': vars1,
    #         'Random': vars2,
    #         'High Degree': vars3,
    #         'PADS': vars4,
    #         'Non-PADS': vars5,
    #         'Weighted PADS': vars6,
    #         'Weighted All': vars7
    #     },
    #     'Opinion Gap': {
    #         'Original': [p - n for p, n in zip(avg_pos1, avg_neg1)],
    #         'Random': [p - n for p, n in zip(avg_pos2, avg_neg2)],
    #         'High Degree': [p - n for p, n in zip(avg_pos3, avg_neg3)],
    #         'PADS': [p - n for p, n in zip(avg_pos4, avg_neg4)],
    #         'Non-PADS': [p - n for p, n in zip(avg_pos5, avg_neg5)],
    #         'Weighted PADS': [p - n for p, n in zip(avg_pos6, avg_neg6)],
    #         'Weighted All': [p - n for p, n in zip(avg_pos7, avg_neg7)]
    #     }
    # }

def opinion_dynamics_reweight(d='Brexit', ax_var=None, ax_diff=None, show_legend=False, ratio=0.3):
    print(f"===Dataset {d}===")
    G = nx.read_gml(f'Output/{d}/graph.gml')
    od = OpinionDynamics(G, 'polarity', ratio)
    
    # Define colors for each algorithm - muted, low contrast palette
    colors = {
        'No reweight': '#8da0cb',  # Muted blue
        'Random': '#66c2a5',       # Soft teal
        'MaxFlow-U': '#fc8d62',    # Muted orange
        'MaxFlow-W': '#e78ac3',    # Soft purple/pink
        'PADS': '#a6d854'          # Soft green
    }
    
    # No reweight
    vars, avg_pos, avg_neg = od.friedkin_johnsen()
    time_steps = range(len(vars))
    
    # Plot variance
    if ax_var is not None:
        ax_var.plot(time_steps, vars, marker='o', linewidth=1, label='No reweight', 
                ms=3, color=colors['No reweight'])
    
    # Plot difference
    if ax_diff is not None:
        diff = [p - n for p, n in zip(avg_pos, avg_neg)]
        ax_diff.plot(time_steps, diff, marker='o', linewidth=1, label='No reweight', 
                    ms=3, color=colors['No reweight'])

    # Random reweight
    border_nodes_pads = [node for node in G.nodes if G.nodes[node]['pads_cpp'] != 0 and any([G.nodes[neigh]['pads_cpp'] == 0 for neigh in G.neighbors(node)])]
    vars, avg_pos, avg_neg = od.friedkin_johnsen(reweight=random.sample(list(G.nodes), len(border_nodes_pads)))
    
    if ax_var is not None:
        ax_var.plot(time_steps, vars, marker='s', linewidth=1, label='Random', 
                ms=3, color=colors['Random'])
    
    if ax_diff is not None:
        diff = [p - n for p, n in zip(avg_pos, avg_neg)]
        ax_diff.plot(time_steps, diff, marker='s', linewidth=1, label='Random', 
                    ms=3, color=colors['Random'])

    # Highest degree reweight
    # Count positive and negative border nodes
    pos_border_count = len([node for node in border_nodes_pads if G.nodes[node]['pads_cpp'] > 0])
    neg_border_count = len([node for node in border_nodes_pads if G.nodes[node]['pads_cpp'] < 0])
    
    # Select high degree nodes with matching polarity counts
    pos_nodes = [node for node in G.nodes if G.nodes[node]['polarity'] > 0]
    neg_nodes = [node for node in G.nodes if G.nodes[node]['polarity'] < 0]
    pos_nodes_sorted = sorted(pos_nodes, key=lambda x: G.degree(x), reverse=True)[:pos_border_count]
    neg_nodes_sorted = sorted(neg_nodes, key=lambda x: G.degree(x), reverse=True)[:neg_border_count]
    high_degree_nodes = pos_nodes_sorted + neg_nodes_sorted
    
    vars, avg_pos, avg_neg = od.friedkin_johnsen(reweight=high_degree_nodes)
    
    if ax_var is not None:
        ax_var.plot(time_steps, vars, marker='*', linewidth=1, label='High Degree', 
                ms=3, color='#e5c494')  # Soft yellow/gold
    
    if ax_diff is not None:
        diff = [p - n for p, n in zip(avg_pos, avg_neg)]
        ax_diff.plot(time_steps, diff, marker='*', linewidth=1, label='High Degree', 
                    ms=3, color='#e5c494')  # Soft yellow/gold

    # MaxFlow-U border
    border_nodes_maxflow_u = [node for node in G.nodes if G.nodes[node]['maxflow_cpp_udsp'] != 0 and any([G.nodes[neigh]['maxflow_cpp_udsp'] == 0 for neigh in G.neighbors(node)])]
    vars, avg_pos, avg_neg = od.friedkin_johnsen(reweight=border_nodes_maxflow_u)
    
    if ax_var is not None:
        ax_var.plot(time_steps, vars, marker='^', linewidth=1, label='MaxFlow-U', 
                ms=3, color=colors['MaxFlow-U'])
    
    if ax_diff is not None:
        diff = [p - n for p, n in zip(avg_pos, avg_neg)]
        ax_diff.plot(time_steps, diff, marker='^', linewidth=1, label='MaxFlow-U', 
                    ms=3, color=colors['MaxFlow-U'])

    # MaxFlow-W border
    border_nodes_maxflow_w = [node for node in G.nodes if G.nodes[node]['maxflow_cpp_wdsp'] != 0 and any([G.nodes[neigh]['maxflow_cpp_wdsp'] == 0 for neigh in G.neighbors(node)])]
    vars, avg_pos, avg_neg = od.friedkin_johnsen(reweight=border_nodes_maxflow_w)
    
    if ax_var is not None:
        ax_var.plot(time_steps, vars, marker='v', linewidth=1, label='MaxFlow-W', 
                ms=3, color=colors['MaxFlow-W'])
    
    if ax_diff is not None:
        diff = [p - n for p, n in zip(avg_pos, avg_neg)]
        ax_diff.plot(time_steps, diff, marker='v', linewidth=1, label='MaxFlow-W', 
                    ms=3, color=colors['MaxFlow-W'])
    
    # PADS border
    vars, avg_pos, avg_neg = od.friedkin_johnsen(reweight=border_nodes_pads)
    
    if ax_var is not None:
        ax_var.plot(time_steps, vars, marker='D', linewidth=1, label='PADS', 
                ms=3, color=colors['PADS'])
        ax_var.set_ylabel('Variance')
    
    if ax_diff is not None:
        diff = [p - n for p, n in zip(avg_pos, avg_neg)]
        ax_diff.plot(time_steps, diff, marker='D', linewidth=1, label='PADS', 
                    ms=3, color=colors['PADS'])
        ax_diff.set_ylabel('Opinion Gap (Avg Pos - Avg Neg)')
    
    # Only show legends in the last subfigure
    if show_legend:
        if ax_var is not None:
            ax_var.legend(loc='upper right', fontsize=9)
        if ax_diff is not None:
            ax_diff.legend(loc='upper right', fontsize=9)
