# Implement Friedkin-Johnsen Model, Altafini Model, Hegselmann-Krause Model, and Deffuant Model in this file
import networkx as nx
import numpy as np
from typing import Dict
import random
from matplotlib.lines import Line2D


class OpinionDynamics:
    def __init__(self, G: nx.Graph, attri_name: str = 'polarity', ratio: float = 0.05):
        self.G = G
        self.s = attri_name
        self.ratio = ratio
        # self.non_core = [node for node in self.G.nodes if self.G.nodes[node]['pads_cpp'] == 0]
    
    def run(self, model: str='friedkin_johnsen_cb', **kwargs):
        if model == 'friedkin_johnsen_simplified_matrix':
            return self.friedkin_johnsen_simplified_matrix(**kwargs)
        elif model == 'friedkin_johnsen_simplified':
            return self.friedkin_johnsen_simplified(**kwargs)
        elif model == 'friedkin_johnsen':
            return self.friedkin_johnsen(**kwargs)
        elif model == 'friedkin_johnsen_cb':
            return self.friedkin_johnsen_cb(**kwargs)
        elif model == 'altafini':
            return self.altafini(**kwargs)
        elif model == 'hegselmann_krause':
            return self.hegselmann_krause(**kwargs)
        elif model == 'deffuant':
            return self.deffuant(**kwargs)
        else:
            raise ValueError(f"Unknown model: {model}")
        
    def friedkin_johnsen_simplified_matrix(self, reweight={}, add_connections=[]):
        # Work on a copy if add_connections is provided
        if add_connections:
            G = self.G.copy()
            G.add_edges_from(add_connections)
        else:
            G = self.G
        nodes = list(G.nodes())
        n = len(nodes)
        # Internal opinions vector (s)
        s_vec = np.array([G.nodes[node][self.s] for node in nodes])
        # Build weighted adjacency matrix W
        W = np.zeros((n, n))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if G.has_edge(u, v) or G.has_edge(v, u):
                    edge = (u, v)
                    edge_rev = (v, u)
                    if reweight and edge in reweight:
                        weight = reweight[edge]
                    elif reweight and edge_rev in reweight:
                        weight = reweight[edge_rev]
                    else:
                        weight = (2 - abs(G.nodes[u][self.s] - G.nodes[v][self.s])) / 2
                    W[i, j] = weight
        # Degree matrix D
        D = np.diag(W.sum(axis=1))
        # Laplacian L = D - W
        L = D - W
        # Identity matrix
        I = np.eye(n)
        # Solve for z
        L_plus_I = L + I
        z_vec = np.linalg.solve(L_plus_I, s_vec)
        # Prepare output: initial and final
        initial_dict = {node: s for node, s in zip(nodes, s_vec)}
        final_dict = {node: z for node, z in zip(nodes, z_vec)}
        return [initial_dict, final_dict]
        
    def friedkin_johnsen_simplified(self, max_iter: int = 100, reweight=[], add_connections=[]):
        # Work on a copy of the graph if add_connections is provided
        if add_connections:
            G = self.G.copy()
            G.add_edges_from(add_connections)
        else:
            G = self.G

        opinions = nx.get_node_attributes(G, self.s)
        initial_opinions = opinions.copy()
        opinions_over_time = [opinions.copy()]

        for _ in range(max_iter):
            old_opinions = opinions.copy()
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                weights = []
                for neigh in neighbors:
                    # Use reweight if provided, otherwise use |s_i - s_j|
                    edge = (node, neigh)
                    edge_rev = (neigh, node)
                    if reweight and (edge in reweight):
                        w = reweight[edge]
                    elif reweight and (edge_rev in reweight):
                        w = reweight[edge_rev]
                    else:
                        w = (2 - abs(initial_opinions[node] - initial_opinions[neigh])) / 2
                    weights.append(w)
                sum_weighted_neighbor_opinions = sum(w * old_opinions[neigh] for w, neigh in zip(weights, neighbors))
                sum_weights = sum(weights)
                numerator = initial_opinions[node] * 1 + sum_weighted_neighbor_opinions
                denominator = 1 + sum_weights
                opinions[node] = numerator / denominator if denominator != 0 else initial_opinions[node]
            opinions_over_time.append(opinions.copy())
        return opinions_over_time

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
    
    def friedkin_johnsen_cb(self, eta=100, max_iter: int = 30, reweight=[]):
        # Thsi is the FJ model with confirmation bias
        # Get initial opinions
        opinions = nx.get_node_attributes(self.G, self.s)
        initial_opinions = opinions.copy()
        
        # Initialize tracking metrics
        vars = [np.var(list(opinions.values()))]
        pos_opinions = [op for op in opinions.values() if op > 0]
        neg_opinions = [op for op in opinions.values() if op < 0]
        avg_pos = [np.mean(pos_opinions) if pos_opinions else 0]
        avg_neg = [np.mean(neg_opinions) if neg_opinions else 0]
        max_deg = max([len(list(self.G.neighbors(node))) for node in self.G.nodes()])
        
        # Initialize weights
        weights = {}
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            if not neighbors:
                continue
                
            # Calculate weights based on opinion multiplication
            incoming_weights = {}
            for neighbor in neighbors:
                # weight = max(0, initial_opinions[node] * initial_opinions[neighbor])
                weight = (2 - abs(initial_opinions[node] - initial_opinions[neighbor])) * len(list(self.G.neighbors(neighbor)))
                if neighbor in reweight:
                    weight *= self.ratio
                incoming_weights[neighbor] = weight
            
            # Normalize weights
            total_weight = sum(incoming_weights.values())
            if total_weight != 0:
                for neighbor, weight in incoming_weights.items():
                    weights[(neighbor, node)] = weight / total_weight
            else:
                equal_weight = 1.0 / len(neighbors)
                for neighbor in neighbors:
                    weights[(neighbor, node)] = equal_weight

        # Simulation loop
        for _ in range(max_iter):
            old_opinions = opinions.copy()
            
            # Update opinions
            for node in self.G.nodes():
                neighbors = list(self.G.neighbors(node))
                if not neighbors:
                    continue
                    
                neighbor_influence = sum(weights[(neighbor, node)] * old_opinions[neighbor] for neighbor in neighbors)
                # stubbornness = abs(old_opinions[node])
                stubbornness = (len(neighbors) / max_deg) ** 2
                opinions[node] = stubbornness * initial_opinions[node] + (1 - stubbornness) * neighbor_influence
            
            # Update weights based on new opinions
            for node in self.G.nodes():
                neighbors = list(self.G.neighbors(node))
                if not neighbors:
                    continue
                
                # Apply weight update rule and gather for normalization
                incoming_weights = {neighbor: max(0, weights.get((neighbor, node), 0) + 
                                               eta * opinions[neighbor] * opinions[node])
                                  for neighbor in neighbors}
                
                # Normalize updated weights
                total_weight = sum(incoming_weights.values())
                if total_weight > 0:
                    for neighbor, weight in incoming_weights.items():
                        weights[(neighbor, node)] = weight / total_weight
            
            # Track metrics for this iteration
            pos_opinions = [op for op in opinions.values() if op > 0]
            neg_opinions = [op for op in opinions.values() if op < 0]
            avg_pos.append(np.mean(pos_opinions) if pos_opinions else 0)
            avg_neg.append(np.mean(neg_opinions) if neg_opinions else 0)
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


def opinion_dynamics_connections(G, num_edges=2000, ax_var=None, ax_diff=None, it=20, show_legend=False, show_ylabel=False):
    # Extract node groups
    pos_nodes = [n for n in G.nodes if G.nodes[n]['polarity'] > 0]
    neg_nodes = [n for n in G.nodes if G.nodes[n]['polarity'] < 0]
    nodes_pads_pos = [n for n in G.nodes if G.nodes[n]['pads_cpp'] == 1]
    nodes_pads_neg = [n for n in G.nodes if G.nodes[n]['pads_cpp'] == -1]
    non_pads_pos = [n for n in pos_nodes if G.nodes[n]['pads_cpp'] != 1]
    non_pads_neg = [n for n in neg_nodes if G.nodes[n]['pads_cpp'] != -1]
    nodes_wdsp_pos = [n for n in G.nodes if G.nodes[n]['maxflow_cpp_wdsp'] == 1]
    nodes_wdsp_neg = [n for n in G.nodes if G.nodes[n]['maxflow_cpp_wdsp'] == -1]
    nodes_udsp_pos = [n for n in G.nodes if G.nodes[n]['maxflow_cpp_udsp'] == 1]
    nodes_udsp_neg = [n for n in G.nodes if G.nodes[n]['maxflow_cpp_udsp'] == -1]
    nodes_gin_pos = [n for n in G.nodes if G.nodes[n]['node2vec_gin'] == 1]
    nodes_gin_neg = [n for n in G.nodes if G.nodes[n]['node2vec_gin'] == -1]
    
    # Get high degree nodes matching PADS counts
    high_degree_pos = sorted(pos_nodes, key=lambda x: G.degree(x), reverse=True)[:len(nodes_pads_pos)]
    high_degree_neg = sorted(neg_nodes, key=lambda x: G.degree(x), reverse=True)[:len(nodes_pads_neg)]
    
    # Helper function to add edges between node groups
    def add_edges_between(source_nodes, target_nodes, weighted=False):
        G_new = G.copy()
        added, attempts = 0, 0
        max_attempts = num_edges * 10
        
        # Setup for weighted selection if needed
        if weighted:
            src_weights, tgt_weights = {}, {}
            for n in source_nodes:
                src_weights[n] = G.degree(n) * (1 / (abs(G.nodes[n]['polarity']) + 0.01))
            for n in target_nodes:
                tgt_weights[n] = G.degree(n) * (1 / (abs(G.nodes[n]['polarity']) + 0.01))
            
            # Normalize weights
            src_nodes = list(src_weights.keys())
            src_w = [src_weights[n] for n in src_nodes]
            src_norm = [w/sum(src_w) for w in src_w] if sum(src_w) > 0 else None
            
            tgt_nodes = list(tgt_weights.keys())
            tgt_w = [tgt_weights[n] for n in tgt_nodes]
            tgt_norm = [w/sum(tgt_w) for w in tgt_w] if sum(tgt_w) > 0 else None
            
            # Add weighted edges
            while added < num_edges and attempts < max_attempts:
                s = np.random.choice(src_nodes, p=src_norm)
                t = np.random.choice(tgt_nodes, p=tgt_norm)
                if not G_new.has_edge(s, t):
                    G_new.add_edge(s, t)
                    added += 1
                attempts += 1
        else:
            # Add random edges
            while added < num_edges and attempts < max_attempts:
                s = random.choice(source_nodes)
                t = random.choice(target_nodes)
                if not G_new.has_edge(s, t):
                    G_new.add_edge(s, t)
                    added += 1
                attempts += 1
        
        # Run simulation
        od = OpinionDynamics(G_new, 'polarity')
        return od.run(max_iter=it)
    
    # Define simulation configurations
    simulations = [
        ("Original", None, None, False),  # Original graph, no edges added
        ("Random", pos_nodes, neg_nodes, False),  # Random edges
        ("High Degree", high_degree_pos, high_degree_neg, False),  # High degree
        ("PADS", nodes_pads_pos, nodes_pads_neg, False),  # PADS communities
        ("Non-PADS", non_pads_pos, non_pads_neg, False),  # Non-PADS nodes
        ("W. All", pos_nodes, neg_nodes, True),  # Weighted selection from all nodes
        ("W. PADS", nodes_pads_pos, nodes_pads_neg, True),  # Weighted selection from PADS
        # ("W. MaxFlow-U", nodes_udsp_pos, nodes_udsp_neg, True),  # Weighted selection from MaxFlow-U
        # ("W. MaxFlow-W", nodes_wdsp_pos, nodes_wdsp_neg, True),  # Weighted selection from MaxFlow-W
        # ("W. GIN", nodes_gin_pos, nodes_gin_neg, True)  # Weighted selection from GIN
    ]
    
    # Run simulations
    results = []
    for name, src, tgt, weighted in simulations:
        if name == "Original":
            od = OpinionDynamics(G, 'polarity')
            results.append((name, *od.run(max_iter=it)))
        else:
            results.append((name, *add_edges_between(src, tgt, weighted)))
    
    # Colors and markers for plotting
    colors = {
        "Original": "#EA8379",
        "Random": "#7DAEE0",
        "High Degree": "#B395BD",
        "PADS": "#299D8F",
        "Non-PADS": "#E9C46A",
        "W. All": "#7D9E72",
        "W. PADS": "#B08970",
        "W. MaxFlow-U": "#F2A900",
        "W. MaxFlow-W": "#D00000",
        "W. GIN": "#8E44AD"
    }
    
    markers = {
        "Original": 'o',
        "Random": '*',
        "High Degree": 'D',
        "PADS": '^',
        "Non-PADS": 'X',
        "W. All": 's',
        "W. PADS": 'v',
        "W. MaxFlow-U": 'P',
        "W. MaxFlow-W": 'h',
        "W. GIN": 'd'
    }
    
    time_steps = range(len(results[0][1]))  # Time steps from first simulation    
    
    # Plot variance
    if ax_var:
        legend_handles = []
        legend_labels = []
        
        for name, vars_data, avg_pos, avg_neg in results:
            label = f'{name}'
            color = colors[name]
            marker = markers[name]
            
            # Plot line with alpha=0.9
            ax_var.plot(time_steps, vars_data, linestyle='-', 
                      color=color, alpha=0.9, linewidth=1)
            
            # Plot scatter with alpha=0.7
            ax_var.scatter(time_steps, vars_data, marker=marker,
                         color=color, alpha=0.7, s=9)
            
            # Create a custom handle for the legend that combines line and marker
            legend_handle = Line2D([0], [0], color=color, marker=marker, 
                                  linestyle='-', markersize=5, label=label)
            legend_handles.append(legend_handle)
            legend_labels.append(label)
        
        if show_ylabel:
            ax_var.set_ylabel('Leaning Variance', fontsize=10)
        
        if show_legend:
            ax_var.legend(handles=legend_handles, labels=legend_labels,
                        loc='best', fontsize=8, frameon=True, edgecolor='grey', 
                        framealpha=0.7, title='Add Edges', title_fontsize=8)
    
    # Plot opinion gap
    if ax_diff:
        legend_handles = []
        legend_labels = []
        
        for name, vars_data, avg_pos, avg_neg in results:
            # Calculate opinion gap
            diff = [p - n for p, n in zip(avg_pos, avg_neg)]
            
            label = f'{name}'
            color = colors[name]
            marker = markers[name]
            
            # Plot line with alpha=0.9
            ax_diff.plot(time_steps, diff, linestyle='-', 
                       color=color, alpha=0.9, linewidth=1)
            
            # Plot scatter with alpha=0.7
            ax_diff.scatter(time_steps, diff, marker=marker,
                          color=color, alpha=0.7, s=9)
            
            # Create a custom handle for the legend that combines line and marker
            legend_handle = Line2D([0], [0], color=color, marker=marker, 
                                  linestyle='-', markersize=5, label=label)
            legend_handles.append(legend_handle)
            legend_labels.append(label)
        
        ax_diff.set_ylabel('Opinion Gap (Avg Pos - Avg Neg)', fontsize=8)
        
        if show_legend:
            ax_diff.legend(handles=legend_handles, labels=legend_labels,
                         loc='best', fontsize=8, frameon=True, edgecolor='grey',
                         framealpha=0.7, title='Add Edges', title_fontsize=8)


def opinion_dynamics_reweight(G, ax_var=None, ax_diff=None, show_legend=False, ratio=0.3, show_ylabel=False):
    od = OpinionDynamics(G, 'polarity', ratio)
    
    # Define colors and methods
    methods = {
        'No Reweight': {'color': "#EA8379", 'marker': 'o', 'nodes': []},
        'Random': {'color': "#7DAEE0", 'marker': '*', 'nodes': None},  
        'High Degree': {'color': "#B395BD", 'marker': 'D', 'nodes': None},  
        'MaxFlow-U': {'color': "#299D8F", 'marker': '^', 'nodes': None},  
        'MaxFlow-W': {'color': "#E9C46A", 'marker': 'X', 'nodes': None},  
        'GIN': {'color': "#7D9E72", 'marker': 's', 'nodes': None},  
        'PADS': {'color': "#B08970", 'marker': 'v', 'nodes': None}  
    }
    
    # Find border nodes for different methods
    border_nodes_pads = [node for node in G.nodes if G.nodes[node]['pads_cpp'] != 0 and 
                         any(G.nodes[neigh]['pads_cpp'] == 0 for neigh in G.neighbors(node))]
    methods['PADS']['nodes'] = border_nodes_pads
    
    # Random nodes for comparison (same count as border_nodes_pads)
    methods['Random']['nodes'] = random.sample(list(G.nodes), len(border_nodes_pads))
    
    # High degree nodes
    pos_border_count = len([n for n in border_nodes_pads if G.nodes[n]['pads_cpp'] > 0])
    neg_border_count = len([n for n in border_nodes_pads if G.nodes[n]['pads_cpp'] < 0])
    pos_nodes = [node for node in G.nodes if G.nodes[node]['polarity'] > 0]
    neg_nodes = [node for node in G.nodes if G.nodes[node]['polarity'] < 0]
    pos_nodes_sorted = sorted(pos_nodes, key=lambda x: G.degree(x), reverse=True)[:pos_border_count]
    neg_nodes_sorted = sorted(neg_nodes, key=lambda x: G.degree(x), reverse=True)[:neg_border_count]
    methods['High Degree']['nodes'] = pos_nodes_sorted + neg_nodes_sorted
    
    # Get border nodes for other methods
    methods['MaxFlow-U']['nodes'] = [n for n in G.nodes if G.nodes[n]['maxflow_cpp_udsp'] != 0 and any(G.nodes[neigh]['maxflow_cpp_udsp'] == 0 for neigh in G.neighbors(n))]
    
    methods['MaxFlow-W']['nodes'] = [n for n in G.nodes if G.nodes[n]['maxflow_cpp_wdsp'] != 0 and any(G.nodes[neigh]['maxflow_cpp_wdsp'] == 0 for neigh in G.neighbors(n))]
    
    methods['GIN']['nodes'] = [n for n in G.nodes if G.nodes[n]['node2vec_gin'] != 0 and any(G.nodes[neigh]['node2vec_gin'] == 0 for neigh in G.neighbors(n))]
    
    # Create legend handles
    legend_handles = []
    legend_labels = []
    
    # Run simulations for each method
    time_steps = None  # Will be set during the first simulation
    
    for method_name, method_info in methods.items():
        vars_data, avg_pos, avg_neg = od.run(reweight=method_info['nodes'])
        
        if time_steps is None:
            time_steps = range(len(vars_data))
        
        # Plot variance data
        if ax_var is not None:
            # Plot line with higher alpha (0.9)
            ax_var.plot(time_steps, vars_data, linestyle='-', linewidth=1, 
                      color=method_info['color'], alpha=0.9, label=method_name)
            
            # Plot scatter with lower alpha (0.7)
            ax_var.scatter(time_steps, vars_data, marker=method_info['marker'], 
                         s=9, color=method_info['color'], alpha=0.7)
            
            # Create a custom legend handle that combines line and marker
            legend_handle = Line2D([0], [0], color=method_info['color'], marker=method_info['marker'], 
                                  linestyle='-', markersize=5, label=method_name)
            legend_handles.append(legend_handle)
            legend_labels.append(method_name)
        
        # Plot opinion gap data
        if ax_diff is not None:
            diff = [p - n for p, n in zip(avg_pos, avg_neg)]
            
            # Plot line with higher alpha (0.9)
            ax_diff.plot(time_steps, diff, linestyle='-', linewidth=1, 
                       color=method_info['color'], alpha=0.9, label=method_name)
            
            # Plot scatter with lower alpha (0.7)
            ax_diff.scatter(time_steps, diff, marker=method_info['marker'], 
                          s=9, color=method_info['color'], alpha=0.7)
    
    # Add labels and legend
    if ax_var is not None and show_ylabel:
        ax_var.set_ylabel('Leaning Variance', fontsize=10)
    
    if ax_diff is not None:
        ax_diff.set_ylabel('Opinion Gap (Avg Pos - Avg Neg)', fontsize=8)
    
    if show_legend:
        if ax_var is not None:
            ax_var.legend(handles=legend_handles, labels=legend_labels,
                        loc='upper right', fontsize=8, frameon=True, edgecolor='grey', 
                        framealpha=0.7, title='Reweight', title_fontsize=8)
        if ax_diff is not None:
            ax_diff.legend(handles=legend_handles, labels=legend_labels,
                         loc='upper right', fontsize=8, frameon=True, edgecolor='grey', 
                         framealpha=0.7, title='Reweight', title_fontsize=8)
            
    # print number of nodes in each method
    for method_name, method_info in methods.items():
        print(f"{method_name}: {len(method_info['nodes'])} nodes")
