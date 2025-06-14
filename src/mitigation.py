import numpy as np
import networkx as nx
import random
import os
import json
import time
from Related_Reps.ged.modules.ps import ge
from Related_Reps.RePBubLik.RepBublik.RWC import RWC_reduction,compute_rwc
from Related_Reps.conflictrisk_public.WCR import WCR
from Related_Reps.RePBubLik.RepBublik.RePBubLik import RePBubLik
from Related_Reps.minimizing_polarization.CD import CD

class MitigationComparison:
    def __init__(self, G=None, graph_file=None, dataset='Referendum', strategies=None, num_edges=1000, random_seed=42):
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        if G is not None:
            self.G = G.copy()
        elif graph_file is not None:
            self.G = nx.read_gml(graph_file)
            # remove self-loops
            self.G.remove_edges_from(nx.selfloop_edges(self.G))
            # check if it is connected, otherwise connect the smaller connect one node 
            # from the non-largest components to one node in the largest component
            if not nx.is_connected(self.G):
                print('Graph is not connected, adding edges to connect the components')
                components = list(nx.connected_components(self.G))
                largest_component = max(components, key=len)
                for component in components:
                    if len(component) < len(largest_component):
                        node = random.choice(list(component))
                        self.G.add_edge(node, random.choice(list(largest_component)))
            # check connectivity again
            if nx.is_connected(self.G):
                print('The graph is connected now')
            # convert node labels to integers
            mapping = {node: int(node) for node in list(self.G.nodes())}
            self.G = nx.relabel_nodes(self.G, mapping)

        else:
            raise ValueError("Either G or graph_file must be provided")
        self.dataset_name = dataset
        self.strategies = self.all_strategies()
        if strategies is not None:
            self.strategies = {k: self.strategies[k] for k in strategies}
        self.num_edges = num_edges

    def get_graph(self):
        return self.G

    def all_strategies(self):
        return {
            # reweight strategies
            'R_O': {'color': "#EA8379", 'marker': 'o', 'nodes': [], 'description': 'Original weights'},
            'R_WCR': {'color': "#7DAEE0", 'marker': '*', 'nodes': None, 'description': 'Reweight edges computed by WCR'},
            'RD_R': {'color': "#7DAEE0", 'marker': '*', 'nodes': None, 'description': 'Decrease weights of random edges'},
            'RI_R': {'color': "#B395BD", 'marker': 'D', 'nodes': None, 'description': 'Increase weights of random edges'},
            'RD_H': {'color': "#299D8F", 'marker': '^', 'nodes': None, 'description': 'Decrease weights of edges with high degree like-minded endpoints'},
            'RI_H': {'color': "#E9C46A", 'marker': 'X', 'nodes': None, 'description': 'Increase weights of edges with high degree opposing endpoints'},
            'RD_MU': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'attr': 'maxflow_cpp_udsp', 'description': 'Decrease weights of edges with one node in MaxFlow-U and the other not'},
            'RI_MU': {'color': "#B08970", 'marker': 'v', 'nodes': None, 'attr': 'maxflow_cpp_udsp', 'description': 'Increase weights of edges with two endpoints in the opposing MaxFlow-U group'},
            'RD_MW': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'attr': 'maxflow_cpp_wdsp', 'description': 'Decrease weights of edges with one node in MaxFlow-W and the other not'},
            'RI_MW': {'color': "#B08970", 'marker': 'v', 'nodes': None, 'attr': 'maxflow_cpp_wdsp', 'description': 'Increase weights of edges with two endpoints in the opposing MaxFlow-W group'},
            'RD_G': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'attr': 'node2vec_gin', 'description': 'Decrease weights of edges with one node in GIN and the other not'},
            'RI_G': {'color': "#B08970", 'marker': 'v', 'nodes': None, 'attr': 'node2vec_gin', 'description': 'Increase weights of edges with two endpoints in the opposing GIN group'},
            'RD_P': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'attr': 'pads_cpp', 'description': 'Decrease weights of edges with one node in PADS and the other not'},
            'RI_P': {'color': "#B08970", 'marker': 'v', 'nodes': None, 'attr': 'pads_cpp', 'description': 'Increase weights of edges with two endpoints in the opposing PADS group'},

            # add edges strategies
            'A_ROV': {'color': "#7DAEE0", 'marker': '*', 'nodes': None, 'description': 'Add edges computed by ROV'},
            'A_RL+': {'color': "#7DAEE0", 'marker': '*', 'nodes': None, 'description': 'Add edges computed by RePBubLik+'},
            'A_CD': {'color': "#7DAEE0", 'marker': '*', 'nodes': None, 'description': 'Add edges computed by CD'},
            'A_R': {'color': "#7DAEE0", 'marker': '*', 'nodes': None, 'description': 'Add non-existing edges with random endpoints'},
            'A_H': {'color': "#B395BD", 'marker': 'D', 'nodes': None, 'description': 'Add non-existing edges with high degree opposing endpoints'},
            'A_P': {'color': "#299D8F", 'marker': '^', 'nodes': None, 'description': 'Add non-existing edges with two endpoints in the opposing PADS group'},
            'A_NP': {'color': "#E9C46A", 'marker': 'X', 'nodes': None, 'description': 'Add non-existing edges with two oppoing endpoints not in the PADS group'},
            'AW_A': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'description': 'Add non-existing edges by weighted selection from all nodes'},
            'AW_P': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'attr': 'pads_cpp', 'description': 'Add non-existing edges by weighted selection from PADS nodes'},
            'AW_MU': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'attr': 'maxflow_cpp_udsp', 'description': 'Add non-existing edges by weighted selection from MaxFlow-U nodes'},
            'AW_MW': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'attr': 'maxflow_cpp_wdsp', 'description': 'Add non-existing edges by weighted selection from MaxFlow-W nodes'},
            'AW_G': {'color': "#7D9E72", 'marker': 's', 'nodes': None, 'attr': 'node2vec_gin', 'description': 'Add non-existing edges by weighted selection from GIN nodes'}
        }
    
    def generate_strategies(self, rerun=False, **kwargs):
        result = {}
        for s in self.strategies:
            MS = MitigationStrategy(self.G, dataset=self.dataset_name)
            if s[0] == 'R':
                new_weights, time_consumed = MS.reweight(strategy=s, num_edges=self.num_edges, attr=self.strategies[s]['attr'] if 'attr' in self.strategies[s] else None, rerun=rerun)
                result[s] = {'new_weights': new_weights, 'time_consumed': time_consumed}
            elif s[0] == 'A':
                new_edges, time_consumed = MS.add_edges(strategy=s, num_edges=self.num_edges, attr=self.strategies[s]['attr'] if 'attr' in self.strategies[s] else None, rerun=rerun, **kwargs)
                result[s] = {'new_edges': new_edges, 'time_consumed': time_consumed}
            else:
                raise ValueError(f"Strategy {s} not implemented")
        return result
        
            

class MitigationMeasurement:
    def __init__(self, G, opinion_list=None):
        self.G = G.copy()
        if opinion_list is not None:
            if isinstance(opinion_list, dict):
                self.op = np.array(list(opinion_list.values()))
            else:
                self.op = np.array(opinion_list)
        else:
            self.op = np.array([self.G.nodes[i]['polarity'] for i in self.G.nodes()])

    def opinion_variance(self):
        return np.var(self.op)

    def opinion_mean(self):
        return np.mean(self.op)

    def opinion_gap(self):
        return np.mean(self.op[self.op > 0]) - np.mean(self.op[self.op < 0])

    def opinion_controversy(self):
        return np.mean(self.op ** 2)

    def rwc(self, perc_rb=0.2, add_edges=[]):
        return compute_rwc(self.G, perc_rb=perc_rb, add_edges=add_edges)
    
    def ged(self):
        # Use opinion values if they are provided
        src = {i: self.op[i] for i in range(len(self.op)) if self.op[i] > 0}
        trg = {i: self.op[i] for i in range(len(self.op)) if self.op[i] <= 0}
        return ge(src, trg, self.G)


class MitigationStrategy:
    def __init__(self, G, opinion_list=None, dataset='Referendum', root_dir='output/mitigation'):
        self.G = G.copy()
        if opinion_list is not None:
            self.op = np.array(opinion_list)
        else:
            self.op = np.array([self.G.nodes[i]['polarity'] for i in self.G.nodes()])
        self.dataset_name = dataset
        self.root_dir = f'{root_dir}/{self.dataset_name}'
        os.makedirs(self.root_dir, exist_ok=True)

    def compute_new_weights(self, edges, strategy='R_O'):
        # this function is just a placeholder for the actual implementation
        # for now, just return a dictionary of edges and their new weights 1 or 0
        new_weights = {}
        for edge in edges:
            if strategy[1] == 'I':
                new_weights[edge] = 1
            elif strategy[1] == 'D':
                new_weights[edge] = 0
            else:
                raise ValueError(f"Not implemented")
        return new_weights

    def reweight(self, strategy='R_O', num_edges=1000, attr=None, rerun=False):
        print(f"Reweighting {strategy} with {num_edges} edges")
        new_weights_file = f'{self.root_dir}/{strategy}_{num_edges}.json'
        time_file = f'{self.root_dir}/time_{num_edges}.json'
        if os.path.exists(new_weights_file) and not rerun:
            with open(new_weights_file, 'r') as f:
                weights_dict = json.load(f)
                # Convert string keys back to tuples
                new_weights = {tuple(map(int, k.strip('()').split(','))): v for k, v in weights_dict.items()}
            with open(time_file, 'r') as f:
                time_data = json.load(f)
                time_consumed = time_data.get(strategy, 0)
        else:
            new_weights_edges = []
            new_weights = {}
            time_start = time.time()
            if strategy == 'R_WCR':
                new_weights = WCR(self.G).run(args={'num_edges': num_edges*2})
            elif strategy in ['RD_MU', 'RD_MW', 'RD_G', 'RD_P']:
                for edge in self.G.edges():
                    if (self.G.nodes[edge[0]][attr] != 0 and self.G.nodes[edge[1]][attr] == 0) or (self.G.nodes[edge[0]][attr] == 0 and self.G.nodes[edge[1]][attr] != 0):
                        new_weights_edges.append(edge)
            elif strategy == 'RD_D':
                pass
            elif strategy != 'R_O':
                raise NotImplementedError(f"Strategy {strategy} not implemented")
            # randomly select num_edges edges from the new_weight list if new_weights_edges is not empty
            if new_weights_edges:
                new_weights_edges = random.sample(new_weights_edges, num_edges)
                new_weights = self.compute_new_weights(new_weights_edges, strategy)

            time_consumed = time.time() - time_start
            # Convert tuple keys to strings for JSON serialization
            weights_dict = {str(k): v for k, v in new_weights.items()}
            with open(new_weights_file, 'w') as f:
                json.dump(weights_dict, f)
            # append the time consumed to the time_{num_edges}.json file and keep the original content
            # if the file does not exist, create it
            if not os.path.exists(time_file):
                time_data = {}
            else:
                with open(time_file, 'r') as f:
                    time_data = json.load(f)
            time_data[strategy] = time_consumed
            with open(time_file, 'w') as f:
                json.dump(time_data, f)

        return new_weights, time_consumed
    
    def compute_weighted_edges(self, attr=None, num_edges=1000):
        source_nodes = [n for n in self.G.nodes() if self.G.nodes[n][attr] == 1]
        target_nodes = [n for n in self.G.nodes() if self.G.nodes[n][attr] == -1]

        src_weights = {n: self.G.degree(n) / (abs(self.G.nodes[n]['polarity']) + 0.01) for n in source_nodes}
        tgt_weights = {n: self.G.degree(n) / (abs(self.G.nodes[n]['polarity']) + 0.01) for n in target_nodes}
        
        src_sum = sum(src_weights.values())
        tgt_sum = sum(tgt_weights.values())
        
        if src_sum == 0 or tgt_sum == 0:
            return []
        src_norm = [w/src_sum for w in src_weights.values()]
        tgt_norm = [w/tgt_sum for w in tgt_weights.values()]
        
        # Pre-compute all possible edges to avoid repeated checks
        possible_edges = [(s, t) for s in source_nodes for t in target_nodes if not self.G.has_edge(s, t)]
        
        if len(possible_edges) < num_edges:
            return possible_edges
            
        # Sample edges using weighted random choice
        new_edges = []
        while len(new_edges) < num_edges:
            s = int(np.random.choice(source_nodes, p=src_norm))  # Convert to Python int
            t = int(np.random.choice(target_nodes, p=tgt_norm))  # Convert to Python int
            if not self.G.has_edge(s, t):
                new_edges.append((s, t))
                
        return new_edges

    def add_edges(self, strategy='A_ROV', num_edges=1000, attr=None, rerun=False, **kwargs):
        print(f"Adding {strategy} with {num_edges} edges")
        new_edges_file = f'{self.root_dir}/{strategy}_{num_edges}.json'
        time_file = f'{self.root_dir}/time_{num_edges}.json'
        if os.path.exists(new_edges_file) and not rerun:
            with open(new_edges_file, 'r') as f:
                new_edges = json.load(f)
            with open(time_file, 'r') as f:
                time_data = json.load(f)
                time_consumed = time_data.get(strategy, 0)
        else:
            new_edges = []
            time_start = time.time()
            if strategy == 'A_ROV':
                ratio = kwargs.get('ratio', 100)
                perc_rb = kwargs.get('perc_rb', 0.2)
                new_edges = RWC_reduction(self.G, unweighted=True, k=num_edges, ratio=ratio, perc_rb=perc_rb)
            elif strategy == 'A_RL+':
                new_edges = RePBubLik(self.G, unweighted=True, maxedges=num_edges)
            elif strategy == 'A_CD':
                _, new_edges = CD(self.G, k=num_edges).coordinate_descent()
            elif strategy in ['AW_P', 'AW_MU', 'AW_MW', 'AW_G']:
                new_edges = self.compute_weighted_edges(attr, num_edges)
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented")

            time_consumed = time.time() - time_start
            with open(new_edges_file, 'w') as f:
                json.dump(new_edges, f)
            # Load or create time data
            if not os.path.exists(time_file):
                time_data = {}
            else:
                with open(time_file, 'r') as f:
                    time_data = json.load(f)
            time_data[strategy] = time_consumed
            with open(time_file, 'w') as f:
                json.dump(time_data, f)

        return new_edges, time_consumed
