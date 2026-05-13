import networkx as nx
import numpy as np


def _prepare_graph(G):
    graph = G.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))

    if graph.number_of_nodes() == 0:
        return graph

    if nx.is_connected(graph):
        return graph

    largest_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_component).copy()


def compute_mht_candidate_edges(G, num_edges=1000, seed=None):
    graph = _prepare_graph(G)

    red_nodes = [node for node in graph.nodes() if graph.nodes[node].get('polarity', 0) > 0]
    blue_nodes = [node for node in graph.nodes() if graph.nodes[node].get('polarity', 0) <= 0]

    candidate_edges = [
        (u, v)
        for u in red_nodes
        for v in blue_nodes
        if u != v and not graph.has_edge(u, v)
    ]

    if not candidate_edges or num_edges <= 0:
        return []

    if len(candidate_edges) <= num_edges:
        return candidate_edges

    if seed is None:
        chosen = np.random.choice(len(candidate_edges), size=num_edges, replace=False)
    else:
        chosen = np.random.default_rng(seed).choice(len(candidate_edges), size=num_edges, replace=False)

    return [candidate_edges[i] for i in chosen]