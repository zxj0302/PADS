import networkx as nx
import pymetis
import numpy as np
from community import community_louvain
from eva import eva_best_partition
from dsd import exact_densest, flowless


def metis_partition(G: nx.Graph) -> None:
    """
    Partition graph using METIS algorithm.

    Args:
        G: Input networkx graph
    """
    if not G.nodes():
        raise ValueError("Empty graph")

    try:
        A = [np.array([int(n) for n in G.neighbors(node)]) for node in G.nodes()]
        (_, metis_return) = pymetis.part_graph(2, A)

        for i, node in enumerate(G.nodes()):
            G.nodes[node]['metis'] = metis_return[i]
    except Exception as e:
        raise RuntimeError(f"METIS partitioning failed: {str(e)}")


def louvain_partition(G: nx.Graph) -> None:
    """
    Detect communities using Louvain algorithm.

    Args:
        G: Input networkx graph
    """
    if not G.nodes():
        raise ValueError("Empty graph")

    try:
        louvain_return = community_louvain.best_partition(G)
        for node in G.nodes():
            G.nodes[node]['louvain'] = louvain_return[node]
    except Exception as e:
        raise RuntimeError(f"Louvain partitioning failed: {str(e)}")


def eva_partition(G: nx.Graph) -> None:
    """
    Detect communities using Eva algorithm.

    Args:
        G: Input networkx graph
    """
    if not G.nodes():
        raise ValueError("Empty graph")

    try:
        eva_part, _ = eva_best_partition(G, weight='stance_label', alpha=0.5)
        for node in G.nodes():
            G.nodes[node]['eva'] = eva_part[node]
    except Exception as e:
        raise RuntimeError(f"Eva partitioning failed: {str(e)}")


def maxflow_udsp(G: nx.Graph) -> None:
    """
    Compute the densest subgraph using max flow algorithm.

    Args:
        G: Input networkx graph
    """
    if not G.nodes():
        raise ValueError("Empty graph")

    try:
        # Create positive and negative graphs
        G_pos = G.copy()
        G_neg = G.copy()

        # Filter edges based on polarity
        G_pos.remove_edges_from([(u, v) for u, v, d in G_pos.edges(data=True)
                                 if d['edge_polarity'] < 0])
        G_neg.remove_edges_from([(u, v) for u, v, d in G_neg.edges(data=True)
                                 if d['edge_polarity'] > 0])

        # Remove isolated nodes
        G_pos.remove_nodes_from(list(nx.isolates(G_pos)))
        G_neg.remove_nodes_from(list(nx.isolates(G_neg)))

        maxflow_pos = exact_densest(G_pos)
        maxflow_neg = exact_densest(G_neg)

        # Assign node values
        pos_nodes = set(maxflow_pos[0])
        neg_nodes = set(maxflow_neg[0])

        for node in G.nodes():
            G.nodes[node]['maxflow'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)
    except Exception as e:
        raise RuntimeError(f"Maxflow computation failed: {str(e)}")


def greedypp_wdsp(G: nx.Graph, **kwargs) -> None:
    """
    Compute the densest subgraph using Greedy++ algorithm.

    Args:
        G: Input networkx graph
        iterations: Number of iterations for the algorithm
    """

    if not G.nodes():
        raise ValueError("Empty graph")
    try:
        iterations = kwargs.get('iterations', 10)
        # print(f"Running Greedy++ with {iterations} iterations")

        # Option 1: make all edges positive
        # for u, v, d in G.edges(data=True):
        #     G.edges[u, v]['edge_polarity_plus1'] = d['edge_polarity'] + 1
        #     G.edges[u, v]['edge_polarity_minus1_abs'] = abs(d['edge_polarity'] - 1)
        #
        # flowless_pos = flowless(G, iterations, 'edge_polarity_plus1')
        # flowless_neg = flowless(G, iterations, 'edge_polarity_minus1_abs')

        # Option 2: delete opposite nodes
        G_pos, G_neg = G.copy(), G.copy()
        G_pos.remove_nodes_from([node for node in G.nodes() if G.nodes[node]['polarity'] < 0])
        G_neg.remove_nodes_from([node for node in G.nodes() if G.nodes[node]['polarity'] > 0])
        # reverse the node polarity in G_neg
        for node in G_neg.nodes():
            G_neg.nodes[node]['polarity'] = -G_neg.nodes[node]['polarity']
        flowless_pos = flowless(G_pos, iterations)
        flowless_neg = flowless(G_neg, iterations)


        pos_nodes = set(flowless_pos[0])
        neg_nodes = set(flowless_neg[0])

        for node in G.nodes():
            G.nodes[node]['flowless'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)
    except Exception as e:
        raise RuntimeError(f"Greedy++ computation failed: {str(e)}")
