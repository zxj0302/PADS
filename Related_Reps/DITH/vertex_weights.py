import networkx as nx
from typing import List, Dict
import math

def single_dijkstra(graph: nx.Graph, source: int) -> Dict[int, float]:
    """
    Single-source shortest distances using NetworkX.
    
    Args:
        graph: NetworkX graph with 'length' edge attribute for distances
        source: Source vertex ID
        
    Returns:
        Dictionary mapping vertex ID to shortest distance from source
    """
    # Use shortest_path_length for distances only (no paths needed)
    return dict(nx.shortest_path_length(graph, source, weight='length'))

def vertex_weights(graph: nx.Graph, A: List[int], R: List[int], 
                  lambda1: float, lambda2: float) -> List[float]:
    """
    Calculate vertex weights based on proximity to set A and distance from set R.
    
    Args:
        graph: NetworkX graph with 'length' edge attribute
        A: List of attraction vertex IDs (single node)
        R: List of repulsion vertex IDs (single node)  
        lambda1: Weight for proximity to A
        lambda2: Weight for distance from R
        
    Returns:
        List of vertex weights (indexed by vertex ID)
    """
    n = graph.number_of_nodes()
    w_v = [0.0] * n
    
    # Since A and R contain only one node each
    source_A = A[0]
    source_R = R[0]
    
    # Calculate distances from attraction node A
    d_A = single_dijkstra(graph, source_A)
    
    # Find Delta (maximum finite distance from A)
    Delta = 0.0
    for v in graph.nodes():
        if v in d_A:
            Delta = max(Delta, d_A[v])
    
    # Add proximity component (higher weight for closer to A)
    for v in graph.nodes():
        if v in d_A:  # Only for reachable vertices from A
            w_v[v] += lambda1 * (Delta - d_A[v])
    
    # Calculate distances from repulsion node R
    d_R = single_dijkstra(graph, source_R)
    
    # Add distance component (higher weight for farther from R)
    for v in graph.nodes():
        if v == source_R:
            w_v[v] += lambda2 * 0.0  # Distance is 0 if v is R
        elif v in d_R:  # Only for reachable vertices from R
            w_v[v] += lambda2 * d_R[v]
    
    return w_v

def create_graph_from_edges(n: int, edges: List[tuple]) -> nx.Graph:
    """
    Helper function to create NetworkX graph from edge list.
    
    Args:
        n: Number of vertices (0 to n-1)
        edges: List of tuples (u, v, length, weight)
        
    Returns:
        NetworkX graph with both 'length' and 'weight' edge attributes
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for u, v, length, weight in edges:
        G.add_edge(u, v, length=length, weight=weight)
    
    return G

# Example usage and testing
if __name__ == "__main__":
    # Create a simple test graph
    #   0---1---2
    #   |   |   |
    #   3---4---5
    edges = [
        (0, 1, 1.0, 1.0),  # (u, v, length, weight)
        (1, 2, 1.0, 1.0),
        (0, 3, 1.0, 1.0),
        (1, 4, 1.0, 1.0),
        (2, 5, 1.0, 1.0),
        (3, 4, 1.0, 1.0),
        (4, 5, 1.0, 1.0)
    ]
    
    g = create_graph_from_edges(6, edges)
    
    # Test single_dijkstra
    source = 0
    distances = single_dijkstra(g, source)
    print("Distances from vertex 0:", distances)
    
    # Test vertex_weights
    A = [0]  # Attraction set
    R = [5]  # Repulsion set
    lambda1 = 1.0
    lambda2 = 1.0
    
    weights = vertex_weights(g, A, R, lambda1, lambda2)
    print("Vertex weights:", weights)
    
    # Additional test: show shortest paths
    print("\nShortest paths from vertex 0:")
    for target in range(6):
        if target in distances:
            try:
                path = nx.shortest_path(g, 0, target, weight='length')
                print(f"  To vertex {target}: distance={distances[target]:.1f}, path={path}")
            except nx.NetworkXNoPath:
                print(f"  To vertex {target}: No path") 