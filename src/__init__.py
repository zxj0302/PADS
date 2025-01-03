from typing import Dict
import time
from .baselines import *
from .greedy import my_greedy
from .gnn import my_gnn
from .utils import *
from .painter import *
from .diffusion import *


def run_exp(G: nx.Graph, method: str, timer: Dict[str, float], **kwargs) -> None:
    """
    Run the specified community detection method and time its execution.

    Args:
        G: Input networkx graph
        method: Name of the method to run
        timer: Dictionary to store execution times
        **kwargs: Additional arguments for specific methods
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("Input must be a networkx graph")

    method_map = {
        'metis': metis_partition,
        'louvain': louvain_partition,
        'eva': eva_partition,
        'maxflow': maxflow_udsp,
        'flowless': greedypp_wdsp,
        'my_greedy': my_greedy,
        'my_gnn': my_gnn
    }

    if method not in method_map:
        raise ValueError(f"Invalid method. Choose from {list(method_map.keys())}")

    try:
        start = time.time()
        method_map[method](G, **kwargs) if method == 'flowless' else method_map[method](G)
        timer[method] = round(time.time() - start, 3)
    except Exception as e:
        timer[method] = -1  # Indicate failure
        raise RuntimeError(f"Method {method} failed: {str(e)}")
