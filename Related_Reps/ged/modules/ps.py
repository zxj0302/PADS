import networkx as nx
import numpy as np
from scipy.special import binom
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.linalg import pinv as scipy_pinv

# Function to calculate the pseudoinverse of the Laplacian of the network
def _ge_Q(network, sparse=False):
    A = nx.adjacency_matrix(network).astype(float)
    L = csgraph.laplacian(A, normed=False)
    if sparse:
        n = network.number_of_nodes()
        vals, vecs = eigsh(L, k=n - 1, which='LM', sigma=0, tol=1e-6)
        mask = vals > 1e-10
        vals, vecs = vals[mask], vecs[:, mask]
        return (vecs / vals) @ vecs.T
    else:
        return scipy_pinv(np.asarray(L.todense()))

def ge(src, trg, network, Q=None, sparse=False):
    """Calculate GE for network.

    Parameters:
    ----------
    srg: vector specifying node polarities
    trg: vector specifying node polarities
    network: networkx graph
    Q: pseudoinverse of Laplacian of the network
    sparse: use sparse eigsh instead of dense pinv (recommended for large graphs)
    """
    if nx.number_connected_components(network) > 1:
        raise ValueError("""Node vector distance is only valid if calculated on a network with a single connected component.
                       The network passed has more than one.""")
    src = np.array([src[n] if n in src else 0. for n in network.nodes()])
    trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
    diff = src - trg

    if Q is None:
        Q = _ge_Q(network, sparse=sparse)

    ge_dist = diff.T.dot(np.array(Q).dot(diff))

    if ge_dist < 0:
        ge_dist = 0

    return np.sqrt(ge_dist)

def resistance(network):
    """Calculate effective resistance for each node pair in the network.

    Parameters:
    ----------
    network: networkx graph.
    """

    n = len(network.nodes)
    L = np.asarray(csgraph.laplacian(nx.adjacency_matrix(network).todense().astype(float), normed=False))
    Phi = np.ones((n, n)) / n
    Gamma = scipy_pinv(L + Phi)

    # calculate resistance for all node pairs
    res = np.array(
        [[Gamma[i, i] + Gamma[j, j] - (2 * Gamma[i, j]) if i != j else 0 for j in range(n)] for i in range(n)])

    return res

def ge_multipolar(os, network, Q=None, sparse=False):
    """Calculate multipolar GE for network.

    Parameters:
    ----------
    os: vector specifying all node polarities
    network: networkx graph
    Q: pseudoinverse of Laplacian of the network
    sparse: use sparse eigsh instead of dense pinv (recommended for large graphs)
    """
    if nx.number_connected_components(network) > 1:
        raise ValueError("""Node vector distance is only valid if calculated on a network with a single connected component.
                       The network passed has more than one.""")
    os = [np.array([os[n][i] for n in network.nodes()]) for i in range(os[0].shape[0])]
    if Q is None:
        Q = _ge_Q(network, sparse=sparse)
    conflict_sum = 0
    for i in range(len(os) - 1):
       for j in range(i + 1, len(os)):
          diff = os[i] - os[j]
          conflict_sum += diff.T.dot(np.array(Q).dot(diff))

    ge_dist = conflict_sum / binom(len(os), 2)

    if ge_dist < 0:
        ge_dist = 0

    return np.sqrt(ge_dist)
