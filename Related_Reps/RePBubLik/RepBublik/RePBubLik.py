import os
import math
from time import time
import pickle
import numpy as np
import networkx as nx
from scipy.sparse import diags
from . import parallel_walks 
from . import parallel_centrality
from .algo import Algorithms
from .load_data import LoadData
from .random_walks import chunk_random_walk
from .utils import load_bubble_diameter, get_bad_and_good_nodes, get_centralities, sort_edges
from .RWC import MyGraphData

def run_fair_random_walks(args):
    politics = MyGraphData(G=args.G, uniform=args.unweighted)

    id_color = politics.id_color
    G = politics.G
    edges_updated = set(list(G.edges()))

    # Indeces for the adj matrix
    node_id_matrix = {n:i for i,n in enumerate(list(G.nodes()))}
    # print(node_id_matrix)
    id_matrix_node = {j:i for i,j in node_id_matrix.items()}
    # print(id_matrix_node)
    # Define adj matrix
    A = nx.adjacency_matrix(G)
    d = diags(1/A.sum(axis=1).ravel())
    A = A.T.dot(d).T

    # Order nodes labels and colors
    labels = np.array([j for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])
    color_nodes = np.array([id_color[j] for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])

    # Define RW parameters
    delta = 0.05
    eps = 1
    r = int((args.t**2)/(eps**2)*np.log(1/delta))
    print("Number of walks per node: ", r)

    if args.proc == 'radius':
        if not os.path.exists(args.topic + '/'):
            os.makedirs(args.topic + '/')

        # Compute bubble diameter
        tt = time()
        result_chunks = parallel_walks.parallelization(A, labels, args.t, color_nodes, r)
        with open(args.topic + '/'  + args.topic + '_' + str(args.t) + '_bubble_diameters.pickle', 'wb') as f:
            pickle.dump(result_chunks, f)
        print(time()-tt)

        bubble_diameter = load_bubble_diameter(args.topic, id_matrix_node, args.t)
        red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices = get_bad_and_good_nodes(bubble_diameter, id_color, args.b, args.t)
        
        # Compute bad nodes centralities Blue
        nodes = np.array([node_id_matrix[n] for n in bad_blue_vertices])
        print('Number blue bad nodes: ', len(nodes))
        print("Number of walks per node: ", r)

        tt = time()
        result_chunks = parallel_centrality.parallelization(A, labels, args.t, color_nodes, r, nodes)
        with open(args.topic + '/'  + args.topic + '_' + 'blue' + '_' + str(args.t) + '_centralities.pickle', 'wb') as f:
            pickle.dump(result_chunks, f)
        print(time()-tt)

        # Compute bad nodes centralities Red
        nodes = np.array([node_id_matrix[n] for n in bad_red_vertices])
        print('Number red bad nodes: ', len(nodes))
        print("Number of walks per node: ", r)

        tt = time()
        result_chunks = parallel_centrality.parallelization(A, labels, args.t, color_nodes, r, nodes)
        with open(args.topic + '/'  + args.topic + '_' + 'red' + '_' + str(args.t) + '_centralities.pickle', 'wb') as f:
            pickle.dump(result_chunks, f)
        print(time()-tt)

    elif args.proc == 'addition':
        bubble_diameter = load_bubble_diameter(args.topic, id_matrix_node, args.t)
        red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices = get_bad_and_good_nodes(bubble_diameter, id_color, args.b, args.t)
        
        # Get partitions' contribution
        s_red = np.sum(list(bad_red_vertices.values()))
        s_blue = np.sum(list(bad_blue_vertices.values()))

        perc_red = s_red/(s_red + s_blue)

        min_part = min(len(bad_red_vertices)*len(blue_bubble_diameter), len(bad_blue_vertices)*len(red_bubble_diameter))
        K_int = np.append([1],np.arange(2, min(min_part, args.maxedges), 2))
        
        K_R = [math.floor(i*perc_red) for i in K_int]
        K_B = [k - K_R[i] for i, k in enumerate(K_int)]

        all_edges = []

        for c in ['blue', 'red']:
            if c=='blue':
                bad = bad_blue_vertices
                K = K_B
            else:
                bad = bad_red_vertices
                K = K_R
            
            nodes = np.array([node_id_matrix[n] for n in bad])
            
            if c=='blue':
                alg = Algorithms(bad, red_bubble_diameter, labels, args.t, 
                            color_nodes, r, nodes, node_id_matrix, edges_updated, 0)
            else:
                alg = Algorithms(bad, blue_bubble_diameter, labels, args.t, 
                            color_nodes, r, nodes, node_id_matrix, edges_updated, 0)
            
            for ITER in range(int(args.iter)):
                if c == 'blue':
                    alg = Algorithms(bad, red_bubble_diameter, labels, args.t, 
                                    color_nodes, r, nodes, node_id_matrix, edges_updated, ITER)
                else:
                    alg = Algorithms(bad, blue_bubble_diameter, labels, args.t, 
                                    color_nodes, r, nodes, node_id_matrix, edges_updated, ITER)
                                        
                # Recall centrality
                centrality = get_centralities(id_matrix_node, args.topic, c, args.t, old=False)
                perc = args.topk
                top_k = int(len(centrality)/100*perc)

                # Pick sorted degree weighted central edges
                candidate_edges = alg._compute_candidate_edge('w_pen_central', top_k, centrality, G)
                new_edges = sort_edges(candidate_edges, K)
                all_edges.extend(new_edges)
                print(f'Number of new_edges: {len(new_edges)}')
                print(new_edges)
        return all_edges

def RePBubLik(G, t=10, b=2, topk=10, maxedges=1000, iter=1, unweighted=True, topic='GraphOutput'):
    # Create an args object with attributes instead of a dictionary
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Extract topic from graph_file if not provided
    # if topic is None:
    #     # Extract topic name from the graph file path
    #     # Assumes format like "data/guns/guns.gpickle"
    #     filename = os.path.basename(graph_file)
    #     topic = os.path.splitext(filename)[0]  # Remove extension
    
    args = Args(
        G=G,
        t=t,
        b=b,
        topk=topk,
        maxedges=maxedges,
        iter=iter,
        unweighted=unweighted,
        topic=topic,
        proc=None  # Placeholder for process type
    )
    
    # First phase: compute bubble diameters
    args.proc = 'radius'
    run_fair_random_walks(args)
    
    # Second phase: add edges
    args.proc = 'addition'
    new_edges = run_fair_random_walks(args)
    
    return new_edges