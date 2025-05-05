import os.path
import networkx as nx
import pymetis
import numpy as np
from community import community_louvain
try:
    from Eva import eva_best_partition
except ImportError:
    try:
        from eva import eva_best_partition
    except ImportError:
        raise ImportError("Neither 'Eva' nor 'eva' module could be imported")
from dsd import exact_densest, flowless
import subprocess


def metis_partition(G: nx.Graph, **kwargs) -> None:
    try:
        A = [np.array([int(n) for n in G.neighbors(node)]) for node in G.nodes()]
        (_, metis_return) = pymetis.part_graph(2, A)

        for i, node in enumerate(G.nodes()):
            G.nodes[node]['metis'] = metis_return[i]
    except Exception as e:
        raise RuntimeError(f"METIS partitioning failed: {str(e)}")


def louvain_partition(G: nx.Graph, **kwargs) -> None:
    try:
        louvain_return = community_louvain.best_partition(G)
        for node in G.nodes():
            G.nodes[node]['louvain'] = louvain_return[node]
    except Exception as e:
        raise RuntimeError(f"Louvain partitioning failed: {str(e)}")


def eva_partition(G: nx.Graph, **kwargs) -> None:
    try:
        eva_part, _ = eva_best_partition(G, weight='polarity_label', alpha=0.5)
        for node in G.nodes():
            G.nodes[node]['eva'] = eva_part[node]
    except Exception as e:
        raise RuntimeError(f"Eva partitioning failed: {str(e)}")


def maxflow_python_udsp(G: nx.Graph) -> None:
    try:
        # Create positive and negative graphs
        G_pos, G_neg = G.copy(), G.copy()
        G_pos.remove_nodes_from([node for node in G.nodes() if G.nodes[node]['polarity'] < 0])
        G_neg.remove_nodes_from([node for node in G.nodes() if G.nodes[node]['polarity'] > 0])

        maxflow_pos = exact_densest(G_pos)
        maxflow_neg = exact_densest(G_neg)

        # Assign node values
        pos_nodes = set(maxflow_pos[0])
        neg_nodes = set(maxflow_neg[0])

        for node in G.nodes():
            G.nodes[node]['maxflow'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)
    except Exception as e:
        raise RuntimeError(f"Maxflow computation failed: {str(e)}")


def maxflow_cpp_udsp(G: nx.Graph, **kwargs) -> None:
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\Greedy++\\exact.exe')
    multiplier = kwargs.get('multiplier', 1000)
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_pos_path = os.path.join(input_folder, 'edgelist_pos_unweighted')
    node_map_pos_path = os.path.join(input_folder, 'node_map_pos')
    input_neg_path = os.path.join(input_folder, 'edgelist_neg_unweighted')
    node_map_neg_path = os.path.join(input_folder, 'node_map_neg')

    # run the cpp program and get the output program prints on the terminal
    pos_command = f"{cpp_exe} {multiplier} < {input_pos_path}"
    neg_command = f"{cpp_exe} {multiplier} < {input_neg_path}"

    # Function to read node mapping
    def read_node_map(map_path):
        node_map = {}
        with open(map_path, 'r') as f:
            # Skip the header line
            next(f)
            for line in f:
                new_label, original_label = map(int, line.strip().split())
                node_map[new_label] = original_label
        return node_map

    # Function to run command and process output
    def run_command_and_process(command, node_map_path):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time
        solution_time = None
        for line in output.split('\n'):
            if "Time for finding solution:" in line:
                solution_time = int(line.split(':')[1].strip().split()[0])
                break

        # Read output file (assuming it's created in the current directory)
        nodes = []
        output_file = "soln.tmp"  # Adjust this if the output file name is different
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                node_map = read_node_map(node_map_path)
                for line in f:
                    new_label = int(line.strip())
                    original_label = node_map.get(new_label)
                    if original_label is not None:
                        nodes.append(original_label)
                    else:
                        raise RuntimeError(f"Node {new_label} not found in node map")
        return solution_time, nodes

    # Run for positive edges
    pos_time, pos_nodes = run_command_and_process(pos_command, node_map_pos_path)
    # Run for negative edges
    neg_time, neg_nodes = run_command_and_process(neg_command, node_map_neg_path)

    for node in G.nodes():
        G.nodes[node]['maxflow_cpp_udsp'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    # delete the  soln.tmp file
    os.remove('soln.tmp')

    # Return both times and node sets
    return (pos_time+neg_time)/1000.0, (pos_nodes, neg_nodes)




def maxflow_cpp_wdsp(G: nx.Graph, **kwargs) -> None:
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\Greedy++\\exactweighted.exe')
    multiplier = kwargs.get('multiplier', 1000)
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_pos_path = os.path.join(input_folder, 'edgelist_pos_weighted')
    node_map_pos_path = os.path.join(input_folder, 'node_map_pos')
    input_neg_path = os.path.join(input_folder, 'edgelist_neg_weighted')
    node_map_neg_path = os.path.join(input_folder, 'node_map_neg')

    # run the cpp program and get the output program prints on the terminal
    pos_command = f"{cpp_exe} {multiplier} < {input_pos_path}"
    neg_command = f"{cpp_exe} {multiplier} < {input_neg_path}"

    # Function to read node mapping
    def read_node_map(map_path):
        node_map = {}
        with open(map_path, 'r') as f:
            # Skip the header line
            next(f)
            for line in f:
                new_label, original_label = map(int, line.strip().split())
                node_map[new_label] = original_label
        return node_map

    # Function to run command and process output
    def run_command_and_process(command, node_map_path):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time
        solution_time = None
        for line in output.split('\n'):
            if "Time for finding solution:" in line:
                solution_time = int(line.split(':')[1].strip().split()[0])
                break

        # Read output file (assuming it's created in the current directory)
        nodes = []
        output_file = "soln.tmp"  # Adjust this if the output file name is different
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                node_map = read_node_map(node_map_path)
                for line in f:
                    new_label = int(line.strip())
                    original_label = node_map.get(new_label)
                    if original_label is not None:
                        nodes.append(original_label)
                    else:
                        raise RuntimeError(f"Node {new_label} not found in node map")
        return solution_time, nodes

    # Run for positive edges
    pos_time, pos_nodes = run_command_and_process(pos_command, node_map_pos_path)
    # Run for negative edges
    neg_time, neg_nodes = run_command_and_process(neg_command, node_map_neg_path)

    for node in G.nodes():
        G.nodes[node]['maxflow_cpp_wdsp'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    # delete the  soln.tmp file
    os.remove('soln.tmp')

    # Return both times and node sets
    return (pos_time+neg_time)/1000.0, (pos_nodes, neg_nodes)


def greedypp_python_wdsp(G: nx.Graph, **kwargs) -> None:
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
            G.nodes[node]['greedypp_python_wdsp'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)
    except Exception as e:
        raise RuntimeError(f"Greedy++ computation failed: {str(e)}")


def greedypp_cpp_wdsp(G: nx.Graph, **kwargs) -> None:
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\Greedy++\\ipnw.exe')
    iterations = kwargs.get('iterations', 1)
    dataset = kwargs.get('dataset', 'Abortion')
    input_folder = kwargs.get('input_file', f'input\\datasets\\static\\{dataset}')
    input_pos_path = os.path.join(input_folder, 'edgelist_pos_weighted')
    node_map_pos_path = os.path.join(input_folder, 'node_map_pos')
    input_neg_path = os.path.join(input_folder, 'edgelist_neg_weighted')
    node_map_neg_path = os.path.join(input_folder, 'node_map_neg')

    # run the cpp program and get the output program prints on the terminal
    pos_command = f"{cpp_exe} {iterations} < {input_pos_path}"
    neg_command = f"{cpp_exe} {iterations} < {input_neg_path}"

    # Function to read node mapping
    def read_node_map(map_path):
        node_map = {}
        with open(map_path, 'r') as f:
            # Skip the header line
            next(f)
            for line in f:
                new_label, original_label = map(int, line.strip().split())
                node_map[new_label] = original_label
        return node_map

    # Function to run command and process output
    def run_command_and_process(command, node_map_path):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time
        solution_time = None
        for line in output.split('\n'):
            if "Avg time per iteration:" in line:
                solution_time = int(line.split(':')[1].strip().split()[0]) * iterations
                break

        # Read output file (assuming it's created in the current directory)
        nodes = []
        output_file = "soln.tmp"  # Adjust this if the output file name is different
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                node_map = read_node_map(node_map_path)
                for line in f:
                    new_label = int(line.strip())
                    original_label = node_map.get(new_label)
                    if original_label is not None:
                        nodes.append(original_label)
                    else:
                        raise RuntimeError(f"Node {new_label} not found in node map")
        return solution_time, nodes

    # Run for positive edges
    pos_time, pos_nodes = run_command_and_process(pos_command, node_map_pos_path)
    # Run for negative edges
    neg_time, neg_nodes = run_command_and_process(neg_command, node_map_neg_path)

    for node in G.nodes():
        G.nodes[node]['greedypp_cpp_wdsp'] = (1 if node in pos_nodes else 0) - (1 if node in neg_nodes else 0)

    # delete the  soln.tmp file
    os.remove('soln.tmp')

    # Return both times and node sets
    return (pos_time+neg_time)/1000, (pos_nodes, neg_nodes)