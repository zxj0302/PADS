import math
from sortedcontainers import SortedSet
import subprocess


def ecc_greedy(G, theta, pos=True, max_neg_count=200, return_fs=False):
    if not pos:
        for node in G.nodes():
            G.nodes[node]['polarity'] = -G.nodes[node]['polarity']
            G.nodes[node]['polarity_label'] = -G.nodes[node]['polarity_label']
        for edge in G.edges():
            G.edges[edge]['edge_polarity'] = -G.edges[edge]['edge_polarity']
    node_promising = max([x for x in G.nodes() if G.nodes[x]['polarity_label']==1], key=lambda x: G.nodes[x]['promising_value'])

    # initialize the selected sets, to_select sets, status of each node and priority key(marginal gain) of each node
    polarity_sum = 0
    num_selected_neu_pos_neg = [0, 0, 0]
    selected_neu_pos_neg = [SortedSet(), SortedSet(), SortedSet()]
    to_select_neu_pos_neg = [SortedSet(), SortedSet(), SortedSet()]
    for node in G.nodes():
        G.nodes[node]['status'] = 'out'
        G.nodes[node]['priority_key'] = None
        # record the number of selected neighbors of a fringe node, it will be used to determine the status of the node
        G.nodes[node]['in_neighbor_count'] = 0
    G.nodes[node_promising]['status'] = 'fringe'
    G.nodes[node_promising]['priority_key'] = G.edges[(node_promising, node_promising)]['edge_polarity'] if G.has_edge(node_promising, node_promising) else 0
    to_select_neu_pos_neg[G.nodes[node_promising]['polarity_label']].add((G.nodes[node_promising]['priority_key'], G.nodes[node_promising]['promising_value'], node_promising))

    next_node = node_promising
    max_f = -math.inf
    neg_count = 0
    best_selected = None
    if return_fs:
        fs = []
    while next_node is not None and neg_count < max_neg_count:
        # FIXME: consider the self-loop edges
        polarity_label = G.nodes[next_node]['polarity_label']
        if G.nodes[next_node]['status'] == 'fringe':
            G.nodes[next_node]['status'] = 'in'
            item = to_select_neu_pos_neg[polarity_label].pop()
            selected_neu_pos_neg[polarity_label].add(item)
            polarity_sum += item[0]
            num_selected_neu_pos_neg[polarity_label] += 1
            # update the status of its neighbors
            for neighbor in G.neighbors(next_node):
                G.nodes[neighbor]['in_neighbor_count'] += 1
                neighbor_polarity_label = G.nodes[neighbor]['polarity_label']
                if G.nodes[neighbor]['status'] == 'out':
                    G.nodes[neighbor]['status'] = 'fringe'
                    G.nodes[neighbor]['priority_key'] = G.edges[(next_node, neighbor)]['edge_polarity'] + (G.edges[(neighbor, neighbor)]['edge_polarity'] if G.has_edge(neighbor, neighbor) else 0)
                    to_select_neu_pos_neg[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                elif G.nodes[neighbor]['status'] == 'fringe':
                    to_select_neu_pos_neg[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                    G.nodes[neighbor]['priority_key'] += G.edges[(next_node, neighbor)]['edge_polarity']
                    to_select_neu_pos_neg[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                elif G.nodes[neighbor]['status'] == 'in':
                    if neighbor != next_node:
                        selected_neu_pos_neg[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                        G.nodes[neighbor]['priority_key'] += G.edges[(next_node, neighbor)]['edge_polarity']
                        selected_neu_pos_neg[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                else:
                    raise ValueError(f"Error: {G.nodes[neighbor]['status']} cannot be a valid neighbor status of fringe nodes")
        elif G.nodes[next_node]['status'] == 'in':
            G.nodes[next_node]['status'] = 'fringe'
            item = selected_neu_pos_neg[polarity_label].pop(0)
            to_select_neu_pos_neg[polarity_label].add(item)
            polarity_sum -= item[0]
            num_selected_neu_pos_neg[polarity_label] -= 1
            # update the status of its neighbors
            for neighbor in G.neighbors(next_node):
                G.nodes[neighbor]['in_neighbor_count'] -= 1
                neighbor_polarity_label = G.nodes[neighbor]['polarity_label']
                if G.nodes[neighbor]['status'] == 'fringe':
                    if G.nodes[neighbor]['in_neighbor_count'] == 0:
                        to_select_neu_pos_neg[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                        # NOTICE: no node selected now, stop the algorithm
                        G.nodes[neighbor]['status'] = 'out'
                        G.nodes[neighbor]['priority_key'] = None
                    elif neighbor != next_node:
                        to_select_neu_pos_neg[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                        G.nodes[neighbor]['priority_key'] -= G.edges[(next_node, neighbor)]['edge_polarity']
                        to_select_neu_pos_neg[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                elif G.nodes[neighbor]['status'] == 'in':
                    # NOTE: can remove the node with 'in_neighbor_count' == 0, but it is not necessary, as it may be
                    # removed in some later iteration
                    selected_neu_pos_neg[neighbor_polarity_label].remove((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                    G.nodes[neighbor]['priority_key'] -= G.edges[(next_node, neighbor)]['edge_polarity']
                    selected_neu_pos_neg[neighbor_polarity_label].add((G.nodes[neighbor]['priority_key'], G.nodes[neighbor]['promising_value'], neighbor))
                else:
                    raise ValueError(f"Error: {G.nodes[neighbor]['status']} cannot be a valid neighbor status of in nodes")
        else:
            raise ValueError(f"Error: {G.nodes[next_node]['status']} cannot be a valid status to add or remove")

        # find the next node with the highest & positive marginal gain, set None if no positive
        marginal_gains = []
        num_selected_now = sum(num_selected_neu_pos_neg)
        polarities = [x / num_selected_now for x in num_selected_neu_pos_neg]
        entropy = -sum([p*math.log2(p) for p in polarities if p != 0])
        value_old = polarity_sum/num_selected_now-theta*entropy
        if value_old >= max_f:
            max_f = value_old
            best_selected = [item.copy() for item in selected_neu_pos_neg]
        if return_fs:
            fs.append((value_old, polarity_sum, num_selected_now, entropy))
        addition_idx = []
        for i in range(3):
            if len(selected_neu_pos_neg[i]) > 0:
                item = selected_neu_pos_neg[i].__getitem__(0)
                temp_label_distribution = num_selected_neu_pos_neg.copy()
                temp_label_distribution[G.nodes[item[2]]['polarity_label']] -= 1
                temp_polarities = [x/(num_selected_now-1) for x in temp_label_distribution if x != 0]
                temp_entropy = -sum([p*math.log2(p) for p in temp_polarities if p != 0])
                mg = ((((polarity_sum-item[0])/(num_selected_now-1)) if num_selected_now > 1 else 0)-theta*temp_entropy) - value_old
                marginal_gains.append((mg, item[2]))
            if len(to_select_neu_pos_neg[i]) > 0:
                item = to_select_neu_pos_neg[i].__getitem__(-1)
                temp_label_distribution = num_selected_neu_pos_neg.copy()
                temp_label_distribution[G.nodes[item[2]]['polarity_label']] += 1
                temp_polarities = [x/(num_selected_now+1) for x in temp_label_distribution if x != 0]
                temp_entropy = -sum([p*math.log2(p) for p in temp_polarities if p != 0])
                mg = ((polarity_sum+item[0])/(num_selected_now+1)-theta*temp_entropy) - value_old
                marginal_gains.append((mg, item[2]))
                addition_idx.append(len(marginal_gains)-1)

        if not marginal_gains:
            next_node = None
        else:
            max_mg_node = max(marginal_gains)
            if value_old + max_mg_node[0] <= max_f:
                neg_count += 1
                if not addition_idx:
                    next_node = None
                else:
                    max_add_mg = max([marginal_gains[i] for i in addition_idx], key=lambda x: x[0])
                    next_node = max_add_mg[1]
            else:
                neg_count = 0
                next_node = max_mg_node[1]

            # TODO: can do lazy update here instead of updating every time
            # if value_old >= max_f:
            #     max_f = value_old
            #     if max_mg_node[0] < 0:
            #         best_selected = [item.copy() for item in selected_neu_pos_neg]

    if return_fs:
        return fs
    selected = []
    for i in range(3):
        for item in best_selected[i].__iter__():
            selected.append(item[2])
    return selected


def pads_python(G, attr_name='pads_python', **kwargs):
    theta=kwargs.get('theta', 0.5)
    return_fs=kwargs.get('return_fs', False)
    for node in G.nodes():
        G.nodes[node]['promising_value'] = 0
        G.nodes[node]['polarity_label'] = 1 if G.nodes[node]['stance_label'] == 'pro' else -1 if G.nodes[node]['stance_label'] == 'anti' else 0
    for edge in G.edges():
        similarity = (2-abs(G.nodes[edge[0]]['polarity']-G.nodes[edge[1]]['polarity']))/2
        G.nodes[edge[0]]['promising_value'] += similarity
        G.nodes[edge[1]]['promising_value'] += similarity
    if return_fs:
        pos_fs = ecc_greedy(G.copy(), theta, True, return_fs=True)
        neg_fs = ecc_greedy(G.copy(), theta, False, return_fs=True)
        return pos_fs, neg_fs
    myg_pos = set(ecc_greedy(G.copy(), theta, True))
    myg_neg = set(ecc_greedy(G.copy(), theta, False))
    for node in G.nodes():
        G.nodes[node][attr_name] = (1 if node in myg_pos else 0) - (1 if node in myg_neg else 0)
    return list(myg_pos), list(myg_neg)


def pads_cpp(G, **kwargs):
    cpp_exe = kwargs.get('cpp_exe', 'Related_Reps\\pads_cpp\\cmake-build-release\\PADS.exe')
    dataset = kwargs.get('dataset', 'Abortion')
    input_file = kwargs.get('input_file', f'Datasets\\Static\\{dataset}\\edgelist_pads')
    theta = kwargs.get('theta', 0.5)
    max_neg = kwargs.get('max_neg', 100)

    # run the cpp program and get the output program prints on the terminal
    command = f"{cpp_exe} {input_file} {theta} {max_neg}"

    # Function to run command and process output
    def run_command_and_process(command):
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, _ = process.communicate()

        # Parse the solution time
        solution_time = None
        for line in output.split('\n'):
            if "Total Elapsed Time:" in line:
                solution_time = float(line.split(':')[1].strip().split()[0])
                break
        return solution_time

    total_time = run_command_and_process(command)
    pos_nodes, neg_nodes = pads_python(G, 'pads_cpp', **kwargs)

    return total_time, (pos_nodes, neg_nodes)