import math
from sortedcontainers import SortedSet


def ecc_greedy(G, theta, pos=True, max_neg_count=100):
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
            # best_selected = copy.deepcopy(selected_neu_pos_neg)
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

    selected = []
    for i in range(3):
        for item in best_selected[i].__iter__():
            selected.append(item[2])
    return selected


# Usage
def my_greedy(G, theta=0.5):
    for node in G.nodes():
        G.nodes[node]['promising_value'] = 0
        G.nodes[node]['polarity_label'] = 1 if G.nodes[node]['stance_label'] == 'pro' else -1 if G.nodes[node]['stance_label'] == 'anti' else 0
    for edge in G.edges():
        similarity = (2-abs(G.nodes[edge[0]]['polarity']-G.nodes[edge[1]]['polarity']))/2
        G.nodes[edge[0]]['promising_value'] += similarity
        G.nodes[edge[1]]['promising_value'] += similarity
    myg_pos = set(ecc_greedy(G.copy(), theta, True))
    myg_neg = set(ecc_greedy(G.copy(), theta, False))
    for node in G.nodes():
        G.nodes[node]['myg'] = (1 if node in myg_pos else 0) - (1 if node in myg_neg else 0)
