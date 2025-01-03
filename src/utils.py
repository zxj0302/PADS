import networkx as nx
import pickle
import numpy as np
import pandas as pd


# compute purity and purity
def statistics(G, dataset):
    baseline_methods = ['cascade', 'metis', 'louvain', 'eva', 'maxflow', 'flowless', 'gnn', 'myg']
    baseline_methods = [method for method in baseline_methods if method in G.nodes[0].keys()]
    dfs = []
    for sign in ['pos', 'neg']:
        attribute_and_values = []
        for attr in baseline_methods:
            if attr in ['maxflow', 'flowless', 'gnn', 'myg']:
                attribute_and_values.append((attr, 1 if sign == 'pos' else -1))
            else:
                label_count = {}
                for node in G.nodes():
                    if G.nodes[node][attr] in label_count:
                        ori = label_count[G.nodes[node][attr]]
                        label_count[G.nodes[node][attr]] = (ori[0]+1, ori[1]+G.nodes[node]['polarity'])
                    else:
                        label_count[G.nodes[node][attr]] = (1, G.nodes[node]['polarity'])
                largest_com = max(label_count.items(), key=lambda x: x[1][1] if sign == 'pos' else -x[1][1])
                attribute_and_values.append((attr, largest_com[0]))

        data= {
            'method': [av[0] for av in attribute_and_values],
            'num_nodes': [],
            'num_edges': [],
            'purity(var)': [],
            'conductance': [],
            'avg_node_polarity': [],
            'unweighted_density': [],
            'weighted_density': []
        }
        for (attribute, value) in attribute_and_values:
            attribute = attribute.split('_')[0]
            community_polarity = []
            out_edge_count = 0
            inner_edge_count = 0
            edge_polarity_sum = 0
            for node in G.nodes():
                if G.nodes[node][attribute] == value:
                    community_polarity.append(G.nodes[node]['polarity'])
                    for neighbor in G.neighbors(node):
                        if G.nodes[neighbor][attribute] != value:
                            out_edge_count += 1
                        else:
                            inner_edge_count += 1
                            edge_polarity_sum += G.edges[(node, neighbor)]['edge_polarity']
            #compute purity as variance of polarities
            purity = np.var(community_polarity)
            #compute conductance
            conductance = out_edge_count/(out_edge_count+inner_edge_count)
            avg_node_polarity = np.mean(community_polarity)
            num_nodes = len(community_polarity)
            num_edges = inner_edge_count/2
            unweighted_density = num_edges/num_nodes
            weighted_density = (edge_polarity_sum/2)/num_nodes

            #append the data to the dictionary
            data['num_nodes'].append(num_nodes)
            data['num_edges'].append(num_edges)
            data['purity(var)'].append('%.5f' % purity)
            data['conductance'].append('%.5f' % conductance)
            data['avg_node_polarity'].append('%.5f' % avg_node_polarity)
            data['unweighted_density'].append('%.5f' % unweighted_density)
            data['weighted_density'].append('%.5f' % weighted_density)

        df = pd.DataFrame(data)
        df.to_csv(f'Output/{dataset}/{sign}.csv')
        dfs.append(df)
    return dfs

def get_graph(dataset):
    if dataset == 'Brexit':
        # construct retweet-network
        with open('Datasets/Static/Brexit/retweet_edgelist.txt', 'r') as file:
            edge_list = [tuple(map(int, line.strip().split()[:2])) for line in file]
        G = nx.Graph()
        num_nodes = 7589
        G.add_nodes_from(range(0, num_nodes))
        G.add_edges_from(edge_list)
        # don't remove self-loop edges to align with the cascade-based method
        # G.remove_edges_from(nx.selfloop_edges(G))

        # result get from the cascade-based method
        with open('Datasets/Static/Brexit/comm_memberships.csv', 'r') as f:
            lines = f.readlines()
        cascade_return = [x.strip().split(',')[1] for x in lines]
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['cascade'] = int(cascade_return[i])

        # read the propagations and polarities from file
        propagations_count = [(0, 0)]*num_nodes
        propagations, polarities = pickle.load(open('Datasets/Static/Brexit/propagations_and_polarities.pkl', 'rb'))
        for i, (propagation, polarity) in enumerate(zip(propagations, polarities)):
            for node in propagation:
                propagations_count[node] = (propagations_count[node][0]+polarity, propagations_count[node][1]+1)
        node_polarities = [x[0]/x[1] if x[1] != 0 else 0 for x in propagations_count]
        for node in G.nodes():
            G.nodes[node]['stance_label'] = 'pro' if node_polarities[node] > 0.3 else 'anti' if node_polarities[node] < -0.3 else 'neutral'
            G.nodes[node]['polarity'] = node_polarities[node]
        for edge in G.edges():
            G.edges[edge]['edge_polarity'] = (G.nodes[edge[0]]['polarity']+G.nodes[edge[1]]['polarity'])/2
    elif dataset == 'Referendum_':
        #construct retweet-network
        with open('Datasets/static/Referendum_/retweet_edgelist.txt', 'r') as file:
            edge_list = [tuple(map(int, line.strip().split()[:2])) for line in file]
        G = nx.Graph()
        num_nodes = 2894
        G.add_nodes_from(range(0, num_nodes))
        G.add_edges_from(edge_list)
        #don't remove self-loop edges to align with the cascade-based method
        # G.remove_edges_from(nx.selfloop_edges(G))

        # result get from the cascade-based method
        with open('Datasets/Static/Referendum_/comm_memberships.csv', 'r') as f:
            lines = f.readlines()
        cascade_return = [x.strip().split(',')[1] for x in lines]
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['cascade'] = int(cascade_return[i])

        # read the propagations and polarities from file
        propagations_count = [(0, 0)]*num_nodes
        propagations, polarities = pickle.load(open('Datasets/Static/Referendum_/propagations_and_polarities.pkl', 'rb'))
        for i, (propagation, polarity) in enumerate(zip(propagations, polarities)):
            for node in propagation:
                propagations_count[node] = (propagations_count[node][0]+polarity, propagations_count[node][1]+1)
        node_polarities = [x[0]/x[1] if x[1] != 0 else 0 for x in propagations_count]
        for node in G.nodes():
            G.nodes[node]['stance_label'] = 'pro' if node_polarities[node] > 0.3 else 'anti' if node_polarities[node] < -0.3 else 'neutral'
            G.nodes[node]['polarity'] = node_polarities[node]
        for edge in G.edges():
            G.edges[edge]['edge_polarity'] = (G.nodes[edge[0]]['polarity']+G.nodes[edge[1]]['polarity'])/2
    elif dataset == 'Gun':
        #read in the gml file
        G = nx.read_gml('Datasets/Static/Gun/graph.gml')
        #change node label from str to int
        mapping = {node: int(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        # delete unrelated attrs
        for edge in G.edges():
            del G.edges[edge]['id']
            del G.edges[edge]['tweet_id']

        for node in G.nodes():
            G.nodes[node]['polarity'] = G.nodes[node]['ideology']
            del G.nodes[node]['ideology']
            G.nodes[node]['stance_label'] = 'pro' if G.nodes[node]['polarity'] > 0.3 else 'anti' if G.nodes[node]['polarity'] < -0.3 else 'neu'
        for edge in G.edges():
            G.edges[edge]['edge_polarity'] = (G.nodes[edge[0]]['polarity']+G.nodes[edge[1]]['polarity'])/2
    elif dataset == 'Abortion':
        #read in the gml file
        G = nx.read_gml('Datasets/Static/Abortion/graph.gml')
        #change node label from str to int
        mapping = {node: int(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        # delete unrelated attrs
        for edge in G.edges():
            del G.edges[edge]['id']
            del G.edges[edge]['tweet_id']

        for node in G.nodes():
            G.nodes[node]['polarity'] = G.nodes[node]['ideology']
            del G.nodes[node]['ideology']
            G.nodes[node]['stance_label'] = 'pro' if G.nodes[node]['polarity'] > 0.3 else 'anti' if G.nodes[node]['polarity'] < -0.3 else 'neu'
        for edge in G.edges():
            G.edges[edge]['edge_polarity'] = (G.nodes[edge[0]]['polarity']+G.nodes[edge[1]]['polarity'])/2
    elif dataset == 'Election':
        #read in the gml file
        G = nx.read_gml('Datasets/Static/Election/graph.gml')
        #change node label from str to int
        mapping = {node: int(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        for node in G.nodes():
            G.nodes[node]['polarity'] = G.nodes[node]['valence']
            del G.nodes[node]['valence']
            G.nodes[node]['stance_label'] = 'pro' if G.nodes[node]['polarity'] > 0.3 else 'anti' if G.nodes[node]['polarity'] < -0.3 else 'neu'
        for edge in G.edges():
            G.edges[edge]['edge_polarity'] = (G.nodes[edge[0]]['polarity']+G.nodes[edge[1]]['polarity'])/2
    elif dataset == 'Partisanship':
        #read in the gml file
        G = nx.read_gml('Datasets/Static/Partisanship/graph.gml')
        #change node label from str to int
        mapping = {node: int(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        for node in G.nodes():
            G.nodes[node]['polarity'] = G.nodes[node]['Partisanship']
            del G.nodes[node]['Partisanship']
            G.nodes[node]['stance_label'] = 'pro' if G.nodes[node]['polarity'] > 0.3 else 'anti' if G.nodes[node]['polarity'] < -0.3 else 'neu'
        for edge in G.edges():
            G.edges[edge]['edge_polarity'] = (G.nodes[edge[0]]['polarity']+G.nodes[edge[1]]['polarity'])/2
    else:
        raise ValueError('Invalid dataset')
    return G