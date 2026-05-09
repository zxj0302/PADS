#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/heap/pairing_heap.hpp>
#include <boost//heap/binomial_heap.hpp>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>

using namespace std;
using namespace boost;

struct NodeProperty {
    double polarity = 0.0;
    unsigned polarity_label = 0;
    double promising_value = 0.0;
    string status = "out";
    double priority_key = 0.0;
    unsigned in_neighbor_count = 0;
};

struct EdgeProperty {
    double edge_polarity = 0.0;
};

// Define the Graph using adjacency_list with bundled properties
using Graph = adjacency_list<vecS, vecS, undirectedS, NodeProperty, EdgeProperty>;
using Vertex = graph_traits<Graph>::vertex_descriptor;
using Edge = graph_traits<Graph>::edge_descriptor;
using Traits = graph_traits<Graph>;

struct PriorityTuple {
    double priority_key;
    double promising_value;
    Vertex vertex;

    bool operator<(const PriorityTuple& other) const {
        if (priority_key != other.priority_key)
            return priority_key < other.priority_key;
        if (promising_value != other.promising_value)
            return promising_value < other.promising_value;
        return vertex > other.vertex;
    }
};

using FibHeap = heap::fibonacci_heap<PriorityTuple>;
// using FibHeap = heap::pairing_heap<PriorityTuple>;
// using FibHeap = heap::binomial_heap<PriorityTuple>;

/**
 * @brief Reads an edge list from a file and constructs a graph.
 *
 * The file format should be:
 * - First line: <num_nodes> <num_edges>
 * - Next <num_edges> lines: <u> <polarity_u> <polarity_label_u> <v> <polarity_v> <polarity_label_v> <edge_polarity>
 *
 * @param filename The path to the edge list file.
 * @return Graph The constructed graph.
 */
Graph read_edgelist(const string& filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

    string line;
    // Read the first line to get number of nodes and edges
    size_t num_nodes = 0, num_edges = 0;
    if (!getline(infile, line)) {
        throw runtime_error("Failed to read the first line for node and edge counts.");
    }
    istringstream iss_first(line);
    if (!(iss_first >> num_nodes >> num_edges)) {
        throw runtime_error("Failed to parse the number of nodes and edges.");
    }

    // Initialize the graph with the given number of nodes
    Graph G(num_nodes);
    vector<bool> node_properties_set(num_nodes, false);

    // Read the next num_edges lines to add edges and set node properties
    size_t edge_count = 0;
    while (edge_count < num_edges && getline(infile, line)) {
        istringstream iss(line);
        unsigned u, v;
        double polarity_u, polarity_v, edge_polarity;
        unsigned polarity_label_u, polarity_label_v;

        if (!(iss >> u >> polarity_u >> polarity_label_u >> v >> polarity_v >> polarity_label_v >> edge_polarity)) {
            throw runtime_error("Failed to parse edge data on line: " + line);
        }

        // Set node properties if not already set
        if (!node_properties_set[u]) {
            G[u].polarity = polarity_u;
            G[u].polarity_label = polarity_label_u;
            G[u].promising_value = 0.0; // Initialize to 0.0 or compute as needed
            node_properties_set[u] = true;
        } else {
            // Optional: Verify consistency
            if (G[u].polarity != polarity_u || G[u].polarity_label != polarity_label_u) {
                throw runtime_error("Inconsistent node properties for node " + to_string(u));
            }
        }

        if (!node_properties_set[v]) {
            G[v].polarity = polarity_v;
            G[v].polarity_label = polarity_label_v;
            G[v].promising_value = 0.0; // Initialize to 0.0 or compute as needed
            node_properties_set[v] = true;
        } else {
            if (G[v].polarity != polarity_v || G[v].polarity_label != polarity_label_v) {
                throw runtime_error("Inconsistent node properties for node " + to_string(v));
            }
        }

        // Add the edge with its property
        add_edge(u, v, EdgeProperty{edge_polarity}, G);
        edge_count++;
    }

    infile.close();

    if (edge_count != num_edges) {
        throw runtime_error("Number of edges read (" + to_string(edge_count) +
                                 ") does not match the specified count (" + to_string(num_edges) + ").");
    }

    return G;
}

/**
 * @brief Greedy algorithm to select a subset of nodes based on polarity and entropy.
 *
 * @param G Reference to the graph.
 * @param theta Threshold parameter for entropy.
 * @param pos If false, invert polarities and labels.
 * @param max_neg_count Maximum number of negative counts to prevent infinite loops.
 * @return vector<Vertex> The selected subset of vertices.
 */
vector<Vertex> ecc_greedy(Graph& G, double theta, bool pos=true, unsigned max_neg_count=100) {
    Vertex null_v = Traits::null_vertex();

    // If pos is false, invert polarities and labels
    if (!pos) {
        Traits::vertex_iterator vi, vi_end;
        for (tie(vi, vi_end) = vertices(G); vi != vi_end; ++vi) {
            G[*vi].polarity = -G[*vi].polarity;
            G[*vi].polarity_label = -G[*vi].polarity_label;
        }
        Traits::edge_iterator ei, ei_end;
        for (tie(ei, ei_end) = edges(G); ei != ei_end; ++ei) {
            G[*ei].edge_polarity = -G[*ei].edge_polarity;
        }
    }

    // Find node_promising: node with polarity_label ==1 and max promising_value
    Vertex node_promising = null_v;
    double max_promising = -numeric_limits<double>::infinity();
    Traits::vertex_iterator vi, vi_end;
    for (tie(vi, vi_end) = vertices(G); vi != vi_end; ++vi) {
        if (G[*vi].polarity_label == 1 && G[*vi].promising_value > max_promising) {
            max_promising = G[*vi].promising_value;
            node_promising = *vi;
        }
    }

    // Initialize variables
    double polarity_sum = 0.0;
    vector<unsigned> num_selected_neu_pos_neg(3, 0); // Assuming labels are 0,1,2

    // Define sorted sets with the custom comparator
    vector selected_neu_pos_neg(3, FibHeap());
    vector to_select_neu_pos_neg(3, FibHeap());

    // Set node_promising status to 'fringe' and set its priority key
    G[node_promising].status = "fringe";

    // Check for self-loop using edge with the same source and target
    bool has_self_loop = false;
    double self_loop_polarity = 0.0;
    Edge e;
    bool found_edge;

    // A list of handles
    vector handles(num_vertices(G), FibHeap::handle_type());

    tie(e, found_edge) = edge(node_promising, node_promising, G);
    if (found_edge) {
        has_self_loop = true;
        self_loop_polarity = G[e].edge_polarity;
    }

    G[node_promising].priority_key = has_self_loop ? self_loop_polarity : 0.0;
    handles[node_promising] = to_select_neu_pos_neg[G[node_promising].polarity_label].push(
        {G[node_promising].priority_key, G[node_promising].promising_value, node_promising});

    // Initialize selection variables
    Vertex next_node = node_promising;
    double max_f = -numeric_limits<double>::infinity();
    unsigned neg_count = 0;
    vector best_selected(3, FibHeap());
    unsigned step = 0;
    // Main loop
    while (next_node != null_v && neg_count < max_neg_count) {
        string status = G[next_node].status;
        unsigned polarity_label = G[next_node].polarity_label;

        if (status == "fringe") {
            // cout << "Step " << step++ << ", Adding fringe " << next_node << endl;
            // Move from fringe to in
            G[next_node].status = "in";

            auto item = to_select_neu_pos_neg[polarity_label].top();
            to_select_neu_pos_neg[polarity_label].pop();
            handles[next_node] = selected_neu_pos_neg[polarity_label].push({-item.priority_key, -item.promising_value, item.vertex});
            G[next_node].priority_key = -item.priority_key;
            polarity_sum += item.priority_key;
            num_selected_neu_pos_neg[polarity_label] += 1;

            // Update neighbors
            Traits::adjacency_iterator ai, ai_end;
            for (tie(ai, ai_end) = adjacent_vertices(next_node, G); ai != ai_end; ++ai) {
                Vertex neighbor = *ai;
                G[neighbor].in_neighbor_count += 1;
                unsigned neighbor_polarity_label = G[neighbor].polarity_label;

                if (G[neighbor].status == "out") {
                    // Change status to 'fringe' and add to to_select_neu_pos_neg
                    G[neighbor].status = "fringe";

                    // Calculate priority_key
                    double edge_polarity = 0.0;
                    bool neighbor_self_loop = false;
                    double neighbor_self_polarity = 0.0;

                    // Find edge polarity between next_node and neighbor
                    tie(e, found_edge) = edge(next_node, neighbor, G);
                    if (found_edge) {
                        edge_polarity += G[e].edge_polarity;
                    }

                    // Check for self-loop of neighbor
                    tie(e, found_edge) = edge(neighbor, neighbor, G);
                    if (found_edge) {
                        neighbor_self_loop = true;
                        neighbor_self_polarity = G[e].edge_polarity;
                    }

                    G[neighbor].priority_key = edge_polarity + (neighbor_self_loop ? neighbor_self_polarity : 0.0);
                    handles[neighbor] = to_select_neu_pos_neg[neighbor_polarity_label].push({G[neighbor].priority_key, G[neighbor].promising_value, neighbor});

                } else if (G[neighbor].status == "fringe") {
                    // Calculate additional edge polarities from next_node to neighbor
                    double edge_polarity = 0.0;
                    tie(e, found_edge) = edge(next_node, neighbor, G);
                    if (found_edge) {
                        edge_polarity += G[e].edge_polarity;
                    }
                    // Update priority_key
                    G[neighbor].priority_key += edge_polarity;

                    // Insert updated tuple
                    to_select_neu_pos_neg[G[neighbor].polarity_label].update(handles[neighbor], {G[neighbor].priority_key, G[neighbor].promising_value, neighbor});

                } else if (G[neighbor].status == "in") {
                    if (neighbor != next_node) {
                        // Calculate additional edge polarities
                        double edge_polarity = 0.0;
                        tie(e, found_edge) = edge(next_node, neighbor, G);
                        if (found_edge) {
                            edge_polarity += G[e].edge_polarity;
                        }
                        // Update priority_key
                        G[neighbor].priority_key -= edge_polarity;

                        // Insert updated tuple
                        selected_neu_pos_neg[G[neighbor].polarity_label].update(handles[neighbor], {G[neighbor].priority_key, G[neighbor].promising_value, neighbor});
                    }
                } else {
                    throw invalid_argument("Invalid neighbor status encountered.");
                }
            }
        }
        else if (status == "in") {
            // cout << "Step " << step++ << ", Removing in " << next_node << endl;
            // Move from in to fringe
            G[next_node].status = "fringe";

            auto item = selected_neu_pos_neg[polarity_label].top();
            selected_neu_pos_neg[polarity_label].pop();
            handles[next_node] = to_select_neu_pos_neg[polarity_label].push({-item.priority_key, -item.promising_value, item.vertex});
            G[next_node].priority_key = -item.priority_key;
            polarity_sum += item.priority_key;
            num_selected_neu_pos_neg[polarity_label] -= 1;

            // Update neighbors
            Traits::adjacency_iterator ai, ai_end;
            for (tie(ai, ai_end) = adjacent_vertices(next_node, G); ai != ai_end; ++ai) {
                Vertex neighbor = *ai;
                G[neighbor].in_neighbor_count -= 1;
                unsigned neighbor_polarity_label = G[neighbor].polarity_label;

                if (G[neighbor].status == "fringe") {
                    if (G[neighbor].in_neighbor_count == 0) {
                        G[neighbor].status = "out";
                        G[neighbor].priority_key = 0.0;
                    }
                    else if (neighbor != next_node) {
                        // Calculate edge_polarity
                        double edge_polarity = 0.0;
                        tie(e, found_edge) = edge(next_node, neighbor, G);
                        if (found_edge) {
                            edge_polarity += G[e].edge_polarity;
                        }
                        // Update priority_key
                        G[neighbor].priority_key -= edge_polarity;

                        // Insert updated tuple
                        to_select_neu_pos_neg[G[neighbor].polarity_label].update(handles[neighbor], {G[neighbor].priority_key, G[neighbor].promising_value, neighbor});
                    }
                }
                else if (G[neighbor].status == "in") {
                    // Calculate edge_polarity
                    double edge_polarity = 0.0;
                    tie(e, found_edge) = edge(next_node, neighbor, G);
                    if (found_edge) {
                        edge_polarity += G[e].edge_polarity;
                    }
                    // Update priority_key
                    G[neighbor].priority_key += edge_polarity;

                    // Insert updated tuple
                    selected_neu_pos_neg[G[neighbor].polarity_label].update(handles[neighbor], {G[neighbor].priority_key, G[neighbor].promising_value, neighbor});
                }
                else {
                    throw invalid_argument("Invalid neighbor status encountered.");
                }
            }
        }
        else {
            throw invalid_argument("Invalid node status encountered.");
        }

        // Calculate the current objective function value
        unsigned num_selected_now = 0;
        for (unsigned count : num_selected_neu_pos_neg) {
            num_selected_now += count;
        }

        // Calculate polarities
        vector<double> polarities;
        polarities.reserve(3);
        for (unsigned count : num_selected_neu_pos_neg) {
            if (count > 0)
                polarities.emplace_back(static_cast<double>(count) / num_selected_now);
        }

        // Calculate entropy
        double entropy = 0.0;
        for (double p : polarities) {
            entropy -= p * log2(p);
        }

        double value_old = (num_selected_now > 0) ? (polarity_sum / num_selected_now - theta * entropy) : 0.0;

        // Update max_f and best_selected
        if (value_old >= max_f) {
            max_f = value_old;
            // Deep copy selected_neu_pos_neg to best_selected
            for (unsigned i = 0; i < 3; ++i) {
                best_selected[i] = selected_neu_pos_neg[i];
            }
        }

        // Calculate marginal gains
        vector<pair<double, Vertex>> marginal_gains;
        vector<unsigned> addition_idx;
        for (unsigned i = 0; i < 3; ++i) {
            // Selected set
            // for (auto item : selected_neu_pos_neg[i]) {
            //     cout << "(" << item.priority_key << ", " << item.vertex << ")" << endl;
            // }
            if (!selected_neu_pos_neg[i].empty()) {
                auto selected_item = selected_neu_pos_neg[i].top();
                // cout << "Top node: " << selected_item.vertex << endl;
                while (selected_item.priority_key != G[selected_item.vertex].priority_key || G[selected_item.vertex].status != "in") {
                    selected_neu_pos_neg[i].pop();
                    selected_item = selected_neu_pos_neg[i].top();
                }
                // Calculate temporary distributions
                vector<unsigned> temp_label_distribution = num_selected_neu_pos_neg;
                temp_label_distribution[i] -= 1;
                // Calculate polarities
                vector<double> temp_polarities;
                for (unsigned cnt : temp_label_distribution) {
                    if (cnt > 0)
                        temp_polarities.push_back(static_cast<double>(cnt) / (num_selected_now - 1));
                }
                // Calculate temp entropy
                double temp_entropy = 0.0;
                for (double p : temp_polarities) {
                    temp_entropy -= p * log2(p);
                }
                // Calculate marginal gain
                double polarity_sum_new = (num_selected_now > 1) ? (polarity_sum + selected_item.priority_key) / (num_selected_now - 1) : 0.0;
                double mg = (polarity_sum_new - theta * temp_entropy) - value_old;
                marginal_gains.emplace_back(mg, selected_item.vertex);
            }

            // To-select set
            // for (auto item : to_select_neu_pos_neg[i]) {
            //     cout << "(" << item.priority_key << ", " << item.vertex << ")" << endl;
            // }
            if (!to_select_neu_pos_neg[i].empty()) {
                auto to_select_item = to_select_neu_pos_neg[i].top();
                // cout << "Top node: " << to_select_item.vertex << endl;
                while (to_select_item.priority_key != G[to_select_item.vertex].priority_key || G[to_select_item.vertex].status != "fringe") {
                    to_select_neu_pos_neg[i].pop();
                    to_select_item = to_select_neu_pos_neg[i].top();
                }
                // Calculate temporary distributions
                vector<unsigned> temp_label_distribution = num_selected_neu_pos_neg;
                temp_label_distribution[i] += 1;
                // Calculate polarities
                vector<double> temp_polarities;
                for (unsigned cnt : temp_label_distribution) {
                    if (cnt > 0)
                        temp_polarities.push_back(static_cast<double>(cnt) / (num_selected_now + 1));
                }
                // Calculate temp entropy
                double temp_entropy = 0.0;
                for (double p : temp_polarities) {
                    temp_entropy -= p * log2(p);
                }
                // Calculate marginal gain
                double polarity_sum_new = (polarity_sum + to_select_item.priority_key) / (num_selected_now + 1);
                double mg = (polarity_sum_new - theta * temp_entropy) - value_old;
                marginal_gains.emplace_back(mg, to_select_item.vertex);
                addition_idx.emplace_back(marginal_gains.size() - 1);
            }
        }

        if (marginal_gains.empty()) {
            next_node = Graph::null_vertex();
        }
        else {
            // Find the node with the maximum marginal gain
            auto max_mg_it = max_element(marginal_gains.begin(), marginal_gains.end(),
                                        [&](const pair<double, Vertex>& a, const pair<double, Vertex>& b) -> bool {
                                            return a.first < b.first;
                                        });
            double max_mg = max_mg_it->first;
            // cout << max_mg << endl;
            Vertex max_mg_node = max_mg_it->second;

            if (value_old + max_mg <= max_f) {
                neg_count += 1;
                if (addition_idx.empty()) {
                    next_node = Graph::null_vertex();
                }
                else {
                    // Find the max addition marginal gain
                    double max_add_mg = -numeric_limits<double>::infinity();
                    Vertex candidate_node = Graph::null_vertex();
                    for (unsigned idx : addition_idx) {
                        if (marginal_gains[idx].first > max_add_mg) {
                            max_add_mg = marginal_gains[idx].first;
                            candidate_node = marginal_gains[idx].second;
                        }
                    }
                    next_node = candidate_node;
                }
            }
            else {
                neg_count = 0;
                next_node = max_mg_node;
            }
        }
    }

    // Prepare the result
    set<Vertex> selected;
    for (const auto& best_s : best_selected) {
        for (const auto& item : best_s) {
            selected.insert(item.vertex);
        }
    }
    // cout << max_f << endl;
    return vector<Vertex>(selected.begin(), selected.end());
}

int main() {
    string filename = "../edgelist_pads"; // Replace with your actual file path

    try {
        // Read the graph from the edge list file
        Graph G = read_edgelist(filename);

        // Compute promising_value for each node by iterating over all edges once
        typedef graph_traits<Graph> Traits;
        Traits::edge_iterator ei, ei_end;
        for (tie(ei, ei_end) = edges(G); ei != ei_end; ++ei) {
            Vertex source = boost::source(*ei, G);
            Vertex target = boost::target(*ei, G);
            double sim = (2.0 - abs(G[source].polarity - G[target].polarity)) / 2.0;
            G[source].promising_value += sim;
            G[target].promising_value += sim;
        }

        // Start timing
        auto start = chrono::high_resolution_clock::now();

        // Execute the eccentricity greedy algorithm
        vector<Vertex> selected = ecc_greedy(G, 0.5, true, 100);

        // End timing
        auto end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;

        // Output the results
        cout << "Selected Nodes Count: " << selected.size() << " | Elapsed Time: "
                  << elapsed << " seconds" << endl;

    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
