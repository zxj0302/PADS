#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <queue>


using namespace std;
using namespace boost;

// Define Node Status using enum class for better performance
enum class Status {
    Out,
    Fringe,
    In
};

// Define Node and Edge Properties
struct NodeProperty {
    double polarity = 0.0;
    unsigned polarity_label = 0;
    double promising_value = 0.0;
    Status status = Status::Out;
    double priority_key = 0.0;
    unsigned in_neighbor_count = 0;
};

// Define Edge Properties
struct EdgeProperty {
    double edge_polarity = 0.0;
};

// Define the Graph using adjacency_list with bundled properties
using Graph = adjacency_list<vecS, vecS, undirectedS, NodeProperty, EdgeProperty>;
using Vertex = graph_traits<Graph>::vertex_descriptor;
using Edge = graph_traits<Graph>::edge_descriptor;
using Traits = graph_traits<Graph>;

// Define a struct for Priority Items instead of using std::tuple
struct PriorityItem {
    double priority_key;
    double promising_value;
    Vertex vertex;

    // A timestamp or unique identifier could be added here to handle duplicates if needed

    // Comparator for priority_queue (max-heap based on priority_key, then promising_value, then vertex)
    bool operator<(const PriorityItem& other) const {
        if (priority_key != other.priority_key)
            return priority_key < other.priority_key; // higher priority_key first
        if (promising_value != other.promising_value)
            return promising_value < other.promising_value; // higher promising_value first
        return vertex > other.vertex; // lower vertex first
    }
};

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
    ifstream infile(filename, ios::binary | ios::ate);
    if (!infile.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

    // Read entire file into buffer
    streamsize size = infile.tellg();
    infile.seekg(0, ios::beg);
    vector<char> buffer(static_cast<size_t>(size));
    if (!infile.read(buffer.data(), size)) {
        throw runtime_error("Failed to read the file into buffer.");
    }
    infile.close();

    size_t pos = 0;
    // Lambda to read next token
    auto next_token = [&](string& token) -> bool {
        if (pos >= buffer.size()) return false;
        // Skip any leading whitespace
        while (pos < buffer.size() && isspace(buffer[pos])) pos++;
        if (pos >= buffer.size()) return false;
        size_t start = pos;
        while (pos < buffer.size() && !isspace(buffer[pos])) pos++;
        token.assign(&buffer[start], pos - start);
        return true;
    };

    // Read number of nodes and edges
    string token;
    if (!next_token(token)) {
        throw runtime_error("Failed to read the first token for node and edge counts.");
    }
    size_t num_nodes = stoull(token);

    if (!next_token(token)) {
        throw runtime_error("Failed to read the second token for edge count.");
    }
    size_t num_edges = stoull(token);

    // Initialize the graph with the given number of nodes
    Graph G(num_nodes);

    // Read each edge and set node properties
    for (size_t edge_count = 0; edge_count < num_edges; ++edge_count) {
        unsigned u, v, polarity_label_u, polarity_label_v;
        double polarity_u, polarity_v, edge_polarity;

        // Read u
        if (!next_token(token)) {
            throw runtime_error("Failed to read u of edge " + to_string(edge_count));
        }
        u = stoul(token);
        // Read polarity_u
        if (!next_token(token)) {
            throw runtime_error("Failed to read polarity_u of edge " + to_string(edge_count));
        }
        polarity_u = stod(token);
        // Read polarity_label_u
        if (!next_token(token)) {
            throw runtime_error("Failed to read polarity_label_u of edge " + to_string(edge_count));
        }
        polarity_label_u = stoul(token);
        // Read v
        if (!next_token(token)) {
            throw runtime_error("Failed to read v of edge " + to_string(edge_count));
        }
        v = stoul(token);
        // Read polarity_v
        if (!next_token(token)) {
            throw runtime_error("Failed to read polarity_v of edge " + to_string(edge_count));
        }
        polarity_v = stod(token);
        // Read polarity_label_v
        if (!next_token(token)) {
            throw runtime_error("Failed to read polarity_label_v of edge " + to_string(edge_count));
        }
        polarity_label_v = stoul(token);
        // Read edge_polarity
        if (!next_token(token)) {
            throw runtime_error("Failed to read edge_polarity of edge " + to_string(edge_count));
        }
        edge_polarity = stod(token);

        // Set node properties if not already set
        auto& node_u = G[u];
        if (node_u.polarity_label == 0 && node_u.promising_value == 0.0) { // Assuming label 0 means not set
            node_u.polarity = polarity_u;
            node_u.polarity_label = polarity_label_u;
            // promising_value will be computed later
        } else {
            // Optional: Verify consistency
            if (node_u.polarity != polarity_u || node_u.polarity_label != polarity_label_u) {
                throw runtime_error("Inconsistent node properties for node " + to_string(u));
            }
        }

        auto& node_v = G[v];
        if (node_v.polarity_label == 0 && node_v.promising_value == 0.0) { // Assuming label 0 means not set
            node_v.polarity = polarity_v;
            node_v.polarity_label = polarity_label_v;
            // promising_value will be computed later
        } else {
            if (node_v.polarity != polarity_v || node_v.polarity_label != polarity_label_v) {
                throw runtime_error("Inconsistent node properties for node " + to_string(v));
            }
        }

        // Add the edge with its property
        add_edge(u, v, EdgeProperty{edge_polarity}, G);
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
    const Vertex null_v = graph_traits<Graph>::null_vertex();

    // If pos is false, invert polarities and labels
    if (!pos) {
        Traits::vertex_iterator vi, vi_end;
        for (tie(vi, vi_end) = vertices(G); vi != vi_end; ++vi) {
            G[*vi].polarity = -G[*vi].polarity;
            // Ensure polarity_label remains positive; define behavior as needed
            // Here, simply inverting the value. If labels can be negative, adjust accordingly
            // Otherwise, you might map label i to (max_label - i) or similar
            G[*vi].polarity_label = static_cast<unsigned>(-static_cast<int>(G[*vi].polarity_label));
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

    if (node_promising == null_v) {
        throw runtime_error("No promising node found with polarity_label == 1.");
    }

    // Initialize variables
    double polarity_sum = 0.0;
    // Assuming polarity labels range from 0 to 2
    const unsigned num_labels = 3;
    vector<unsigned> num_selected_neu_pos_neg(num_labels, 0);

    // Define priority queues for to_select and selected sets
    // Using separate priority queues for each polarity label
    priority_queue<PriorityItem> to_select_neu_pos_neg[num_labels];
    priority_queue<PriorityItem> selected_neu_pos_neg[num_labels];

    // Set node_promising status to 'fringe' and set its priority key
    G[node_promising].status = Status::Fringe;

    // Check for self-loop
    bool has_self_loop = false;
    double self_loop_polarity = 0.0;
    Edge e;
    bool found_edge = false;

    tie(e, found_edge) = edge(node_promising, node_promising, G);
    if (found_edge) {
        has_self_loop = true;
        self_loop_polarity = G[e].edge_polarity;
    }

    G[node_promising].priority_key = has_self_loop ? self_loop_polarity : 0.0;

    // Push the node_promising into the appropriate to_select priority queue
    to_select_neu_pos_neg[G[node_promising].polarity_label].push(
        PriorityItem{G[node_promising].priority_key, G[node_promising].promising_value, node_promising}
    );

    // Initialize selection variables
    Vertex next_node = node_promising;
    double max_f = -numeric_limits<double>::infinity();
    unsigned neg_count = 0;
    // To track best_selected nodes
    vector<bool> is_best_selected(num_vertices(G), false);
    double best_f = max_f;

    // Main loop
    while (next_node != null_v && neg_count < max_neg_count) {
        Status status = G[next_node].status;
        unsigned polarity_label = G[next_node].polarity_label;

        if (status == Status::Fringe) {
            // Move from fringe to in
            G[next_node].status = Status::In;

            // Retrieve and remove the top element from to_select_neu_pos_neg
            if (to_select_neu_pos_neg[polarity_label].empty()) {
                throw runtime_error("Attempting to pop from an empty to_select_neu_pos_neg priority queue.");
            }

            PriorityItem item = to_select_neu_pos_neg[polarity_label].top();
            to_select_neu_pos_neg[polarity_label].pop();

            // To handle stale entries, verify that the popped item's priority_key matches the current node's priority_key
            if (item.priority_key != G[item.vertex].priority_key) {
                // Stale entry, skip and continue
                continue;
            }

            selected_neu_pos_neg[polarity_label].push(PriorityItem{-item.priority_key, -item.promising_value, item.vertex});
            polarity_sum += item.priority_key;
            num_selected_neu_pos_neg[polarity_label] += 1;

            // Mark as best_selected if needed
            is_best_selected[item.vertex] = true;

            // Update neighbors
            Traits::adjacency_iterator ai, ai_end;
            for (tie(ai, ai_end) = adjacent_vertices(next_node, G); ai != ai_end; ++ai) {
                Vertex neighbor = *ai;
                G[neighbor].in_neighbor_count += 1;
                unsigned neighbor_polarity_label = G[neighbor].polarity_label;

                if (G[neighbor].status == Status::Out) {
                    // Change status to 'fringe' and add to to_select_neu_pos_neg
                    G[neighbor].status = Status::Fringe;

                    // Calculate priority_key
                    double edge_polarity_total = 0.0;

                    // Find edge polarity between next_node and neighbor
                    tie(e, found_edge) = edge(next_node, neighbor, G);
                    if (found_edge) {
                        edge_polarity_total += G[e].edge_polarity;
                    }

                    // Check for self-loop of neighbor
                    bool neighbor_self_loop = false;
                    double neighbor_self_polarity = 0.0;
                    tie(e, found_edge) = edge(neighbor, neighbor, G);
                    if (found_edge) {
                        neighbor_self_loop = true;
                        neighbor_self_polarity = G[e].edge_polarity;
                    }

                    G[neighbor].priority_key = edge_polarity_total + (neighbor_self_loop ? neighbor_self_polarity : 0.0);
                    to_select_neu_pos_neg[neighbor_polarity_label].push(
                        PriorityItem{G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                    );

                } else if (G[neighbor].status == Status::Fringe) {
                    // Update priority_key by adding edge_polarity
                    double additional_edge_polarity = 0.0;
                    tie(e, found_edge) = edge(next_node, neighbor, G);
                    if (found_edge) {
                        additional_edge_polarity += G[e].edge_polarity;
                    }

                    G[neighbor].priority_key += additional_edge_polarity;

                    // Push updated priority item
                    to_select_neu_pos_neg[neighbor_polarity_label].push(
                        PriorityItem{G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                    );

                } else if (G[neighbor].status == Status::In) {
                    if (neighbor != next_node) {
                        // Update priority_key by adding edge_polarity
                        double additional_edge_polarity = 0.0;
                        tie(e, found_edge) = edge(next_node, neighbor, G);
                        if (found_edge) {
                            additional_edge_polarity -= G[e].edge_polarity;
                        }

                        G[neighbor].priority_key -= additional_edge_polarity;

                        // Push updated priority item
                        selected_neu_pos_neg[neighbor_polarity_label].push(
                            PriorityItem{G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                        );
                    }
                } else {
                    throw invalid_argument("Invalid neighbor status encountered.");
                }
            }
        }
        else if (status == Status::In) {
            // Move from in to fringe
            G[next_node].status = Status::Fringe;

            // Remove from selected_neu_pos_neg and add back to to_select_neu_pos_neg
            if (selected_neu_pos_neg[polarity_label].empty()) {
                throw runtime_error("Attempting to pop from an empty selected_neu_pos_neg priority queue.");
            }

            PriorityItem item = selected_neu_pos_neg[polarity_label].top();
            selected_neu_pos_neg[polarity_label].pop();

            // To handle stale entries, verify that the popped item's priority_key matches the current node's priority_key
            if (item.priority_key != G[item.vertex].priority_key) {
                // Stale entry, skip and continue
                continue;
            }

            to_select_neu_pos_neg[polarity_label].push(item);
            polarity_sum -= item.priority_key;
            num_selected_neu_pos_neg[polarity_label] -= 1;

            // Unmark as best_selected
            is_best_selected[item.vertex] = false;

            // Update neighbors
            Traits::adjacency_iterator ai, ai_end;
            for (tie(ai, ai_end) = adjacent_vertices(next_node, G); ai != ai_end; ++ai) {
                Vertex neighbor = *ai;
                if (G[neighbor].in_neighbor_count > 0) {
                    G[neighbor].in_neighbor_count -= 1;
                }

                unsigned neighbor_polarity_label = G[neighbor].polarity_label;

                if (G[neighbor].status == Status::Fringe) {
                    if (G[neighbor].in_neighbor_count == 0) {
                        // Node becomes 'Out'
                        G[neighbor].status = Status::Out;
                        G[neighbor].priority_key = 0.0;
                        // Note: Actual removal from priority_queue is not possible, handled by ignoring stale entries
                    }
                    else {
                        // Adjust priority_key by removing edge_polarity
                        double removed_edge_polarity = 0.0;
                        tie(e, found_edge) = edge(next_node, neighbor, G);
                        if (found_edge) {
                            removed_edge_polarity += G[e].edge_polarity;
                        }

                        G[neighbor].priority_key -= removed_edge_polarity;

                        // Push updated priority item
                        to_select_neu_pos_neg[neighbor_polarity_label].push(
                            PriorityItem{G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                        );
                    }
                }
                else if (G[neighbor].status == Status::In) {
                    // Adjust priority_key by removing edge_polarity
                    double removed_edge_polarity = 0.0;
                    tie(e, found_edge) = edge(next_node, neighbor, G);
                    if (found_edge) {
                        removed_edge_polarity += G[e].edge_polarity;
                    }

                    G[neighbor].priority_key -= removed_edge_polarity;

                    // Push updated priority item
                    selected_neu_pos_neg[neighbor_polarity_label].push(
                        PriorityItem{G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                    );
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
        polarities.reserve(num_labels);
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
        if (value_old >= best_f) {
            best_f = value_old;
            // Update best_selected
            // Reset previous best_selected
            // Only track nodes currently marked as best_selected
            // Implemented via is_best_selected vector
        }

        // Calculate marginal gains
        vector<pair<double, Vertex>> marginal_gains;
        // To keep track of nodes contributing to marginal gains
        // Iterate through the top of each to_select_neu_pos_neg priority queue and selected_neu_pos_neg
        for (unsigned label = 0; label < num_labels; ++label) {
            // To-select set: top element (highest priority)
            if (!to_select_neu_pos_neg[label].empty()) {
                PriorityItem current = to_select_neu_pos_neg[label].top();
                // Check for stale entry
                if (current.priority_key == G[current.vertex].priority_key) {
                    // Calculate potential entropy if this node is added
                    unsigned new_count = num_selected_neu_pos_neg[label] + 1;
                    double new_polarity = static_cast<double>(new_count) / (num_selected_now + 1);
                    // Assuming other labels remain the same, approximate entropy change
                    // For simplicity, use the existing entropy calculation
                    // A more accurate incremental entropy update can be implemented
                    // Here, we perform the full entropy calculation for correctness

                    // Temporary counts
                    vector<unsigned> temp_counts = num_selected_neu_pos_neg;
                    temp_counts[label] += 1;
                    unsigned temp_total = num_selected_now + 1;

                    // Calculate new polarities
                    vector<double> temp_polarities;
                    temp_polarities.reserve(num_labels);
                    for (unsigned cnt : temp_counts) {
                        if (cnt > 0)
                            temp_polarities.emplace_back(static_cast<double>(cnt) / temp_total);
                    }

                    // Calculate new entropy
                    double temp_entropy = 0.0;
                    for (double p : temp_polarities) {
                        temp_entropy -= p * log2(p);
                    }

                    double temp_polarity_sum = polarity_sum + current.priority_key;
                    double new_value = (temp_polarity_sum / temp_total) - theta * temp_entropy;
                    double mg = new_value - value_old;
                    marginal_gains.emplace_back(mg, current.vertex);
                }
            }

            // Selected set: top element (highest priority)
            if (!selected_neu_pos_neg[label].empty()) {
                PriorityItem current = selected_neu_pos_neg[label].top();
                // Check for stale entry
                if (current.priority_key == G[current.vertex].priority_key) {
                    // Calculate potential entropy if this node is removed
                    unsigned new_count = num_selected_neu_pos_neg[label] - 1;
                    double new_polarity = (num_selected_now > 1) ? static_cast<double>(new_count) / (num_selected_now - 1) : 0.0;
                    // For simplicity, perform full entropy calculation

                    // Temporary counts
                    vector<unsigned> temp_counts = num_selected_neu_pos_neg;
                    temp_counts[label] -= 1;
                    unsigned temp_total = (num_selected_now > 0) ? (num_selected_now - 1) : 0;

                    // Calculate new polarities
                    vector<double> temp_polarities;
                    temp_polarities.reserve(num_labels);
                    for (unsigned cnt : temp_counts) {
                        if (cnt > 0)
                            temp_polarities.emplace_back(static_cast<double>(cnt) / temp_total);
                    }

                    // Calculate new entropy
                    double temp_entropy = 0.0;
                    for (double p : temp_polarities) {
                        temp_entropy -= p * log2(p);
                    }

                    double temp_polarity_sum = polarity_sum - current.priority_key;
                    double new_value = (temp_polarity_sum / temp_total) - theta * temp_entropy;
                    double mg = new_value - value_old;
                    marginal_gains.emplace_back(mg, current.vertex);
                }
            }
        }

        if (!marginal_gains.empty()) {
            // Find the node with the maximum marginal gain
            auto max_mg_it = max_element(marginal_gains.begin(), marginal_gains.end(),
                                        [&](const pair<double, Vertex>& a, const pair<double, Vertex>& b) -> bool {
                                            return a.first < b.first;
                                        });
            double max_mg = max_mg_it->first;
            Vertex max_mg_node = max_mg_it->second;

            if ((value_old + max_mg) > best_f) {
                // Accept the change
                next_node = max_mg_node;
                neg_count = 0;
            }
            else {
                // Reject the change
                neg_count += 1;
                next_node = null_v;
            }
        }
        else {
            // No marginal gains to consider
            next_node = null_v;
        }
    }

    // Collect the best_selected nodes
    vector<Vertex> selected;
    selected.reserve(num_labels); // Reserve space to avoid reallocations
    for (unsigned label = 0; label < num_labels; ++label) {
        priority_queue<PriorityItem> pq = selected_neu_pos_neg[label];
        while (!pq.empty()) {
            PriorityItem item = pq.top();
            pq.pop();
            if (item.priority_key == G[item.vertex].priority_key) { // Ensure it's the latest
                selected.push_back(item.vertex);
            }
        }
    }

    return selected;
}

int main() {
    // Optimize I/O speed
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

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
        vector<Vertex> selected = ecc_greedy(G, 0.5, true, 200);

        // End timing
        auto end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;

        // Output the results
        cout << "Selected Nodes Count: " << selected.size() << " | Elapsed Time: "
             << fixed << setprecision(6) << elapsed << " seconds" << endl;

        // Optionally, output selected nodes
        /*
        cout << "Selected Nodes: ";
        for (const auto& v : selected) {
            cout << v << " ";
        }
        cout << endl;
        */

    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}