#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <limits>
#include <tuple>
#include <iterator>

// Define a structure for the graph node
struct Node {
    int polarity;
    int polarity_label;
    double promising_value;
    std::string status;
    double priority_key;
    int in_neighbor_count;
};

// Define a structure for the graph edge
struct Edge {
    double edge_polarity;
};

// Define a structure for the graph
struct Graph {
    std::unordered_map<int, Node> nodes;
    std::unordered_map<std::pair<int, int>, Edge, boost::hash<std::pair<int, int>>> edges;

    bool has_edge(int u, int v) {
        return edges.find({u, v}) != edges.end();
    }

    std::vector<int> neighbors(int node) {
        std::vector<int> result;
        for (const auto& edge : edges) {
            if (edge.first.first == node) {
                result.push_back(edge.first.second);
            }
        }
        return result;
    }
};

// Function to perform the ECC greedy algorithm
std::vector<int> ecc_greedy(Graph& G, double theta, bool pos = true, int max_neg_count = 200, bool return_fs = false) {
    if (!pos) {
        for (auto& node : G.nodes) {
            node.second.polarity = -node.second.polarity;
            node.second.polarity_label = -node.second.polarity_label;
        }
        for (auto& edge : G.edges) {
            edge.second.edge_polarity = -edge.second.edge_polarity;
        }
    }

    int node_promising = -1;
    double max_promising_value = -std::numeric_limits<double>::infinity();
    for (const auto& node : G.nodes) {
        if (node.second.polarity_label == 1 && node.second.promising_value > max_promising_value) {
            max_promising_value = node.second.promising_value;
            node_promising = node.first;
        }
    }

    double polarity_sum = 0;
    std::vector<int> num_selected_neu_pos_neg(3, 0);
    std::vector<std::set<std::tuple<double, double, int>>> selected_neu_pos_neg(3);
    std::vector<std::set<std::tuple<double, double, int>>> to_select_neu_pos_neg(3);

    for (auto& node : G.nodes) {
        node.second.status = "out";
        node.second.priority_key = 0;
        node.second.in_neighbor_count = 0;
    }

    G.nodes[node_promising].status = "fringe";
    G.nodes[node_promising].priority_key = G.has_edge(node_promising, node_promising) ? G.edges[{node_promising, node_promising}].edge_polarity : 0;
    to_select_neu_pos_neg[G.nodes[node_promising].polarity_label].emplace(G.nodes[node_promising].priority_key, G.nodes[node_promising].promising_value, node_promising);

    int next_node = node_promising;
    double max_f = -std::numeric_limits<double>::infinity();
    int neg_count = 0;
    std::vector<std::set<std::tuple<double, double, int>>> best_selected;
    std::vector<std::tuple<double, double, int>> fs;

    while (next_node != -1 && neg_count < max_neg_count) {
        int polarity_label = G.nodes[next_node].polarity_label;
        if (G.nodes[next_node].status == "fringe") {
            G.nodes[next_node].status = "in";
            auto item = *to_select_neu_pos_neg[polarity_label].rbegin();
            to_select_neu_pos_neg[polarity_label].erase(item);
            selected_neu_pos_neg[polarity_label].insert(item);
            polarity_sum += std::get<0>(item);
            num_selected_neu_pos_neg[polarity_label]++;

            for (int neighbor : G.neighbors(next_node)) {
                G.nodes[neighbor].in_neighbor_count++;
                int neighbor_polarity_label = G.nodes[neighbor].polarity_label;
                if (G.nodes[neighbor].status == "out") {
                    G.nodes[neighbor].status = "fringe";
                    G.nodes[neighbor].priority_key = G.edges[{next_node, neighbor}].edge_polarity + (G.has_edge(neighbor, neighbor) ? G.edges[{neighbor, neighbor}].edge_polarity : 0);
                    to_select_neu_pos_neg[neighbor_polarity_label].emplace(G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor);
                } else if (G.nodes[neighbor].status == "fringe") {
                    to_select_neu_pos_neg[neighbor_polarity_label].erase({G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor});
                    G.nodes[neighbor].priority_key += G.edges[{next_node, neighbor}].edge_polarity;
                    to_select_neu_pos_neg[neighbor_polarity_label].emplace(G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor);
                } else if (G.nodes[neighbor].status == "in") {
                    if (neighbor != next_node) {
                        selected_neu_pos_neg[neighbor_polarity_label].erase({G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor});
                        G.nodes[neighbor].priority_key += G.edges[{next_node, neighbor}].edge_polarity;
                        selected_neu_pos_neg[neighbor_polarity_label].emplace(G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor);
                    }
                } else {
                    throw std::runtime_error("Error: Invalid neighbor status of fringe nodes");
                }
            }
        } else if (G.nodes[next_node].status == "in") {
            G.nodes[next_node].status = "fringe";
            auto item = *selected_neu_pos_neg[polarity_label].begin();
            selected_neu_pos_neg[polarity_label].erase(item);
            to_select_neu_pos_neg[polarity_label].insert(item);
            polarity_sum -= std::get<0>(item);
            num_selected_neu_pos_neg[polarity_label]--;

            for (int neighbor : G.neighbors(next_node)) {
                G.nodes[neighbor].in_neighbor_count--;
                int neighbor_polarity_label = G.nodes[neighbor].polarity_label;
                if (G.nodes[neighbor].status == "fringe") {
                    if (G.nodes[neighbor].in_neighbor_count == 0) {
                        to_select_neu_pos_neg[neighbor_polarity_label].erase({G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor});
                        G.nodes[neighbor].status = "out";
                        G.nodes[neighbor].priority_key = 0;
                    } else if (neighbor != next_node) {
                        to_select_neu_pos_neg[neighbor_polarity_label].erase({G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor});
                        G.nodes[neighbor].priority_key -= G.edges[{next_node, neighbor}].edge_polarity;
                        to_select_neu_pos_neg[neighbor_polarity_label].emplace(G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor);
                    }
                } else if (G.nodes[neighbor].status == "in") {
                    selected_neu_pos_neg[neighbor_polarity_label].erase({G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor});
                    G.nodes[neighbor].priority_key -= G.edges[{next_node, neighbor}].edge_polarity;
                    selected_neu_pos_neg[neighbor_polarity_label].emplace(G.nodes[neighbor].priority_key, G.nodes[neighbor].promising_value, neighbor);
                } else {
                    throw std::runtime_error("Error: Invalid neighbor status of in nodes");
                }
            }
        } else {
            throw std::runtime_error("Error: Invalid status to add or remove");
        }

        std::vector<std::pair<double, int>> marginal_gains;
        int num_selected_now = std::accumulate(num_selected_neu_pos_neg.begin(), num_selected_neu_pos_neg.end(), 0);
        std::vector<double> polarities(3);
        for (int i = 0; i < 3; ++i) {
            polarities[i] = static_cast<double>(num_selected_neu_pos_neg[i]) / num_selected_now;
        }
        double entropy = 0;
        for (double p : polarities) {
            if (p != 0) {
                entropy -= p * std::log2(p);
            }
        }
        double value_old = polarity_sum / num_selected_now - theta * entropy;
        if (value_old >= max_f) {
            max_f = value_old;
            best_selected = selected_neu_pos_neg;
        }
        if (return_fs) {
            fs.emplace_back(value_old, polarity_sum, num_selected_now, entropy);
        }
        std::vector<int> addition_idx;
        for (int i = 0; i < 3; ++i) {
            if (!selected_neu_pos_neg[i].empty()) {
                auto item = *selected_neu_pos_neg[i].begin();
                std::vector<int> temp_label_distribution = num_selected_neu_pos_neg;
                temp_label_distribution[G.nodes[std::get<2>(item)].polarity_label]--;
                std::vector<double> temp_polarities;
                for (int x : temp_label_distribution) {
                    if (x != 0) {
                        temp_polarities.push_back(static_cast<double>(x) / (num_selected_now - 1));
                    }
                }
                double temp_entropy = 0;
                for (double p : temp_polarities) {
                    if (p != 0) {
                        temp_entropy -= p * std::log2(p);
                    }
                }
                double mg = (((polarity_sum - std::get<0>(item)) / (num_selected_now - 1)) - theta * temp_entropy) - value_old;
                marginal_gains.emplace_back(mg, std::get<2>(item));
            }
            if (!to_select_neu_pos_neg[i].empty()) {
                auto item = *to_select_neu_pos_neg[i].rbegin();
                std::vector<int> temp_label_distribution = num_selected_neu_pos_neg;
                temp_label_distribution[G.nodes[std::get<2>(item)].polarity_label]++;
                std::vector<double> temp_polarities;
                for (int x : temp_label_distribution) {
                    if (x != 0) {
                        temp_polarities.push_back(static_cast<double>(x) / (num_selected_now + 1));
                    }
                }
                double temp_entropy = 0;
                for (double p : temp_polarities) {
                    if (p != 0) {
                        temp_entropy -= p * std::log2(p);
                    }
                }
                double mg = ((polarity_sum + std::get<0>(item)) / (num_selected_now + 1) - theta * temp_entropy) - value_old;
                marginal_gains.emplace_back(mg, std::get<2>(item));
                addition_idx.push_back(marginal_gains.size() - 1);
            }
        }

        if (marginal_gains.empty()) {
            next_node = -1;
        } else {
            auto max_mg_node = *std::max_element(marginal_gains.begin(), marginal_gains.end());
            if (value_old + max_mg_node.first <= max_f) {
                neg_count++;
                if (addition_idx.empty()) {
                    next_node = -1;
                } else {
                    auto max_add_mg = *std::max_element(addition_idx.begin(), addition_idx.end(), [&](int a, int b) {
                        return marginal_gains[a].first < marginal_gains[b].first;
                    });
                    next_node = marginal_gains[max_add_mg].second;
                }
            } else {
                neg_count = 0;
                next_node = max_mg_node.second;
            }
        }
    }

    if (return_fs) {
        return fs;
    }
    std::vector<int> selected;
    for (int i = 0; i < 3; ++i) {
        for (const auto& item : best_selected[i]) {
            selected.push_back(std::get<2>(item));
        }
    }
    return selected;
}

// Function to perform the PADS algorithm in C++
void pads_cpp(Graph& G, double theta = 0.5) {
    for (auto& node : G.nodes) {
        node.second.promising_value = 0;
        node.second.polarity_label = (node.second.polarity == 1) ? 1 : (node.second.polarity == -1) ? -1 : 0;
    }
    for (const auto& edge : G.edges) {
        double similarity = (2 - std::abs(G.nodes[edge.first.first].polarity - G.nodes[edge.first.second].polarity)) / 2;
        G.nodes[edge.first.first].promising_value += similarity;
        G.nodes[edge.first.second].promising_value += similarity;
    }

    auto myg_pos = ecc_greedy(G, theta, true);
    auto myg_neg = ecc_greedy(G, theta, false);

    for (auto& node : G.nodes) {
        node.second.polarity_label = (std::find(myg_pos.begin(), myg_pos.end(), node.first) != myg_pos.end() ? 1 : 0) -
                                     (std::find(myg_neg.begin(), myg_neg.end(), node.first) != myg_neg.end() ? 1 : 0);
    }
}
