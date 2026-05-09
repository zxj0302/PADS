#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std;

class SignedNetwork {
private:
    int n;
    Eigen::MatrixXd adjacency_matrix;
    map<int, int> vertex_map;
    vector<int> reverse_map;

public:
    SignedNetwork() : n(0) {}

    void addVertex(int vertex_id) {
        if (vertex_map.find(vertex_id) == vertex_map.end()) {
            vertex_map[vertex_id] = n;
            reverse_map.push_back(vertex_id);
            n++;
        }
    }

    void finalizeVertices() {
        adjacency_matrix = Eigen::MatrixXd::Zero(n, n);
    }

    void setEdge(int u, int v, double weight) {
        if (vertex_map.find(u) != vertex_map.end() && vertex_map.find(v) != vertex_map.end()) {
            int u_idx = vertex_map[u];
            int v_idx = vertex_map[v];
            adjacency_matrix(u_idx, v_idx) = weight;
            adjacency_matrix(v_idx, u_idx) = weight;
        }
    }

    const Eigen::MatrixXd& getAdjacencyMatrix() const {
        return adjacency_matrix;
    }

    int getNumVertices() const {
        return n;
    }

    int getOriginalVertexId(int matrix_index) const {
        if (matrix_index >= 0 && matrix_index < reverse_map.size()) {
            return reverse_map[matrix_index];
        }
        return -1;
    }

    static SignedNetwork loadFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Cannot open file: " + filename);
        }

        SignedNetwork network;
        string line;
        vector<tuple<int, int, double>> edges;

        while (getline(file, line)) {
            istringstream iss(line);
            int u, v;
            double weight;
            if (iss >> u >> v >> weight) {
                network.addVertex(u);
                network.addVertex(v);
                edges.push_back({u, v, weight});
            }
        }

        network.finalizeVertices();

        for (const auto& edge : edges) {
            int u = get<0>(edge);
            int v = get<1>(edge);
            double weight = get<2>(edge);
            network.setEdge(u, v, weight);
        }

        return network;
    }
};

struct PolarizedCommunities {
    vector<int> community1;
    vector<int> community2;
    vector<int> neutral;
    double polarity;
    double eigenvalue;
    double computation_time;
    double threshold;
};

class EigensignAlgorithm {
public:
    // Eigensign algorithm matching the Python implementation
    static PolarizedCommunities eigensign(const SignedNetwork& network) {
        auto start_time = chrono::high_resolution_clock::now();
        
        const Eigen::MatrixXd& A = network.getAdjacencyMatrix();
        int n = network.getNumVertices();

        PolarizedCommunities result;
        result.polarity = numeric_limits<double>::lowest(); // equivalent to np.finfo(float).min
        result.threshold = 0.0;

        if (n == 0) {
            result.eigenvalue = 0.0;
            result.computation_time = 0.0;
            return result;
        }

        // Get the eigenvector corresponding to the maximum eigenvalue
        // Using GeneralizedSelfAdjointEigenSolver for largest eigenvalue (equivalent to eigsh with 'LA')
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
        if (solver.info() != Eigen::Success) {
            throw runtime_error("Eigenvalue computation failed");
        }

        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        Eigen::MatrixXd eigenvectors = solver.eigenvectors();

        // Find the largest eigenvalue (eigenvalues are in ascending order)
        int max_idx = n - 1;
        double lambda1 = eigenvalues(max_idx);
        Eigen::VectorXd maximum_eigenvector = eigenvectors.col(max_idx);

        result.eigenvalue = lambda1;

        // Get thresholds from the eigenvector (discretized at third decimal digit)
        set<double> thresholds;
        for (int i = 0; i < n; ++i) {
            double abs_val = abs(maximum_eigenvector(i));
            // Discretize to 3 decimal places: int(abs_val * 1000) / 1000.0
            double threshold = floor(abs_val * 1000.0) / 1000.0;
            thresholds.insert(threshold);
        }

        // Compute x for all values of threshold
        vector<int> best_x(n, 0);
        
        for (double threshold : thresholds) {
            vector<int> x(n);
            
            // Create x: sign(element) if |element| >= threshold else 0
            for (int i = 0; i < n; ++i) {
                if (abs(maximum_eigenvector(i)) >= threshold) {
                    x[i] = (maximum_eigenvector(i) > 0) ? 1 : -1;
                } else {
                    x[i] = 0;
                }
            }

            // Evaluate objective function
            double objective_function = evaluateObjectiveFunction(A, x);
            
            // Update solution if needed
            if (objective_function > result.polarity) {
                best_x = x;
                result.polarity = objective_function;
                result.threshold = threshold;
            }
        }

        // Build the final solution from best_x
        for (int i = 0; i < n; ++i) {
            if (best_x[i] == 1) {
                result.community1.push_back(network.getOriginalVertexId(i));
            } else if (best_x[i] == -1) {
                result.community2.push_back(network.getOriginalVertexId(i));
            } else {
                result.neutral.push_back(network.getOriginalVertexId(i));
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        result.computation_time = duration.count() / 1000000.0;

        return result;
    }

private:
    // Evaluate objective function: x^T A x / x^T x
    static double evaluateObjectiveFunction(const Eigen::MatrixXd& A, const vector<int>& x) {
        int n = A.rows();
        double numerator = 0.0;
        double denominator = 0.0;

        // Calculate x^T A x
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                numerator += x[i] * A(i, j) * x[j];
            }
            denominator += x[i] * x[i];
        }

        return (denominator > 0) ? numerator / denominator : numeric_limits<double>::lowest();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    try {
        // Read the graph from file
        SignedNetwork network = SignedNetwork::loadFromFile(argv[1]);

        // Run the Eigensign algorithm (matching Python implementation)
        PolarizedCommunities result = EigensignAlgorithm::eigensign(network);

        // Output format:
        // Line 1: Time consumption (without file reading) in seconds
        cout << fixed << setprecision(6) << result.computation_time << endl;

        // Line 2: Community 1 (vertices assigned +1)
        for (size_t i = 0; i < result.community1.size(); ++i) {
            if (i > 0) cout << " ";
            cout << result.community1[i];
        }
        cout << endl;

        // Line 3: Community 2 (vertices assigned -1)
        for (size_t i = 0; i < result.community2.size(); ++i) {
            if (i > 0) cout << " ";
            cout << result.community2[i];
        }
        cout << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}