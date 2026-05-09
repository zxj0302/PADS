// g++ -O3 -march=native -std=c++17 main.cpp -I/path/to/eigen -I/path/to/spectra
#include <bits/stdc++.h>
#include <Eigen/Sparse>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/Util/SelectionRule.h>

using namespace std;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using Triplet = Eigen::Triplet<double>;

struct SignedNetwork {
    int n = 0;
    vector<Triplet> edges;
    unordered_map<int,int> id2idx;
    vector<int> idx2id;

    void add_vertex(int v) {
        if(id2idx.emplace(v,n).second){
            idx2id.push_back(v);
            ++n;
        }
    }
    
    void add_edge(int u, int v, double w) {
        edges.emplace_back(id2idx[u], id2idx[v], w);
        edges.emplace_back(id2idx[v], id2idx[u], w);
    }
    
    static SignedNetwork load(const string& fn) {
        ifstream in(fn);
        if(!in) throw runtime_error("cannot open " + fn);
        
        SignedNetwork g;
        string line;
        int u, v;
        double w;
        vector<tuple<int,int,double>> buff;
        
        while(getline(in, line)) {
            if(!(istringstream(line) >> u >> v >> w)) continue;
            g.add_vertex(u);
            g.add_vertex(v);
            buff.emplace_back(u, v, w);
        }
        
        for(auto &[a, b, wt] : buff) 
            g.add_edge(a, b, wt);
        return g;
    }
    
    SpMat to_sparse() const {
        SpMat A(n, n);
        A.setFromTriplets(edges.begin(), edges.end());
        return A;
    }
};

struct Result {
    vector<int> c1, c2, neutral;
    double polarity = numeric_limits<double>::lowest();
    double lambda1 = 0, thresh = 0, time = 0;
};

Result eigensign(const SignedNetwork& g, double tau = -1.0) {
    auto tic = chrono::high_resolution_clock::now();
    
    const SpMat A = g.to_sparse();
    if(g.n == 0) return {};
    
    // Compute dominant eigenvector
    Spectra::SparseSymMatProd<double> op(A);
    Spectra::SymEigsSolver<decltype(op)> eigs(op, 1, 6);
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestAlge);
    
    if(eigs.info() != Spectra::CompInfo::Successful) 
        throw runtime_error("eigs failed");
    
    Eigen::VectorXd v = eigs.eigenvectors(1);
    double lambda1 = eigs.eigenvalues()(0);
    
    Result best;
    best.lambda1 = lambda1;
    
    // Build adjacency lists (needed for both manual and automatic modes)
    vector<vector<pair<int,double>>> nbr(g.n);
    for(int k = 0; k < A.outerSize(); ++k) {
        for(SpMat::InnerIterator it(A, k); it; ++it) {
            nbr[k].emplace_back(it.row(), it.value());
        }
    }
    
    auto sign = [](double z) { return z > 0 ? 1 : (z < 0 ? -1 : 0); };
    
    if(tau >= 0.0) {
        // Use manual threshold τ
        best.thresh = tau;
        
        vector<int8_t> label(g.n, 0);
        long long denom = 0;
        double numer = 0;
        
        // First pass: assign labels
        for(int i = 0; i < g.n; ++i) {
            if(abs(v[i]) >= tau) {
                int s = sign(v[i]);
                if(s != 0) {
                    label[i] = s;
                    ++denom;
                }
            }
        }
        
        // Second pass: compute edge contributions
        for(int i = 0; i < g.n; ++i) {
            if(label[i] != 0) {
                for(auto [j, w] : nbr[i]) {
                    if(label[j] != 0) {
                        numer += w * label[i] * label[j];
                    }
                }
            }
        }
        
        // Avoid double counting (each edge counted twice in undirected graph)
        numer /= 2.0;
        best.polarity = (denom > 0) ? 2.0 * numer / denom : 0.0;
        
        // Build final communities
        for(int i = 0; i < g.n; ++i) {
            if(abs(v[i]) >= tau) {
                int s = sign(v[i]);
                if(s == 1) best.c1.push_back(g.idx2id[i]);
                else if(s == -1) best.c2.push_back(g.idx2id[i]);
                else best.neutral.push_back(g.idx2id[i]);
            } else {
                best.neutral.push_back(g.idx2id[i]);
            }
        }
    } else {
        // Use automatic threshold selection with preference for two communities
        vector<pair<double, int>> vec;
        vec.reserve(g.n);
        for(int i = 0; i < g.n; ++i) 
            vec.emplace_back(abs(v[i]), i);
        sort(vec.rbegin(), vec.rend()); // descending by absolute value
        
        vector<int8_t> label(g.n, 0);
        long long denom = 0;
        double numer = 0;
        
        Result best_two_communities;  // Track best solution with two communities
        bool found_two_communities = false;
        
        for(size_t pos = 0; pos < vec.size(); ) {
            double current_thr = vec[pos].first;
            
            // Activate all vertices with same threshold
            while(pos < vec.size() && vec[pos].first == current_thr) {
                int i = vec[pos].second;
                int s = sign(v[i]);
                
                if(s != 0) {
                    label[i] = s;
                    ++denom;
                    for(auto [j, w] : nbr[i]) {
                        if(label[j] != 0) {
                            numer += w * s * label[j];
                        }
                    }
                }
                ++pos;
            }
            
            if(denom > 0) {
                double pol = 2.0 * numer / denom;
                
                // Check if this solution has two non-empty communities
                bool has_pos = false, has_neg = false;
                for(int i = 0; i < g.n; ++i) {
                    if(abs(v[i]) >= current_thr) {
                        int s = sign(v[i]);
                        if(s == 1) has_pos = true;
                        else if(s == -1) has_neg = true;
                    }
                }
                
                // Update best overall solution
                if(pol > best.polarity) {
                    best.polarity = pol;
                    best.thresh = current_thr;
                }
                
                // Track best solution with two communities
                if(has_pos && has_neg && (!found_two_communities || pol > best_two_communities.polarity)) {
                    best_two_communities.polarity = pol;
                    best_two_communities.thresh = current_thr;
                    best_two_communities.lambda1 = lambda1;
                    found_two_communities = true;
                }
            }
        }
        
        // Use two-community solution if found, otherwise use best overall
        if(found_two_communities) {
            best = best_two_communities;
        }
        
        // Build final solution using selected threshold
        for(int i = 0; i < g.n; ++i) {
            if(abs(v[i]) >= best.thresh) {
                int s = sign(v[i]);
                if(s == 1) best.c1.push_back(g.idx2id[i]);
                else if(s == -1) best.c2.push_back(g.idx2id[i]);
                else best.neutral.push_back(g.idx2id[i]);
            } else {
                best.neutral.push_back(g.idx2id[i]);
            }
        }
    }
    
    best.time = chrono::duration<double>(chrono::high_resolution_clock::now() - tic).count();
    return best;
}

// Helper functions for common threshold strategies
double get_percentile_threshold(const Eigen::VectorXd& v, double percentile) {
    vector<double> abs_vals;
    for(int i = 0; i < v.size(); ++i) 
        abs_vals.push_back(abs(v[i]));
    sort(abs_vals.rbegin(), abs_vals.rend());
    
    int idx = static_cast<int>(percentile * abs_vals.size());
    return abs_vals[min(idx, static_cast<int>(abs_vals.size() - 1))];
}

double get_fraction_threshold(const Eigen::VectorXd& v, double fraction) {
    double max_val = 0;
    for(int i = 0; i < v.size(); ++i) 
        max_val = max(max_val, abs(v[i]));
    return fraction * max_val;
}

double get_normalized_threshold(const Eigen::VectorXd& v, double factor) {
    double norm1 = 0;
    for(int i = 0; i < v.size(); ++i) 
        norm1 += abs(v[i]);
    return factor / norm1;
}

int main(int argc, char* argv[]) {
    if(argc < 2 || argc > 3) {
        cerr << "Usage: " << argv[0] << " <input_file> [threshold]\n";
        cerr << "  threshold: manual threshold value (optional, default: auto-select)\n";
        cerr << "  Examples:\n";
        cerr << "    " << argv[0] << " data.txt        # automatic threshold\n";
        cerr << "    " << argv[0] << " data.txt 0.05   # manual threshold τ=0.05\n";
        return 1;
    }
    
    try {
        SignedNetwork g = SignedNetwork::load(argv[1]);
        
        double tau = -1.0;  // Default: automatic
        if(argc == 3) {
            tau = stod(argv[2]);
            if(tau < 0) {
                cerr << "Error: threshold must be non-negative\n";
                return 1;
            }
        }
        
        Result r = eigensign(g, tau);
        
        // Enhanced output with more information
        cout << fixed << setprecision(6);
        cout << "Runtime: " << r.time << " seconds\n";
        cout << "Threshold: " << r.thresh << "\n";
        cout << "Polarity: " << r.polarity << "\n";
        cout << "Lambda1: " << r.lambda1 << "\n";
        cout << "Community sizes: " << r.c1.size() << " + " << r.c2.size() 
             << " = " << (r.c1.size() + r.c2.size()) << " (+" << r.neutral.size() << " neutral)\n";
        cout << "\n";
        
        cout << "Community 1 (" << r.c1.size() << " vertices):\n";
        for(size_t i = 0; i < r.c1.size(); ++i) {
            if(i) cout << " ";
            cout << r.c1[i];
        }
        cout << "\n\n";
        
        cout << "Community 2 (" << r.c2.size() << " vertices):\n";
        for(size_t i = 0; i < r.c2.size(); ++i) {
            if(i) cout << " ";
            cout << r.c2[i];
        }
        cout << "\n\n";
        
        // if(!r.neutral.empty()) {
        //     cout << "Neutral vertices (" << r.neutral.size() << "):\n";
        //     for(size_t i = 0; i < r.neutral.size(); ++i) {
        //         if(i) cout << " ";
        //         cout << r.neutral[i];
        //     }
        //     cout << "\n";
        // }
        
    } catch(const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}