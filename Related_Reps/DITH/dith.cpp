/**********************************************************************
 *  dith.cpp  --  Down-in-the-Hollow (DITH) with Fibonacci heap
 *                0-based ids, A & R in first input line, twin query
 *                WEIGHTED VERSION
 *
 *  Output
 *    line-1 : runtime_seconds (6 decimal places)
 *    line-2 : community close to A  far from R      (ids separated by blank)
 *    line-3 : community close to R  far from A
 *********************************************************************/

#include <bits/stdc++.h>
#include <boost/heap/fibonacci_heap.hpp>
using namespace std;

/* ------------------------------------------------------------------ */
/*  Graph (0-based)                                                   */
struct Edge { int to; double len, w; };

struct Graph {
    int n;                                        // 0 … n-1
    vector<vector<Edge>> adj;
    explicit Graph(int N=0){ init(N); }
    void init(int N){ n=N; adj.assign(n,{}); }
    void addEdge(int u,int v,double len,double w){
        adj[u].push_back({v,len,w});
        adj[v].push_back({u,len,w});
    }
};

using DSet = vector<int>;                         // set represented as list
static const double INF = numeric_limits<double>::infinity();

/* ------------------------------------------------------------------ */
/*  Reading function (as requested)                                   */
struct InputData{
    Graph G;
    int A, R;
};

InputData readGraph(const string &fname)
{
    ifstream fin(fname);
    if(!fin) { cerr<<"cannot open "<<fname<<'\n'; exit(1); }

    int n,m, A,R;  fin>>n>>m>>A>>R;
    if(n<=0||m<0||A<0||A>=n||R<0||R>=n){
        cerr<<"Invalid header line\n"; exit(1);
    }
    Graph G(n);
    for(int i=0;i<m;++i){
        int u,v; double d,w; fin>>u>>v>>d>>w;
        if(u<0||u>=n||v<0||v>=n){
            cerr<<"Edge "<<i<<" contains invalid vertex id\n"; exit(1);
        }
        G.addEdge(u,v,d,w);
    }
    return {move(G),A,R};
}

/* ------------------------------------------------------------------ */
/*  Multi-source Dijkstra (binary heap, lazy)                         */
vector<double> multiDijkstra(const Graph &G,const DSet &src)
{
    vector<double> dist(G.n,INF);
    using QN = pair<double,int>;
    priority_queue<QN,vector<QN>,greater<QN>> pq;
    
    for(int s:src){ 
        dist[s]=0.0; 
        pq.emplace(0.0,s); 
    }
    
    while(!pq.empty()){
        auto [du,u]=pq.top(); pq.pop();
        if(du > dist[u]) continue;  // Skip outdated entries
        
        for(auto &e:G.adj[u])
            if(dist[e.to] > du + e.len){
                dist[e.to] = du + e.len;
                pq.emplace(dist[e.to], e.to);
            }
    }
    return dist;
}

/* ------------------------------------------------------------------ */
/*  Algorithm-2: vertex weights                                       */
vector<double> vertexWeights(const Graph &G,
                             const DSet &A, const DSet &R,
                             double l1,double l2)
{
    vector<double> wV(G.n,0.0);
    
    // Create sets for faster lookup
    set<int> setA(A.begin(), A.end());
    set<int> setR(R.begin(), R.end());

    // Calculate proximity to A
    auto dA = multiDijkstra(G,A);
    double Delta = 0.0;
    for(double x: dA) if(x < INF/2) Delta = max(Delta,x);
    
    for(int v=0; v<G.n; ++v) {
        if(dA[v] < INF/2) {
            wV[v] += l1 * (Delta - dA[v]);          // proximity
        }
    }

    // Calculate distance from R
    auto dR = multiDijkstra(G,R);
    for(int v=0; v<G.n; ++v) {
        if(setR.count(v)) {
            wV[v] += l2 * 0.0;  // distance is 0 if v is in R
        } else if(dR[v] < INF/2) {
            wV[v] += l2 * dR[v];                    // distance
        }
    }

    return wV;
}

/* ------------------------------------------------------------------ */
/*  Algorithm-4: peeling with Fibonacci heap                          */
struct DITHres{ double val; DSet S; };

DITHres dithSolve(const Graph &G,const vector<double>&wV,
                  double gamma,int Tmax)
{
    int n=G.n;
    
    // Calculate initial degrees (sum of edge weights)
    vector<double> deg0(n,0.0);
    for(int u=0;u<n;++u)
        for(auto &e:G.adj[u]) deg0[u]+=e.w;
    
    double totE = accumulate(deg0.begin(),deg0.end(),0.0)/2.0;
    double wSum0= accumulate(wV.begin(),wV.end(),0.0);

    if(totE==0.0) return {0.0,{}};

    double LB=(totE+wSum0)/n, UB=INF;
    DSet bestS(n); iota(bestS.begin(),bestS.end(),0);

    struct Node{ 
        int v; 
        double key;
        bool operator>(const Node&o)const{return key>o.key;}
    };
    using Fib = boost::heap::fibonacci_heap<Node,
               boost::heap::compare<greater<Node>>>;
    using Handle = Fib::handle_type;

    vector<double> ell(n,0.0);

    for(int t=1;t<=Tmax;++t){
        vector<bool> alive(n,true);
        vector<double> deg = deg0;
        double eS=totE, wSum=wSum0; 
        int k=n;

        Fib Q; 
        vector<Handle> H(n);
        for(int v=0;v<n;++v)
            H[v]=Q.push({v, ell[v]+deg[v]+wV[v]});

        while(k>0){
            if(Q.empty()) break;
            
            int v=Q.top().v; 
            Q.pop();
            if(!alive[v]) continue;

            // Check if current solution is better
            double cur=(eS+wSum)/k;
            if(cur>LB){
                LB=cur; 
                bestS.clear();
                for(int u=0;u<n;++u) 
                    if(alive[u]) bestS.push_back(u);
            }

            // Store the marginal contribution BEFORE removing vertex v
            double marginal_v = deg[v] + wV[v];
            
            // Remove vertex v
            alive[v]=false; 
            --k; 
            wSum-=wV[v];
            
            // Update neighbors using key decrease operation
            for(auto &e:G.adj[v]) {
                if(alive[e.to]){
                    deg[e.to] -= e.w;  // Update our local degree tracking
                    eS -= e.w;         // Decrease total edge weight
                    
                    // Extract current key, subtract edge weight, update
                    double currentKey = (*H[e.to]).key;
                    double newKey = currentKey - e.w;
                    Q.update(H[e.to], Node{e.to, newKey});
                }
            }
            
            // Update ell[v] with the marginal contribution
            ell[v] += marginal_v;
        }
        
        // Update upper bound according to paper's line 15
        double maxEll = 0.0;
        for(int v=0; v<n; ++v) {
            maxEll = max(maxEll, ell[v]);
        }
        UB = min(UB, maxEll / t);
        
        // Check termination condition
        if(UB > 0 && LB/UB >= 1.0-gamma) break;
    }
    return {LB,bestS};
}

/* ------------------------------------------------------------------ */
int main(int argc,char**argv)
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if(argc<2){ 
        cerr<<"usage: dith file [--lambda1 x] [--lambda2 y] "
               "[--gamma g] [--T k]\n"; 
        return 1; 
    }

    double l1=1.0,l2=1.0,gamma=0.01; 
    int Tmax=10000;
    
    for(int i=2;i<argc;++i){
        string f=argv[i];
        if(f=="--lambda1"&&i+1<argc) l1=atof(argv[++i]);
        else if(f=="--lambda2"&&i+1<argc) l2=atof(argv[++i]);
        else if(f=="--gamma"  &&i+1<argc) gamma=atof(argv[++i]);
        else if(f=="--T"      &&i+1<argc) Tmax=atoi(argv[++i]);
        else { cerr<<"unknown / incomplete flag "<<f<<'\n'; return 1;}
    }

    /* -------- read graph (file I/O not timed) -------- */
    InputData in = readGraph(argv[1]);
    const Graph &G = in.G;
    int A = in.A, R = in.R;

    /* Prepare attraction / repulsion sets */
    DSet setA{A}, setR{R};
    DSet setR_A{R}, setA_R{A};              // swapped sets

    /* ----------- run algorithm (timed) ------------- */
    using clk = chrono::steady_clock;
    auto t0 = clk::now();

    // Query 1: close to A, far from R
    auto wV1 = vertexWeights(G,setA,setR,l1,l2);
    auto res1= dithSolve(G,wV1,gamma,Tmax);

    // Query 2: close to R, far from A
    auto wV2 = vertexWeights(G,setR_A,setA_R,l1,l2);
    auto res2= dithSolve(G,wV2,gamma,Tmax);

    auto t1 = clk::now();
    
    // Convert to seconds with 6 decimal places
    auto duration = chrono::duration_cast<chrono::nanoseconds>(t1-t0).count();
    double seconds = duration / 1e9;

    /* ---------------- output ----------------------- */
    cout << fixed << setprecision(6) << seconds << '\n';
    
    // Output community close to A, far from R
    for(size_t i=0;i<res1.S.size();++i)
        cout<<res1.S[i]<<(i+1==res1.S.size()?'\n':' ');
    
    // Output community close to R, far from A
    for(size_t i=0;i<res2.S.size();++i)
        cout<<res2.S[i]<<(i+1==res2.S.size()?'\n':' ');
        
    return 0;
}