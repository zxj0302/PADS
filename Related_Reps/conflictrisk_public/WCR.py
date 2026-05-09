import networkx as nx
import numpy as np
import scipy.io as sio
import os
import matlab.engine
import tempfile
import io


class WCR:
    def __init__(self, G, max_weight=2):
        self.G = G.copy()
        self.compute_weight(max_weight)
        # compute the weighted adjacency matrix
        self.A = nx.to_numpy_array(self.G, weight='weight')
        # sio.savemat(os.path.join(os.getcwd(), 'adjacency_matrix.mat'), {'A': self.A})

    def compute_weight(self, max_weight, polarity_attr='polarity'):
        # Compute the weight of each edge as the 'polarity' difference of the two endpoints
        # This is a placeholder for the actual logic to compute the weight of each edge
        for u, v in self.G.edges():
            if 'weight' not in self.G[u][v]:
                self.G[u][v]['weight'] = (max_weight-np.abs(self.G.nodes[u][polarity_attr] - self.G.nodes[v][polarity_attr]))/max_weight

    def run(self, args={}, num_cores=1, output_dir=None):
        # Create stdout and stderr streams to capture MATLAB output
        stdout_stream = io.StringIO()
        stderr_stream = io.StringIO()
        
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()
        print("MATLAB engine started")

        # Set number of cores if specified
        if num_cores is not None:
            eng.eval(f"maxNumCompThreads({num_cores});", nargout=0, stdout=stdout_stream, stderr=stderr_stream)
            print(f"Set MATLAB to use {num_cores} cores")
        
        # Add the MATLAB script directory to the path
        matlab_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matlab_original')
        eng.addpath(matlab_dir, nargout=0, stdout=stdout_stream, stderr=stderr_stream)
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Create a temporary file to store the adjacency matrix
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
            temp_file = tmp.name
        
        # Save adjacency matrix to the temporary file
        sio.savemat(temp_file, {'A': self.A})
        
        # Set up arguments
        iter_count = args.get('iter', 1)
        k = args.get('k', 2000)
        dim = args.get('dim', 50)
        min_eig = args.get('min_eig', 1e-8)
        
        try:
            print("\n----- STARTING MATLAB EXECUTION -----\n")
            # Load the adjacency matrix into MATLAB's workspace
            eng.workspace['A'] = matlab.double(self.A.tolist())
            
            # Execute the MATLAB function with the provided arguments
            try:
                print(f"Running WCROpt with parameters:")
                print(f" k={k}, dim={dim}, min_eig={min_eig}, iter={iter_count}")
                
                opt_a, acr, wcr, conflicts = eng.WCROpt(
                    matlab.double(self.A.tolist()), 
                    float(iter_count), 
                    float(k), 
                    float(dim), 
                    float(min_eig), 
                    nargout=4,
                    stdout=stdout_stream,
                    stderr=stderr_stream
                )
                
                # Print any output from the function
                print_matlab_output(stdout_stream, stderr_stream)
                
            except Exception as e:
                print(f"Error calling WCROpt directly: {e}")

            print("\n----- MATLAB EXECUTION COMPLETED -----\n")
            print(f"Results retrieved: OptA shape={np.array(opt_a).shape}, acr length={len(np.array(acr))}")
            
            # Convert MATLAB arrays to NumPy arrays
            opt_a = np.array(opt_a)
            acr = np.array(acr)
            wcr = np.array(wcr)
            conflicts = np.array(conflicts)
            
            # Save results to files if output_dir is provided
            if output_dir:
                sio.savemat(os.path.join(output_dir, 'OptA.mat'), {'OptA': opt_a})
                sio.savemat(os.path.join(output_dir, 'acr.mat'), {'acr': acr})
                sio.savemat(os.path.join(output_dir, 'wcr.mat'), {'wcr': wcr})
                sio.savemat(os.path.join(output_dir, 'conflicts.mat'), {'conflicts': conflicts})
                print(f"Results saved to directory: {output_dir}")
            
            # results = {
            #     'OptA': opt_a,
            #     'acr': acr,
            #     'wcr': wcr,
            #     'conflicts': conflicts
            # }
            results = {}
            # compare OptA with A, and get all the edges that have difference larger than 1e-4 and return the edges and the new weights
            # note that as OptA is a symmetric matrix, we only need to consider the upper triangular part of OptA
            # results are like {(u, v): new_weight}
            diff = np.abs(opt_a - self.A)
            edges = np.argwhere(diff > 1e-4)
            results = {(u, v): opt_a[u, v] for u, v in edges}
            # make sure only one of (u, v) and (v, u) is in the results
            results = {edge: results[edge] for edge in results if edge[0] < edge[1]}

        finally:
            # Clean up
            eng.quit()
            try:
                os.unlink(temp_file)
            except:
                pass
    
        return results

def print_matlab_output(stdout_stream, stderr_stream):
    """Helper function to print MATLAB output"""
    # Get the output from the buffers
    stdout_output = stdout_stream.getvalue()
    stderr_output = stderr_stream.getvalue()
    
    # Print stdout if it's not empty
    if stdout_output:
        print("MATLAB Output:")
        print(stdout_output)
    
    # Print stderr if it's not empty
    if stderr_output:
        print("MATLAB Errors/Warnings:")
        print(stderr_output)