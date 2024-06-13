"""
Functions to build the graph.
Hits have the following scheme for features:
0: ID
1: time
2: layer
3: x
4: y
5: z
6: wire phi
7: wire theta
"""
import numpy as np
import torch


def load_data(filename):
    """
    Load hits data from a file
    and transform them into a pandas dataframe.
    Separate events looking at hit ID
    """
    hitID, wireID, time, layer, ch0, ch1, ampl0, ampl1, x, y, z, truth, trackID, mom, trackPhi, trackTheta = np.loadtxt(filename, unpack=True)
    events_separators = [i for i, k in enumerate(hitID) if k==0]
    #print(f"Events starting indexes = {events_separators}")
    #print(f"Number of events = {len(events_separators)}")
    feature_names = ['wireID', 't', 'layer', 'charge0', 'charge1', 'ampl0', 'ampl1', 'x', 'y', 'z', 'truth', 'trackID', 'mom', 'trackPhi', 'trackTheta']
    events = [{'wireID' : wireID[events_separators[i] : events_separators[i + 1]],
               't' : time[events_separators[i] : events_separators[i + 1]] * 1e9,
               'layer' : layer[events_separators[i] : events_separators[i + 1]],
               'charge0' : ch0[events_separators[i] : events_separators[i + 1]],
               'charge1' : ch1[events_separators[i] : events_separators[i + 1]],
               'ampl0' : ampl0[events_separators[i] : events_separators[i + 1]],
               'ampl1' : ampl1[events_separators[i] : events_separators[i + 1]],
               'x' : x[events_separators[i] : events_separators[i + 1]],
               'y' : y[events_separators[i] : events_separators[i + 1]],
               'z' : z[events_separators[i] : events_separators[i + 1]],
               'truth' : truth[events_separators[i] : events_separators[i + 1]],
               'trackID' : trackID[events_separators[i] : events_separators[i + 1]],
               'mom' : mom[events_separators[i] : events_separators[i + 1]],
               'trackPhi' : trackPhi[events_separators[i] : events_separators[i + 1]],
               'trackTheta' : trackTheta[events_separators[i] : events_separators[i + 1]]}
              for i in range(0, len(events_separators) - 1)]

    return events


def filter_hits(hits, ampl_cut=0.025, charge_cut=5e-10):
    """
    Apply simple cuts to filter noise hits
    """
    hits_features = list(hits.keys())
    filter = (hits['ampl0'] >= ampl_cut | hits['ampl1'] >= ampl_cut) & (hits['charge0'] >= charge_cut | hits['charge1'] >= charge_cut) 
    filtered_hits = {}

    # Fill the new dictionary
    for key in hits_features:
        filtered_hits.update({key : hits[key][filter]})

    return filtered_hits
        
def quantile(sorted_array, f):
    """
    Evaluate value of array at f-th quantile of the sorted array
    
    sum = 0
    for x in sorted_array:
        sum += x
        if sum > f:
            return x
    """
    sorted_array = np.asarray(sorted_array)
    cumulative_sum = np.cumsum(sorted_array)
    
    # Find the index where cumulative sum exceeds f
    index = np.searchsorted(cumulative_sum, f, side='right')

    return sorted_array[index]

def min_dist(hits, i, j):
    """
    Return minimum distance between hits i and j
    """
    # Properties of the hits
    #layer_i = hits[i][2]
    #layer_j = hits[j][2]
    x_ij = hits['x'][i] - hits['x'][j]
    y_ij = hits['y'][i] - hits['y'][j]
    z_ij = hits['z'][i] - hits['z'][j]
    #phi_i = hits[i][6]
    #phi_j = hits[j][6]
    #theta_i = hits[i][7]
    #theta_j = hits[j][7]
    
    return np.sqrt(x_ij * x_ij + y_ij * y_ij + z_ij * z_ij)

def calculate_edge_features(hits, edge_matrix, adj_matrix):
    """
    Evaluate edge features.
    For each edge, the following features are evaluated:
    1. time difference between hits;
    2. minimum distance between hits;
    3. data driven edge probability.
    Return:
    edges_features = matrix of shape = (num_edges, num_edge_features)
    """
    # Initialize edge_features matrix
    edges_features = []
    
    # Loop over edges
    for e in edge_matrix:
        # Get nodes id
        i = e[0]
        j = e[1]
        
        # get hits' IDs
        hit_id_i = int(i)
        hit_id_j = int(j)
        """
        for idx, wire in enumerate(hits['wireID']):
            if wire == i:
                hit_id_i = idx
            if wire == j:
                hit_id_j = idx
        """
        # edge e features vector
        e_features = []
        
        # Calculate time diff
        e_features.append(hits['t'][hit_id_i] - hits['t'][hit_id_j])
        # Calculate minimum distance
        e_features.append(min_dist(hits, hit_id_i, hit_id_j)) # Min dist
        e_features.append(hits['z'][hit_id_i] - hits['z'][hit_id_j]) # Delta z
        e_features.append(np.arctan2(hits['y'][hit_id_i], hits['x'][hit_id_i]) - np.arctan2(hits['y'][hit_id_j], hits['x'][hit_id_j])) # Delta Phi
        # Get data driven weight (probability) of each connection
        # from the adjacency matrix
        e_features.append(adj_matrix[int(hits['wireID'][hit_id_i])][int(hits['wireID'][hit_id_j])])#e_features.append(adj_matrix[i][j])
        
        edges_features.append(e_features)
    
    return np.array(edges_features)

def build_adjacency_matrix(file_name="utils/edgeMatrix.txt",
                           f_cdch=0.015,
                           f_spx=0.005):
    """
    Create the data driven adjacency matrix for the graph
    from a data driven file of connection occurrence.
    The adjacency matrix has shape = (n_nodes_tot x n_nodes_tot)
    and is filled with p>0 (connected, p = probability of the connection) and 0 (not connected).
    Apply a cut for the edge connection selection contributing less
    than f.
    Return:
    adjacency_matrix
    """
    # Load file of occurrence
    occurrence_matrix = np.loadtxt(file_name)

    # print out variable
    num_edges = 0

    # Initialize adjacency matrix with dim = n x n
    NUM_TOT_NODES = occurrence_matrix.shape[0]
    adj_matrix = np.zeros(shape=(NUM_TOT_NODES, NUM_TOT_NODES))

    # Fill the matrix
    for i in range(NUM_TOT_NODES):
        # Evaluate cut to discard connections
        norm_cdch = np.array(occurrence_matrix[i][:1920]).sum()
        norm_spx = np.array(occurrence_matrix[i][1920:]).sum()
        
        # Evaluate the (1 - f) percentile of the occupancy matrix as threshold
        # for node connections
        if norm_cdch > 0:
            cut_cdch = norm_cdch * quantile(np.sort(np.array(occurrence_matrix[i][:1920]) / norm_cdch), f_cdch)
        else:
            cut_cdch = 1e30
        if norm_spx > 0:
            cut_spx = norm_spx * quantile(np.sort(np.array(occurrence_matrix[i][1920:]) / norm_spx), f_spx)
        else:
            cut_spx = 1e30
        

        for j in range(NUM_TOT_NODES):
            # Select cut based on CDCH or TC connections
            norm = norm_cdch
            cut = cut_cdch
            if j > 1920 or i > 1920:
                norm = norm_spx
                cut = cut_spx

            if occurrence_matrix[i][j] > cut and norm > 0:
                adj_matrix[i][j] = occurrence_matrix[i][j] / norm
                num_edges += 1
                
    print(f"Number of Total Nodes = {NUM_TOT_NODES}")
    print(f"Number of Total Edges = {num_edges}")

    return adj_matrix

def build_edge_matrix(hits_id, adj_matrix):
    """
    Build a graph from a list of hits ID, provided the adjacency matrix.
    """

    # Initialize variables
    edge_matrix = []
    num_edges = 0
    NUM_TOT_NODES = adj_matrix.shape[0]

    # Create a vector of bools with 1 if hit is in hits_id and 0 if it is not
    hits_id_set = set(hits_id)
    hit_is_in_list = np.isin(np.arange(NUM_TOT_NODES + 1), hits_id)

    # Convert hits_id to a NumPy array for efficient indexing
    hits_id_np = np.array(hits_id)

    # Loop over hits ID to build the graph
    for i, wire_i in enumerate(hits_id):
        # Get the indices where adj_matrix[wire_i] > 0
        neighbors = np.where(adj_matrix[wire_i, :wire_i] > 0)[0]

        # Filter out neighbors that are not in hits_id
        valid_neighbors = neighbors[hit_is_in_list[neighbors]]

        # Get the corresponding indices in hits_id
        indices = np.nonzero(np.isin(hits_id_np, valid_neighbors))[0]

        # Build the edges
        edges = np.column_stack((np.full(len(indices), i), indices))
        edge_matrix.append(edges)
        num_edges += len(edges)

    edge_matrix = np.vstack(edge_matrix)
    #return edge_matrix
    
    
    #Initialize the edge matrix: shape = 2 x num_edges
    #edge_matrix_1 = []
    #num_edges = 0

    #NUM_TOT_NODES = adj_matrix.shape[0]

    # Create a vector of bools with 1 if hit is in hit_id
    # and 0 if it is not
    #hit_is_in_list_1 = [1 if x in hits_id else 0 for x in range(NUM_TOT_NODES + 1)]

    # Loop over hits ID to build the graph
    """
    for i, wire_i in enumerate(hits_id):
        for wire_j in range(wire_i): # Just one directional graph
            if adj_matrix[wire_i][wire_j] > 0:
                # Check that id == wire_j is in hits_id
                if hit_is_in_list[wire_j]:
                    j = np.argwhere(hits_id == wire_j)
                    # Build the edge between node i and j
                    for k in j:
                        edge_matrix_1.append([i, k[0]])
                        #num_edges += 1
    
    edge_matrix_1 = np.array(edge_matrix_1)
    print(f"edge_matrix - edge_matrix_1 : {edge_matrix - edge_matrix_1}")
    """
    # Print outs
    #print(f"Number of nodes = {np.array(hit_is_in_list).sum()}")
    #print(f"Number of edges = {num_edges}")
    #print(f"edge_matrix with shape = ({edge_matrix.shape[0]} x {edge_matrix.shape[1]})")
    #print(f"EDGE MATRIX =\n", edge_matrix)
    
    return edge_matrix

def build_graph(hits, adj_matrix):
    """
    Build the graph.
    Input:
    param hits: matrix X of shape (n_hits, num_node_features)
    Output:
    graph: list.
    At index 0: matrix of hits features X
    At index 1: matrix of edge features R of shape = (num_edges, num_edge_features)
    At index 2: edge matrix
    """

    # Filter hits first
    hits = filter_hits(hits)
    
    # X is simple
    X = np.array(#[hits['wireID'],
                 [hits['t'],
                 #hits['layer'],
                 hits['x'],
                 hits['y'],
                 hits['z']])
    
    # Build the adjacency matrix
    hits_id = hits['wireID'].astype('int') # Param 0 of hits is the hit ID
    #print("Hits ID = ", hits_id)
    edge_matrix = build_edge_matrix(hits_id, adj_matrix)
    #print(edge_matrix)
    # Build the edge_feature vector
    R = calculate_edge_features(hits, edge_matrix, adj_matrix)
    
    # Return the graph
    #print(f"X shape = {(X.T).shape}\nedge_index shape = {edge_matrix.T.shape}\nedge_attr shape = {R.shape}", )
    return {'x' : X.T, 'edge_index' : edge_matrix.T, 'edge_attr' : R}


if __name__ == "__main__" :

    PLOT = False
    TIME = True

    if (TIME):
        """
        Time these functions
        """
        import time
    
        
        t_start = time.time()
        adj_matrix = build_adjacency_matrix()
        t_stop = time.time()
        print(f"{t_stop - t_start :.3f} s to initialize the adjacency matrix")
    
        deltaT = []
        for i in range(0, 100):
            hitIDs = [i for i in range(0, 1920 + 512) if np.random.uniform() > 0.9]
            t_0 = time.time()
            my_graph = build_graph(hitIDs, adj_matrix)
            t_1 = time.time()
            deltaT.append(t_1 - t_0)
        T = np.array(deltaT)
        print(f"Average time to make a graph: {np.mean(T):.3f} s +/- {np.std(T):.3f} s")

    if (PLOT):
        """
        Plot a graph
        """
        from plot_graph import plot
        
        
        adj_matrix = build_adjacency_matrix()
        hits = [i for i in range(0, 1920 + 512) if np.random.uniform() > 0.9]
        my_graph = build_graph(hits, adj_matrix)
        plot(my_graph['edge_index'].T)
