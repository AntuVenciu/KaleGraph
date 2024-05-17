"""
Functions to build the graph.
Hits have the following scheme for features:
0: ID
1: time
2: layer
3: x
4: y
5: ztimediff
6: wire phi
7: wire theta
"""
import numpy as np
import matplotlib.pyplot as plt


def min_dist(hits, i, j):
    """
    Return minimum distance between wires i and j
    """
    # Properties of the hits
    #layer_i = hits[i][2]
    #layer_j = hits[j][2]
    x_i = hits[i][3]
    x_j = hits[j][3]
    y_i = hits[i][4]
    y_j = hits[j][4]
    z_i = hits[i][5]
    z_j = hits[j][5]
    #phi_i = hits[i][6]
    #phi_j = hits[j][6]
    #theta_i = hits[i][7]
    #theta_j = hits[j][7]
    
    return np.sqrt(x_i * x_j + y_i * y_j + z_i * z_j)

def calculate_edge_features(edge_matrix, hits, prob):
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
        # Get nodes
        i = e[0]
        j = e[1]
        
        # edge e features vector
        e_features = []
        
        # Calculate time diff
        e_features.append(hits[i][1] - hits[j][1])
        # Calculate minimum distance
        e_features.append(min_dist(hits, i, j))
        # Get data driven weight (probability) of each connection
        e_features.append(prob[i][j])
        
        edges_features.append(e_features)
    
    return edges_features
    
    

def build_adjacency_matrix(file_name="edgeMatrix.txt",
                           f_cdch=0.015,
                           f_spx=0.005):
    """
    Create the data driven adjacency matrix for the graph
    from a data driven file of connection occurrence.
    Apply a cut for the edge connection selection contributing less
    than f.
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
        cut_cdch = norm_cdch * f_cdch
        norm_spx = np.array(occurrence_matrix[i][1920:]).sum()
        cut_spx = norm_spx * f_spx

        for j in range(NUM_TOT_NODES):
            # Select cut based on CDCH or TC connections
            cut = cut_cdch
            if j > 1920 or i > 1920:
                cut = cut_spx
            
            if occurrence_matrix[i][j] > cut:
                adj_matrix[i][j] += 1
                num_edges += 1

    #print(f"Number of nodes = {NUM_TOT_NODES}")
    #print(f"Number of edges = {number_of_edges}")

    return adj_matrix

def build_edge_matrix(hits_id):
    """
    Build a graph from a list of hits ID.
    """
    #Initialize the edge matrix: shape = 2 x num_edges
    edge_matrix = []
    num_edges = 0

    # Calculate the adjacency matrix
    adj_matrix = build_adjacency_matrix()
    NUM_TOT_NODES = adj_matrix.shape[0]
    
    # Create a vector of bools with 1 if hit is in hit_id
    # and 0 if it is not
    hit_is_in_list = [1 if x in hits_id else 0 for x in range(0, NUM_TOT_NODES)]

    # Loop over hits ID to build the graph
    for i in hits_id:
        for j in range(0, i): # Just one directional graph
            if adj_matrix[i][j] == 1:
                # Check that id == j is in hits_id
                if hit_is_in_list[int(j)]:
                    # Build the edge between node i and j
                    edge_matrix.append([i, j])
                    num_edges += 1
    
    edge_matrix = np.array(edge_matrix)

    # Print outs
    print(f"Number of nodes = {np.array(hit_is_in_list).sum()}")
    print(f"Number of edges = {num_edges}")
    print(f"edge_matrix with shape = ({edge_matrix.shape[0]} x {edge_matrix.shape[1]})")
    #print(f"EDGE MATRIX =\n", edge_matrix)

    return edge_matrix

def build_graph(hits):
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
    # X is simple
    X = hits

    # Build the adjacency matrix
    hits_id = hits[0] # Param 0 of hits is the hit ID
    edge_matrix = build_edge_matrix(hits_id)

    # Build the edge_feature vector
    R = calculate_edge_features(edge_matrix, hits)
    
    # Return the graph
    return [X, edge_matrix.T, R]


if __name__ == "__main__" :
    """
    Test these functions
    """
    hitIDs = [i for i in range(0, 1920 + 512) if np.random.uniform() > 0.5]
    build_edge_matrix(hitIDs)
