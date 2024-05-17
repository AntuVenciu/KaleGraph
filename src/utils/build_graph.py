"""
Build the graph.
"""
import numpy as np
import matplotlib.pyplot as plt


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

def build_graph(hits_id):
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
    
    print(f"Number of nodes = {np.array(hit_is_in_list).sum()}")
    print(f"Number of edges = {num_edges}")
    return edge_matrix


if __name__ == "__main__" :
    """
    Test these functions
    """
    hitIDs = [i for i in range(0, 1920 + 512) if np.random.uniform() > 0.5]
    build_graph(hitIDs)
