"""
Plotting the 2D graph of hits based on the 'edgeMatrix.txt' file
"""
import numpy as np
import matplotlib.pyplot as plt

def calculate_coordinates(node_id):
    # Evaluate x-y coordinate of a wire at CDCH center:
    # input:
    # node_id = wire ID
    layer = int((node_id - 192) / 192)
    wireID_in_layer = node_id % 192
    radius = 270 - layer * 10;
    angle = wireID_in_layer * 360 / 192 * (np.pi / 180);
    x = radius * np.cos(-(angle) + 30. * np.pi / 180.)
    y = radius * np.sin(-(angle) + 30. * np.pi / 180.)
    return x, y

def read_adjacency_matrix(file_path):
    # Read adjacency matrix from file
    adjacency_matrix = np.loadtxt(file_path)
    #print("data shape = ", adjacency_matrix)
    return adjacency_matrix

def plot_graph(adjacency_matrix):
    # Get number of nodes
    num_nodes = adjacency_matrix.shape[0]
    is_node_plotted = np.zeros(num_nodes, dtype='int32')
    print(f"Number of possible nodes = {num_nodes}")

    # Plot nodes and edges 
    for node_id in range(num_nodes):
        # Put a cut on connections contributing less then 0.1% in the number of connections
        # of a node
        norm = np.array(adjacency_matrix[node_id]).sum()
        cut = norm * 0.025
        for j in range(num_nodes):
            # Plot only connected edges
            if adjacency_matrix[node_id][j] > 0:

                # Plot nodes
                if not is_node_plotted[node_id]:
                    node_x, node_y = calculate_coordinates(node_id)
                    plt.errorbar(node_x, node_y, fmt='o', alpha=.5, markersize=10, color='blue')
                    is_node_plotted[node_id] = 1
                if not is_node_plotted[j]:
                    node_x, node_y = calculate_coordinates(j)
                    plt.errorbar(node_x, node_y, fmt='o', alpha=.5, markersize=10, color='blue')
                    is_node_plotted[j] = 1

                # Plot edges
                if adjacency_matrix[node_id][j] > cut:
                    xi, yi = calculate_coordinates(node_id)
                    xj, yj = calculate_coordinates(j)
                    plt.plot([xi, xj], [yi, yj], 'k-', linewidth=0.5, alpha=.5)

    plt.axis('off')
    plt.axis('equal')
    plt.title('Graph Nodes and Edges')
    plt.show()

if __name__ == "__main__":
    file_path = "edgeMatrix.txt"
    adjacency_matrix = read_adjacency_matrix(file_path)
    plot_graph(adjacency_matrix)
