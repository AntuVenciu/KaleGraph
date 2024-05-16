"""
Plotting the 2D graph of hits based on the 'edgeMatrix.txt' file
"""
import numpy as np
import matplotlib.pyplot as plt


def calculate_coordinates(node_id, pixel_geo):
    # Evaluate x-y coordinate of a wire and TC tile at CDCH center:
    # input:
    # node_id = wire ID (if < 1920) or pixelID (if > 1920)
    if node_id < 1920:
        layer = int((node_id - 192) / 192)
        wireID_in_layer = node_id % 192
        radius = 270 - layer * 10;
        angle = wireID_in_layer * 360 / 192 * (np.pi / 180);
        x = radius * np.cos(-(angle) + 30. * np.pi / 180.)
        y = radius * np.sin(-(angle) + 30. * np.pi / 180.)
        hittype = 0
    else:
        pixel_id = node_id - 1920
        x = pixel_geo[pixel_id][1] * 10
        y = pixel_geo[pixel_id][2] * 10
        hittype = 1
    return hittype, x, y

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

    number_of_nodes = 0
    number_of_edges = 0

    pixel_geo = np.loadtxt("spxGeometry.txt")

    colors = ["pink", "blue"] # colors for CDCH and TC nodes
    fmts = ["o", "s"] # fmts for CDCH and TC nodes

    # Plot nodes and edges 
    for node_id in range(num_nodes):
        # Put a cut on connections contributing less then x in the number of connections
        # of a node
        norm_cdch = np.array(adjacency_matrix[node_id][:1920]).sum()
        cut_cdch = norm_cdch * 0.015
        norm_spx = np.array(adjacency_matrix[node_id][1920:]).sum()
        cut_spx = norm_spx * 0.005
        for j in range(num_nodes):
            cut = cut_cdch
            if j > 1920 or node_id > 1920:
                cut = cut_spx
            # Plot only connected edges
            if adjacency_matrix[node_id][j] > 0:

                # Plot nodes
                if not is_node_plotted[node_id]:
                    hittype, node_x, node_y = calculate_coordinates(node_id, pixel_geo)
                    plt.errorbar(node_x, node_y, fmt=fmts[hittype], alpha=.5, markersize=10, color=colors[hittype])
                    is_node_plotted[node_id] = 1
                    number_of_nodes += 1
                if not is_node_plotted[j]:
                    hittype, node_x, node_y = calculate_coordinates(j, pixel_geo)
                    plt.errorbar(node_x, node_y, fmt=fmts[hittype], alpha=.5, markersize=10, color=colors[hittype])
                    is_node_plotted[j] = 1
                    number_of_nodes += 1
                # Plot edges
                if adjacency_matrix[node_id][j] > cut:
                    hittype, xi, yi = calculate_coordinates(node_id, pixel_geo)
                    hittype, xj, yj = calculate_coordinates(j, pixel_geo)
                    plt.plot([xi, xj], [yi, yj], 'k-', linewidth=0.5, alpha=.5)
                    number_of_edges += 1

    print(f"Number of nodes = {number_of_nodes}")
    print(f"Number of edges = {number_of_edges}")

    plt.axis('off')
    plt.axis('equal')
    plt.title('Graph Nodes and Edges')
    plt.show()

if __name__ == "__main__":
    file_path = "edgeMatrix.txt"
    adjacency_matrix = read_adjacency_matrix(file_path)
    plot_graph(adjacency_matrix)
