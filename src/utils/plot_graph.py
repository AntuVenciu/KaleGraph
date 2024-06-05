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

def plot(edge_matrix):
    """
    Plot the graph based on an edge matrix of shape 2 x num_edges
    containing hit ID connected
    """
    
    # Load TC pixel geometry
    pixel_geo = np.loadtxt("spxGeometry.txt")

    colors = ["pink", "blue"] # colors for CDCH and TC nodes
    fmts = ["o", "s"] # fmts for CDCH and TC nodes

    # Plot nodes and edges 
    for e in edge_matrix:
        # nodes ID
        i = e[0]
        j = e[1]
        
        hittype_i, xi, yi = calculate_coordinates(i, pixel_geo)
        plt.errorbar(xi, yi, fmt=fmts[hittype_i], alpha=.6, markersize=10, color=colors[hittype_i])
        hittype_j, xj, yj = calculate_coordinates(j, pixel_geo)
        plt.errorbar(xj, yj, fmt=fmts[hittype_j], alpha=.5, markersize=10, color=colors[hittype_j])
        plt.plot([xi, xj], [yi, yj], 'grey', linewidth=0.5, linestyle='-', alpha=.2)

    plt.axis('off')
    plt.axis('equal')
    plt.title('Graph Nodes and Edges')
    plt.show()

if __name__ == "__main__":
    """
    Test plot function
    """
    import build_graph as bg

    hitIDs = [i for i in range(0, 1920 + 512) if np.random.uniform() > 0.9]
    edges['edge_index'] = bg.build_graph(hitIDs)
    plot(edges)
