"""
Plotting the 2D graph of hits based on the 'edgeMatrix.txt' file
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab10
from utils.tools import load_graph_npz

import pandas as pd

# Functions to draw CDCH planes and sectors

# Function to draw circles
def draw_circle(radius, linewidth, color='black'):
    circle = plt.Circle((0, 0), radius, color=color, fill=False, linewidth=linewidth)
    plt.gca().add_patch(circle)

# Function to draw radial lines
def draw_radial_lines(min_radius, max_radius, num_lines, linewidth, color='black'):
    for i in range(num_lines):
        angle = i * (360 / num_lines)
        x_min = min_radius * np.cos(np.radians(angle))
        y_min = min_radius * np.sin(np.radians(angle))
        x_max = max_radius * np.cos(np.radians(angle))
        y_max = max_radius * np.sin(np.radians(angle))
        plt.plot([x_min, x_max], [y_min, y_max], linewidth=linewidth, color=color)
 

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

def plot(hits, edge_matrix, y):
    """
    Plot the graph based on an edge matrix of shape 2 x num_edges
    containing hit ID connected
    """
    #print(np.shape(hits))
    
    # Draw CDCH scheleton
    num_circles = 10
    min_radius = 17
    max_radius = 25
    delta_radius = (max_radius - min_radius) / num_circles

    # Divide sectors with thinner radial lines
    for i in range(12):
        draw_radial_lines(min_radius, max_radius, 12, 0.25)

    # Draw circles with varying radii and linewidths
    for i in range(num_circles):
        radius = min_radius + i * delta_radius
        if i == 0 or i == num_circles - 1:
            linewidth = 1.5
        else:
            linewidth = 0.5
        draw_circle(radius, linewidth)
    
    # Load TC pixel geometry
    pixel_geo = np.loadtxt("utils/spxGeometry.txt")

    colors = ["pink", "blue"] # colors for CDCH and TC nodes
    fmts = ["o", "s"] # fmts for CDCH and TC nodes
    signal_color = ['grey', 'blue', 'orange', 'red', 'purple', 'green', 'brown', 'magenta', 'cyan', 'yellow', 'black'] # colors for different turns

    # Plot nodes and edges
    # Watch out: if you load from a npz file, you can access only the normalized x and y coordinates..
    # This is not the case now, so let's keep going
    # Dropping hits_id, we can just use the radius sqrt(x0^2 + y0^2) to assert if it is cdch or spx
    
    mask_spx = hits[:,5].astype(int).astype(bool)
    mask_cdch = ~mask_spx
    
    np.set_printoptions(threshold=np.inf)
    #print(f"We have a total of {np.shape(hits)} hits")
    #print(edge_matrix)
    #print(pd.DataFrame(hits))
    # Plot nodes and edges
    # Watch out: if you load from a npz file, you can access only the normalized x and y coordinates...
    x_min = 1000
    y_min = 1000
    x_max = -1000
    y_max = -1000
    

    for i, hit in enumerate(hits):

        plt.errorbar(hit[0], hit[1], fmt=fmts[round(hit[7])], alpha=.6, markersize=10, color=signal_color[int(y[i])])


    for k, e in enumerate(edge_matrix.T):

        # nodes ID
        i = int(e[0])
        j = int(e[1])

        # Try to visualize from npz files
        xi = hits[i, 0]
        yi = hits[i, 1]
        if np.sqrt(xi**2 + yi**2) < 29:
            hittype_i = 0
        else:
            hittype_i = 1
    
        xj = hits[j, 0]
        yj = hits[j, 1]
        if np.sqrt(xj**2 + yj**2) < 29:
            hittype_j = 0
        else:
            hittype_j = 1
        
        x_max = max(x_max, max(xi, xj))
        x_min = min(x_min, min(xi, xj))
        y_max = max(y_max, max(yi, yj))
        y_min = min(y_min, min(yi, yj))
        

        """
        #hittype_i, xi, yi = calculate_coordinates(i, pixel_geo)
        plt.errorbar(xi, yi, fmt=fmts[hittype_i], alpha=.6, markersize=10, color=signal_color[int(y[i])])
        #hittype_j, xj, yj = calculate_coordinates(j, pixel_geo)
        plt.errorbar(xj, yj, fmt=fmts[hittype_j], alpha=.5, markersize=10, color=signal_color[int(y[j])])
        """
        
        
        plt.plot([xi, xj], [yi, yj], color='grey', linewidth=0.5, linestyle='-.', alpha=.25)
    plt.draw()  # Ridisegna la figura mantenendo i punti già plottati

    
    plt.axis('off')
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Graph Nodes and Edges')
    plt.show()
    
if __name__ == "__main__":
    """
    Test plot function
    """
    #import build_graph as bg

    #hitIDs = [i for i in range(0, 1920 + 512) if np.random.uniform() > 0.9]
    #edges['edge_index'] = bg.build_graph(hitIDs)
    #plot(edges)

    """
    Test plotting a graph from npz files
    """
    filename = "/home/antu/KaleGraph/graph_files_val_1e6/event1005_sectors1.npz"
    filename = "file01002_event0_sectors0.npz"
    graph = load_graph_npz(filename)

    
    plot(graph['X'], graph['edge_index'], graph['truth'])
