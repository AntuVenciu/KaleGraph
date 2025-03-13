"""
Plotting the 2D graph of hits based on the 'edgeMatrix.txt' file
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab10
from utils.tools import load_graph_npz
from matplotlib.collections import LineCollection


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
    signal_color = ['grey','blue', 'orange', 'red', 'purple', 'green', 'brown', 'magenta', 'cyan', 'yellow', 'black'] # colors for different turns
    ev_value = [0.1,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
    # Plot nodes and edges
    # Watch out: if you load from a npz file, you can access only the normalized x and y coordinates...
    
    mask1 = hits[0] >= 1920
    mask2 = hits[0] < 1920

    #points that are not masked by mask 1: those are spx hits.
    plt.errorbar(hits[1].loc[mask1].to_numpy(), hits[2].loc[mask1].to_numpy(), fmt = fmts[1], markersize = 10, color = colors[1])   
    
    #points that are not masked by mask2 : those are cdch hits 
    plt.errorbar(hits[1].loc[mask2].to_numpy(), hits[2].loc[mask2].to_numpy(), fmt = fmts[0], markersize = 10, color = colors[0])    
    
    
    for i in range(len(hits[1].loc[mask1].to_numpy())):
        plt.text(hits[1].loc[mask1].to_numpy()[i], hits[2].loc[mask1].to_numpy()[i] + 0.5, str(int(hits[0].loc[mask1].to_numpy()[i])), fontsize=8, ha='center', color='black', fontweight='bold')            
    
    
    for i in range(len(hits[1].loc[mask2].to_numpy())):
        plt.text(hits[1].loc[mask2].to_numpy()[i], hits[2].loc[mask2].to_numpy()[i] + 0.5, str(int(hits[0].loc[mask2].to_numpy()[i])), fontsize=8, ha='center', color='black', fontweight='bold')     
    #draw edges.
    ID_FirstNode = edge_matrix.iloc[0].to_numpy()
    ID_SecondNode = edge_matrix.iloc[1].to_numpy()  
    
    
    #WARNING: NOT ALL NODES ARE CONNECTED! so not all nodes are in the edge matrix, put nan as temporary measurement.
    
    x_first_node = np.array([hits.loc[hits[0] == id_, 1].values[0] if not hits.loc[hits[0] == id_, 1].empty else np.nan for id_ in ID_FirstNode])
    y_first_node = np.array([hits.loc[hits[0] == id_, 2].values[0] if not hits.loc[hits[0] == id_, 2].empty else np.nan for id_ in ID_FirstNode])

    x_second_node = np.array([hits.loc[hits[0] == id_, 1].values[0] if not hits.loc[hits[0] == id_, 1].empty else np.nan for id_ in ID_SecondNode])
    y_second_node = np.array([hits.loc[hits[0] == id_, 2].values[0] if not hits.loc[hits[0] == id_, 2].empty else np.nan for id_ in ID_SecondNode])

    # let's find valid indexes
    valid_indices = ~np.isnan(x_first_node) & ~np.isnan(y_first_node) & ~np.isnan(x_second_node) & ~np.isnan(y_second_node)
    
    x_first_node = x_first_node[valid_indices]
    y_first_node = y_first_node[valid_indices]
    x_second_node = x_second_node[valid_indices]
    y_second_node = y_second_node[valid_indices]
   

    y_true = y[0].to_numpy().astype(int)[valid_indices]
    


    segments = np.stack([np.column_stack([x_first_node, y_first_node]), 
                         np.column_stack([x_second_node, y_second_node])], axis=1)
                         
    
    # take only valid edges
    colors = np.array(signal_color)[y_true]
    
    # also here, but this is for visualization effects
    evanescence_values = np.array(ev_value)[y_true]
    
    colors_with_alpha = [(*plt.matplotlib.colors.to_rgba(c)[:3], alpha) for c, alpha in zip(colors, evanescence_values)]

    lc = LineCollection(segments, colors=colors_with_alpha, linewidths=1, linestyles='-')
    
    # plot segments
    ax = plt.gca()  
    ax.add_collection(lc) 
    
    plt.draw()  # Ridisegna la figura mantenendo i punti già plottati
    


    
    plt.axis('off')
    plt.axis('equal')
    plt.xlim(min(min(x_first_node)-1, min(x_second_node)), max(max(x_first_node), max(x_second_node))+1)
    plt.ylim(min(min(y_first_node)-1, min(y_second_node)), max(max(y_first_node), max(y_second_node))+1)
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
    graph = load_graph_npz(filename)
    print(graph['X'].shape)
    print(graph['edge_index'].shape)
    
    plot(graph['X'], graph['edge_index'], graph['truth'])
