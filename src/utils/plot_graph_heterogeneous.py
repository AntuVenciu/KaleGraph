"""
Plotting the 2D graph of hits based on the 'edgeMatrix.txt' file
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab10
from utils.tools import load_graph_npz

import pandas as pd




def plot_Heterogenous_graph(hits_cdch, edge_matrix_cdch, y_cdch, hits_spx, edge_matrix_spx, y_spx,edge_matrix_cdch_spx, y_cdch_spx ):
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
    signal_color = ['blue', 'orange', 'red', 'purple', 'green', 'brown', 'magenta', 'cyan', 'yellow', 'black'] # colors for different turns
    plt.errorbar(hits_cdch[:,0], hits_cdch[:,1], fmt = fmts[0], markersize = 10, color = colors[0])
    plt.errorbar(hits_spx[:,0], hits_spx[:,1], fmt = fmts[1], markersize = 10, color = colors[1])

    x_min = 1000
    y_min = 1000
    x_max = -1000
    y_max = -1000

    for k, e in enumerate(edge_matrix_cdch.T):

        # nodes ID
        i = int(e[0])
        j = int(e[1])

        # Try to visualize from npz files
        xi = hits_cdch[i, 0]
        yi = hits_cdch[i, 1]
        
    
        xj = hits_cdch[j, 0]
        yj = hits_cdch[j, 1]
        
        
        x_max = max(x_max, max(xi, xj))
        x_min = min(x_min, min(xi, xj))
        y_max = max(y_max, max(yi, yj))
        y_min = min(y_min, min(yi, yj))
        
        if y_cdch[k] > 0:
            plt.plot([xi, xj], [yi, yj], color=signal_color[int(y_cdch[k]) - 1], linewidth=2, linestyle='-', alpha=1)
        else:
            plt.plot([xi, xj], [yi, yj], color='grey', linewidth=0.5, linestyle='-.', alpha=.25)
            
    for k, e in enumerate(edge_matrix_spx.T):

        # nodes ID
        i = int(e[0])
        j = int(e[1])

        # Try to visualize from npz files
        xi = hits_spx[i, 0]
        yi = hits_spx[i, 1]
        
    
        xj = hits_spx[j, 0]
        yj = hits_spx[j, 1]
        
        
        x_max = max(x_max, max(xi, xj))
        x_min = min(x_min, min(xi, xj))
        y_max = max(y_max, max(yi, yj))
        y_min = min(y_min, min(yi, yj))
        
        if y_spx[k] > 0:
            plt.plot([xi, xj], [yi, yj], color=signal_color[int(y_spx[k]) - 1], linewidth=2, linestyle='-', alpha=1)
        else:
            plt.plot([xi, xj], [yi, yj], color='grey', linewidth=0.5, linestyle='-.', alpha=.25)        
            
            
    for k, e in enumerate(edge_matrix_cdch_spx.T):

        # nodes ID
        i = int(e[0])
        j = int(e[1])
        
        # Try to visualize from npz files
        xi = hits_cdch[i, 0]
        yi = hits_cdch[i, 1]
        
    
        xj = hits_spx[j, 0]
        yj = hits_spx[j, 1]
        
        
        x_max = max(x_max, max(xi, xj))
        x_min = min(x_min, min(xi, xj))
        y_max = max(y_max, max(yi, yj))
        y_min = min(y_min, min(yi, yj))
        
        if y_cdch_spx[k] > 0:
            print(y_cdch_spx[k])
            print(i)
            print(j)
            print(len(hits_cdch))
            plt.plot([xi, xj], [yi, yj], color=signal_color[int(y_cdch_spx[k]) - 1], linewidth=2, linestyle='-', alpha=1)
        else:
            plt.plot([xi, xj], [yi, yj], color='grey', linewidth=0.5, linestyle='-.', alpha=.25)
                
    plt.draw()  # Ridisegna la figura mantenendo i punti già plottati

    
    plt.axis('off')
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Graph Nodes and Edges')
    plt.show()

