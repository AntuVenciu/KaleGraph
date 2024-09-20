"""
Create a PyG dataset for training.
Provided a set of hits with their attributes:
a) build the graph and evaluate edge matrix and attributes;
b) create a unique dictionary with also y true labels and phi and p and theta
of the particle for training.
"""
import os
from time import time

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected

from . import build_graph as bg
from . import plot_graph


class GraphDataset(Dataset):
    def __init__(self,
                 datafile,
                 adj_matrix,
                 max_size=-1,
                 transform=None,
                 pre_transform=None):

        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.file = datafile
        self.adj_matrix = adj_matrix
        #t0 = time()
        self.hits_dataset = bg.load_data(self.file)[:max_size]
        #t1 = time()
        #print(f"{t1 - t0:.3f} s to load dataset")
        
    def len(self):
        return len(self.hits_dataset)

    def evaluate_edge_truth(self, edge_index, next_hits, trackID):
        """
        Given hits label, create a vector of 1s and 0s if
        the edge connects two consecutive true hits.
        """
        #nedges = edge_index.shape[1]
        edges_y = np.array([1 if (next_hits[e[0]] == e[1] or next_hits[e[1]] == e[0]) else 0 for e in edge_index.T]) #np.zeros(nedges)
        #print(f"Fraction of ones / zeros = {tot_ones / (nedges - tot_ones)}")
        return edges_y
        

    def get(self, idx):
        # Load attributes of the graph
        # Here we use some different name convensions for more easy references
        # to literature
        #t0 = time()
        graph = bg.build_graph(self.hits_dataset[idx], self.adj_matrix)
        #t1 = time()
        x = torch.from_numpy(graph['x'])
        edge_attr = torch.from_numpy(graph['edge_attr'])
        edge_index = torch.from_numpy(graph['edge_index'])
        # evaluate truth of edges
        y = torch.from_numpy(self.evaluate_edge_truth(graph['edge_index'], self.hits_dataset[idx]['nextHit'], self.hits_dataset[idx]['trackID']))
        pid = torch.from_numpy(self.hits_dataset[idx]['trackID'])
        mom = torch.from_numpy(self.hits_dataset[idx]['mom'])
        phi = torch.from_numpy(self.hits_dataset[idx]['trackPhi'])
        theta = torch.from_numpy(self.hits_dataset[idx]['trackTheta'])

        # make graph undirected
        #row, col = edge_index
        #row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        #edge_index = torch.stack([row, col], dim=0)
        #edge_attr = torch.cat([edge_attr, -1*edge_attr], dim=1)
        #y = torch.cat([y,y])
        #print(x)
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y, pid=pid, mom=mom, phi=phi, theta=theta)
        data.num_nodes = len(x)
        #t2 = time()
        
        #print(f"{t1 - t0:.3f} s to prepare a graph = {(t1 - t0) / (t2 - t0) * 100:.2f} %  of time in an event")
        return data

    def plot(self, idx, compare_y=[], save_fig=0):
        """
        Plot a graph
        """
        import matplotlib.pyplot as plt
        
        from . import plot_graph as pg

        #Initialize graph
        graph = bg.build_graph(self.hits_dataset[idx], self.adj_matrix)
        y = self.evaluate_edge_truth(graph['edge_index'], self.hits_dataset[idx]['nextHit'], self.hits_dataset[idx]['trackID'])
        x = self.hits_dataset[idx]
        ids = x['wireID']
        truth = x['truth']

        print(f"Number of hits = {len(ids)}")
        print(f"Number of edges = {len(truth)}")

        # pixel geometry for 2D visualization
        pixel_geo = np.loadtxt("utils/spxGeometry.txt")

        # colors
        colors = ["red", "blue"] # red for bad hits, blue for good hits 
        fmts = ["o", "s"] # circle for cdch hits, square for spx hits 
        
        plt.figure(1)
        plt.title("Example Graph in x-y view")
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        
        drawn_hits = []

        # plot nodes and edges:
        for k, e in enumerate(graph['edge_index'].T):
            #print(f"Edge number {k} = {e}")
            # hit IDs and properties
            i = ids[e[0]]
            j = ids[e[1]]
            truth_i = truth[e[0]]
            color_i = colors[int(truth_i)]
            truth_j = truth[e[1]]
            color_j = colors[int(truth_j)]
            hittype_i, x_i, y_i = plot_graph.calculate_coordinates(int(i), pixel_geo)
            if i not in drawn_hits:
                #plt.errorbar(x_i, y_i, fmt=fmts[hittype_i], alpha=.6, markersize=10, color=color_i)
                drawn_hits.append(i)
            hittype_j, x_j, y_j = plot_graph.calculate_coordinates(int(j), pixel_geo)
            if j not in drawn_hits:
                #plt.errorbar(x_j, y_j, fmt=fmts[hittype_j], alpha=.6, markersize=10, color=color_j)
                drawn_hits.append(j)

            # decide color of edge: red if correct not found,  blue if correct found, grey if not correct not found
            ecolor = None
            if len(compare_y) == 0:
                if y[k] == 1:
                    ecolor = 'blue'
                #if y[k] == 0 and truth_i + truth_j < 2:
                #    ecolor = 'yellow'
                if y[k] == 0:
                    ecolor = 'red'
                plt.plot([x_i, x_j], [y_i, y_j], ecolor, linewidth=.5, linestyle='-')

            if len(compare_y) > 0 and len(compare_y) == len(y):
                # compare_y is an array of y output to be compared with the true y
                # color scale is set according to results y - compare_y
                # Red: good connection not found (FN)
                # Green: good connection found (TP)
                # Blue: bad connection found (FP)
                # No connection: bad connection not found (TN)
                if y[k] == 1 and compare_y[k] >= 0.5:
                    ecolor = 'green'
                if y[k] == 1 and compare_y[k] < 0.5:
                    ecolor = 'red'
                if y[k] == 0 and compare_y[k] >= 0.5:
                    ecolor = 'blue'
                #if y[k] == 0 and compare_y[k] < 0.5:
                #    ecolor = 'white'
                plt.plot([x_i, x_j], [y_i, y_j], ecolor, linewidth=.5, linestyle='-')
                
        plt.axis('equal')
        plt.tight_layout()
        if len(compare_y) > 0:
            plt.text(0, 0, 'red: FN\ngreen: TP\nblue: FP')
        # Save the plot or show it
        if save_fig > 0:
            plt.savefig(f"plot_graph_training_{save_fig}.pdf")
        else:
            plt.show()
