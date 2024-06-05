"""
Create a PyG dataset for training.
Provided a set of hits with their attributes:
a) build the graph and evaluate edge matrix and attributes;
b) create a unique dictionary with also y true labels and phi and p and theta
of the particle for training.
"""
import os

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected

import build_graph as bg
import plot_graph as pg


class GraphDataset(Dataset):
    def __init__(self,
                 hits,
                 adj_matrix,
                 transform=None,
                 pre_transform=None):

        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.hits = hits
        self.adj_matrix = adj_matrix
        # Evaluate the graph
        self.graph = bg.build_graph(hits, adj_matrix)

    def __call__(self):
        # Load attributes of the graph
        x = torch.from_numpy(self.graph['x'])
        edge_attr = torch.from_numpy(self.graph['edge_attr'])
        edge_index = torch.from_numpy(self.graph['edge_index'])
        y = torch.from_numpy(hits['truth'])
        pid = torch.from_numpy(hits['trackID'])
        mom = torch.from_numpy(hits['mom'])
        phi = torch.from_numpy(hits['trackPhi'])
        theta = torch.from_numpy(hits['trackTheta'])

        # make graph undirected
        row, col = edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = torch.cat([edge_attr, -1*edge_attr], dim=1)
        y = torch.cat([y,y])

        data = Data(x=x, edge_index=edge_index,
                    edge_attr=torch.transpose(edge_attr, 0, 1),
                    y=y, pid=pid, mom=mom, phi=phi, theta=theta)
        data.num_nodes = len(x)

        return data

    def plot(self):
        """
        Plot this graph
        """
        pg.plot(self.graph['edge_index'])


def load_data(filename):
    """
    Load hits data from a file
    and transform them into a pandas dataframe.
    Separate events looking at hit ID
    """
    hitID, wireID, time, layer, x, y, z, phiWire, thetaWire, truth, trackID, mom, trackPhi, trackTheta = np.loadtxt(filename, unpack=True)
    events_separators = [k if k==0 for k in hitID]
    feature_names = ['wireID', 't', 'layer', 'x', 'y', 'z', 'phiWire', 'thetaWire', 'truth', 'trackID', 'mom', 'trackPhi', 'trackTheta']
    events = [{'wireID' : wireID[events_separators[i] : events_separators[i + 1]],
               't' : time[events_separators[i] : events_separators[i + 1]],
               'layer' : layer[events_separators[i] : events_separators[i + 1]],
               'x' : x[events_separators[i] : events_separators[i + 1]],
               'y' : y[events_separators[i] : events_separators[i + 1]],
               'z' : z[events_separators[i] : events_separators[i + 1]],
               'phiWire' : phiWire[events_separators[i] : events_separators[i + 1]],
               'thetaWire' : thetaWire[events_separators[i] : events_separators[i + 1]],
               'truth' : truth[events_separators[i] : events_separators[i + 1]],
               'trackID' : trackID[events_separators[i] : events_separators[i + 1]],
               'mom' : mom[events_separators[i] : events_separators[i + 1]],
               'trackPhi' : trackPhi[events_separators[i] : events_separators[i + 1]],
               'trackTheta' : trackTheta[events_separators[i] : events_separators[i + 1]]}]
    return events
    
