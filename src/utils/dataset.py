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

from . import build_graph as bg


def load_data(filename):
    """
    Load hits data from a file
    and transform them into a pandas dataframe.
    Separate events looking at hit ID
    """
    hitID, wireID, time, layer, x, y, z, truth, trackID, mom, trackPhi, trackTheta = np.loadtxt(filename, unpack=True)
    events_separators = [i for i, k in enumerate(hitID) if k==0]
    feature_names = ['wireID', 't', 'layer', 'x', 'y', 'z', 'phiWire', 'thetaWire', 'truth', 'trackID', 'mom', 'trackPhi', 'trackTheta']
    events = [{'wireID' : wireID[events_separators[i] : events_separators[i + 1]],
               't' : time[events_separators[i] : events_separators[i + 1]],
               'layer' : layer[events_separators[i] : events_separators[i + 1]],
               'x' : x[events_separators[i] : events_separators[i + 1]],
               'y' : y[events_separators[i] : events_separators[i + 1]],
               'z' : z[events_separators[i] : events_separators[i + 1]],
               'truth' : truth[events_separators[i] : events_separators[i + 1]],
               'trackID' : trackID[events_separators[i] : events_separators[i + 1]],
               'mom' : mom[events_separators[i] : events_separators[i + 1]],
               'trackPhi' : trackPhi[events_separators[i] : events_separators[i + 1]],
               'trackTheta' : trackTheta[events_separators[i] : events_separators[i + 1]]}
              for i in range(0, len(events_separators) - 1)]

    return events

class GraphDataset(Dataset):
    def __init__(self,
                 datafile,
                 adj_matrix,
                 transform=None,
                 pre_transform=None):

        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.file = datafile
        self.adj_matrix = adj_matrix
        self.hits_dataset = load_data(self.file)

    def len(self):
        return len(self.hits_dataset)

    def evaluate_edge_truth(self, edge_index, hits_truth):
        """
        Given hits label, create a vector of 1s and 0s if
        the edge connects two true hits.
        """
        nedges = edge_index.shape[1]
        edges_y = np.zeros(nedges)
        for k, e in enumerate(edge_index):
            i = e[0]
            j = e[1]
            if hits_truth[i] and hits_truth[j]:
                edges_y[k] = 1
        return edges_y
        

    def get(self, idx):
        # Load attributes of the graph
        # Here we use some different name convensions for more easy references
        # to literature
        graph = bg.build_graph(self.hits_dataset[idx], self.adj_matrix)
        x = torch.from_numpy(graph['x'])
        edge_attr = torch.from_numpy(graph['edge_attr'])
        edge_index = torch.from_numpy(graph['edge_index'])
        # evaluate truth of edges
        y = torch.from_numpy(self.evaluate_edge_truth(graph['edge_index'], self.hits_dataset[idx]['truth']))
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

        return data
