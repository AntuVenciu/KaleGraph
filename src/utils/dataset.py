"""
Create a PyG dataset for training.
"""
import os
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected

import build_graph_segmented as bg
from utils import plot_graph


class GraphDataset(Dataset):
    def __init__(self,
                 file_names,
                 maxsize=-1,
                 transform=None,
                 pre_transform=None,
                 scalers=None,
                 fitted=False):

        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graph_files = file_names
        self.scalers = {'X': StandardScaler(), 'edge_attr': StandardScaler()} if scalers is None else scalers
        self._fitted = fitted # Flag to check if scaling is applied

    def len(self):
        return len(self.graph_files)

    def scale(self):
        """
        Fit scalers to dataset
        """

        print("Scaling dataset...")
        
        all_x = []
        all_edge_attr = []

        # Collect all node features and edge attributes to fit scalers
        for file in self.graph_files:
            with np.load(file) as graph:
                all_x.append(graph['X'])
                all_edge_attr.append(graph['edge_attr'])

        # Stack arrays to fit StandardScaler
        all_x = np.vstack(all_x)  # Shape: (num_nodes_total, num_features)
        all_edge_attr = np.vstack(all_edge_attr)  # Shape: (num_edges_total, num_edge_features)

        # Fit scalers
        self.scalers['X'].fit(all_x)
        self.scalers['edge_attr'].fit(all_edge_attr)

        self._fitted = True  # Mark dataset as scaled

        print("Scaling finished!")
        
    
    def get(self, idx):
        # Load attributes of the graph
        # Here we use some different name convensions for more easy references
        # to literature
        try:
            with np.load(self.graph_files[idx]) as graph:

                x = graph['X']
                edge_attr = graph['edge_attr']
                edge_index = graph['edge_index']
                y = graph['truth']
                
                # Apply scaling
                if self._fitted:
                    x = self.scalers['X'].transform(x)
                    edge_attr = self.scalers['edge_attr'].transform(edge_attr)
                else:
                    print("Data not scaled. You may want to check it.")

                # convert to tensors
                x = torch.from_numpy(x).to(torch.float64)
                edge_attr = torch.from_numpy(edge_attr).to(torch.float64)
                edge_index = torch.from_numpy(edge_index).to(torch.int64)
                y = torch.from_numpy(y).to(torch.long)

                # make graph undirected
                """
                row, col = edge_index
                row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
                edge_index = torch.stack([row, col], dim=0)
                edge_attr = torch.cat([edge_attr, -1*edge_attr], dim=1)
                y = torch.cat([y,y])
                """
                    
                data = Data(x=x,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=y)

                return data

        except FileNotFoundError:
            print(f"{self.graph_files[idx]} doesn't exist.")

    def get_X_dim(self):
        """
        Return number of features of nodes X
        """
        try:
            with np.load(self.graph_files[0]) as graph:
                return graph['X'].shape[1]
        except FileNotFoundError:
            print(f"{self.graph_files[0]} doesn't exist.")

    def get_edge_attr_dim(self):
        """
        Return number of features of edges
        """
        try:
            with np.load(self.graph_files[0]) as graph:
                return graph['edge_attr'].shape[1]
        except FileNotFoundError:
            print(f"{self.graph_files[0]} doesn't exist.")

            
    def plot(self, idx):
        """
        Plot a graph
        """
        import matplotlib.pyplot as plt
        
        from utils.plot_graph import plot

        try:
            with np.load(self.graph_files[idx]) as graph:
                plot(graph['X'], graph['edge_index'], graph['truth'])
            
        except FileNotFoundError:
            print(f"{self.graph_files[idx]} doesn't exist. Can not plot it.")
