"""
Create a PyG dataset for training.
"""
import os
from time import time

import numpy as np
import pandas as pd
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
                 pre_transform=None):

        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graph_files = file_names
        
    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        # Load attributes of the graph
        # Here we use some different name convensions for more easy references
        # to literature
        try:
            with np.load(self.graph_files[idx]) as graph:

                x = torch.from_numpy(graph['X'])
                edge_attr = torch.from_numpy(graph['edge_attr'])
                edge_index = torch.from_numpy(graph['edge_index'])
                # evaluate truth of edges
                y = torch.from_numpy(graph['edge_truth'])

                # make graph undirected
                row, col = edge_index
                row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
                edge_index = torch.stack([row, col], dim=0)
                edge_attr = torch.cat([edge_attr, -1*edge_attr], dim=1)
                y = torch.cat([y,y])

                data = Data(x=x,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=y)

                return data

        except FileNotFoundError:
            print(f"{self.graph_files[idx]} doesn't exist.")

    def plot(self, idx):
        """
        Plot a graph
        """
        import matplotlib.pyplot as plt
        
        from utils.plot_graph import plot


        with np.load(self.graph_files[idx]) as graph:
            plot(graph['X'], graph['edge_index'], graph['truth'])
            
        except FileNotFoundError:
            print(f"{self.graph_files[idx]} doesn't exist. Can not plot it.")
