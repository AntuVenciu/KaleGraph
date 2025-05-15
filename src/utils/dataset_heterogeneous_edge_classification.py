"""
Create a PyG dataset for training an heterogeneous graph.
"""
import os
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
import torch
from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected









class HeterogeneousGraphDatasetEdgeClassification(Dataset):
    def __init__(self,
                 file_names,
                 maxsize=-1,
                 transform=None,
                 pre_transform=None,
                 scalers=None,
                 fitted=False):

        super(HeterogeneousGraphDatasetEdgeClassification, self).__init__(None, transform, pre_transform)
        self.graph_files = file_names
        self.scalers = {'X_cdch': StandardScaler(), 'edge_attr_cdch': StandardScaler() ,
                        'X_spx': StandardScaler(), 'edge_attr_spx': StandardScaler(),  
                        'edge_attr_cdch_spx': StandardScaler()   } if scalers is None else scalers
        self._fitted = fitted # Flag to check if scaling is applied

    def len(self):
        return len(self.graph_files)

    def scale(self):
        """
        Fit scalers to dataset
        """

        print("Scaling dataset...")
        
        all_x_cdch = []
        all_edge_attr_cdch = []
        all_x_spx =[]
        all_edge_attr_spx =[]
        all_edge_attr_cdch_spx =[]
        # Collect all node features and edge attributes to fit scalers
        for file in self.graph_files:
            with np.load(file) as graph:
                all_x_cdch.append(graph['X_cdch'])
                all_edge_attr_cdch.append(graph['edge_attr_cdch'])
                all_x_spx.append(graph['X_spx'])
                all_edge_attr_spx.append(graph['edge_attr_spx'])
                all_edge_attr_cdch_spx.append(graph['edge_attr_cdch_spx'])

        # Stack arrays to fit StandardScaler
        all_x_cdch = np.vstack(all_x_cdch)  # Shape: (num_nodes_total, num_features)
        all_edge_attr_cdch = np.vstack(all_edge_attr_cdch)  # Shape: (num_edges_total, num_edge_features)
        all_x_spx = np.vstack(all_x_spx)
        all_edge_attr_spx = np.vstack(all_edge_attr_spx)
        all_edge_attr_cdch_spx = np.vstack(all_edge_attr_cdch_spx)
        # Fit scalers
        
        
        self.scalers['X_cdch'].fit(all_x_cdch)
        self.scalers['edge_attr_cdch'].fit(all_edge_attr_cdch)
        self.scalers['X_spx'].fit(all_x_spx)
        self.scalers['edge_attr_spx'].fit(all_edge_attr_spx)
        self.scalers['edge_attr_cdch_spx'].fit(all_edge_attr_cdch_spx)
        self._fitted = True  # Mark dataset as scaled

        print("Scaling finished!")
        
    
    def get(self, idx):
        # Load attributes of the graph
        # Here we use some different name convensions for more easy references
        # to literature
        try:
            with np.load(self.graph_files[idx]) as graph:

                x_cdch = graph['X_cdch']
                edge_attr_cdch = graph['edge_attr_cdch']
                edge_index_cdch = graph['edge_index_cdch']
                y_cdch = graph['truth_cdch']
                x_spx = graph['X_spx']
                edge_attr_spx = graph['edge_attr_spx']
                edge_index_spx = graph['edge_index_spx']
                y_spx = graph['truth_spx']
                edge_attr_cdch_spx = graph['edge_attr_cdch_spx']
                edge_index_cdch_spx = graph['edge_index_cdch_spx']
                y_cdch_spx = graph['truth_cdch_spx']
                
                
                
                
                # Apply scaling
                if self._fitted:
                    x_cdch = self.scalers['X_cdch'].transform(x_cdch)
                    edge_attr_cdch = self.scalers['edge_attr_cdch'].transform(edge_attr_cdch)
                    x_spx = self.scalers['X_spx'].transform(x_spx)
                    edge_attr_spx = self.scalers['edge_attr_spx'].transform(edge_attr_spx)
                    edge_attr_cdch_spx = self.scalers['edge_attr_cdch_spx'].transform(edge_attr_cdch_spx)
                else:
                    print("Data not scaled. You may want to check it.")

                # convert to tensors
                x_cdch = torch.from_numpy(x_cdch).to(torch.float64)
                edge_attr_cdch = torch.from_numpy(edge_attr_cdch).to(torch.float64)
                edge_index_cdch = torch.from_numpy(edge_index_cdch).to(torch.int64)
                y_cdch = torch.from_numpy(y_cdch).to(torch.long)

                x_spx = torch.from_numpy(x_spx).to(torch.float64)
                edge_attr_spx = torch.from_numpy(edge_attr_spx).to(torch.float64)
                edge_index_spx = torch.from_numpy(edge_index_spx).to(torch.int64)
                y_spx = torch.from_numpy(y_spx).to(torch.long)
                
                edge_attr_cdch_spx = torch.from_numpy(edge_attr_cdch_spx).to(torch.float64)
                edge_index_cdch_spx= torch.from_numpy(edge_index_cdch_spx).to(torch.int64)
                y_cdch_spx = torch.from_numpy(y_cdch_spx).to(torch.long)
                
                
                
                # make graph undirected
                """
                row, col = edge_index
                row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
                edge_index = torch.stack([row, col], dim=0)
                edge_attr = torch.cat([edge_attr, -1*edge_attr], dim=1)
                y = torch.cat([y,y])
                """
                
                
                data = HeteroData()
                data['CDCHHit'].x = x_cdch
                data['SPXHit'].x = x_spx
                
                data['SPXHit', 'SPX_to_SPX_edge', 'SPXHit'].edge_index = edge_index_spx
                data['CDCHHit', 'CDCH_to_CDCH_edge', 'CDCHHit'].edge_index = edge_index_cdch
                data['CDCHHit', 'CDCH_to_SPX_edge', 'SPXHit'].edge_index = edge_index_cdch_spx
                data['SPXHit', 'SPX_to_CDCH_edge', 'CDCHHit'].edge_index = edge_index_cdch_spx
                
                data['SPXHit', 'SPX_to_SPX_edge', 'SPXHit'].edge_attr = edge_attr_spx
                data['CDCHHit', 'CDCH_to_CDCH_edge', 'CDCHHit'].edge_attr = edge_attr_cdch
                data['CDCHHit', 'CDCH_to_SPX_edge', 'SPXHit'].edge_attr = edge_attr_cdch_spx
                data['SPXHit', 'SPX_to_CDCH_edge', 'CDCHHit'].edge_attr = edge_attr_cdch_spx
                
                data['SPXHit', 'SPX_to_SPX_edge', 'SPXHit'].edge_label = y_spx
                data['CDCHHit', 'CDCH_to_CDCH_edge', 'CDCHHit'].edge_label = y_cdch
                data['CDCHHit', 'CDCH_to_SPX_edge', 'SPXHit'].edge_label = y_cdch_spx
                data['SPXHit', 'SPX_to_CDCH_edge', 'CDCHHit'].edge_label = torch.zeros(len(y_cdch_spx))
                
                """
                data = Data(x_cdch=x_cdch,
                            edge_index_cdch=edge_index_cdch,
                            edge_attr_cdch=edge_attr_cdch,
                            x_spx=x_spx,
                            edge_index_spx=edge_index_spx,
                            edge_attr_spx=edge_attr_spx,
                            edge_index_cdch_spx=edge_index_cdch_spx,
                            edge_attr_cdch_spx=edge_attr_cdch_spx,
                            y_cdch=y_cdch,
                            y_spx = y_spx,
                            y_cdch_spx = y_cdch_spx
                            )
                """
                
                
                return data

        except FileNotFoundError:
            print(f"{self.graph_files[idx]} doesn't exist.")

    def get_X_CDCH_dim(self):
        """
        Return number of features of nodes X
        """
        try:
            with np.load(self.graph_files[0]) as graph:
                return graph['X_cdch'].shape[1]
        except FileNotFoundError:
            print(f"{self.graph_files[0]} doesn't exist.")

    def get_edge_attr_cdch_dim(self):
        """
        Return number of features of edges
        """
        try:
            with np.load(self.graph_files[0]) as graph:
                return graph['edge_attr_cdch'].shape[1]
        except FileNotFoundError:
            print(f"{self.graph_files[0]} doesn't exist.")
    def get_X_SPX_dim(self):
        """
        Return number of features of nodes X
        """
        try:
            with np.load(self.graph_files[0]) as graph:
                return graph['X_spx'].shape[1]
        except FileNotFoundError:
            print(f"{self.graph_files[0]} doesn't exist.")

    def get_edge_attr_spx_dim(self):
        """
        Return number of features of edges
        """
        try:
            with np.load(self.graph_files[0]) as graph:
                return graph['edge_attr_spx'].shape[1]
        except FileNotFoundError:
            print(f"{self.graph_files[0]} doesn't exist.")
    
    def get_edge_attr_cdch_spx_dim(self):
        """
        Return number of features of edges
        """
        try:
            with np.load(self.graph_files[0]) as graph:
                return graph['edge_attr_cdch_spx'].shape[1]
        except FileNotFoundError:
            print(f"{self.graph_files[0]} doesn't exist.")
    
            
    def plot(self, idx):
        """
        Plot a graph
        """
        import matplotlib.pyplot as plt
        
        from utils.plot_graph_heterogeneous_edge_classification import plot_Heterogenous_graph

        try:
            with np.load(self.graph_files[idx]) as graph:
                plot_Heterogenous_graph(graph['X_cdch'], graph['edge_index_cdch'], graph['truth_cdch'],
                                                graph['X_spx'], graph['edge_index_spx'], graph['truth_spx'],
                                                graph['edge_index_cdch_spx'],graph['truth_cdch_spx']
                                               )
            
        except FileNotFoundError:
            print(f"{self.graph_files[idx]} doesn't exist. Can not plot it.")       
