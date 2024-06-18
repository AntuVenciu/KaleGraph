"""
Declare the Interaction Network Model
"""
import torch
import torch_geometric
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        return self.layers(m)

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)

class InteractionNetwork(MessagePassing):
    def __init__(self,
                 hidden_size,
                 node_features_dim,
                 edge_features_dim,
                 time_steps=1):
        super(InteractionNetwork, self).__init__(aggr='mean', 
                                                 flow='source_to_target')
        
        self.R1 = RelationalModel(2 * node_features_dim + edge_features_dim, edge_features_dim, hidden_size)
        
        self.O = ObjectModel(node_features_dim + edge_features_dim, node_features_dim, hidden_size)
        
        self.R2 = RelationalModel(2 * node_features_dim + edge_features_dim, 1, hidden_size)
        
        self.E: Tensor = Tensor()

        self.T = time_steps
        
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:

        # propagate_type: (x: Tensor, edge_attr: Tensor)
        x_tilde = x
        self.E = edge_attr
        for t in range(self.T):
            x_tilde = self.propagate(edge_index, x=x_tilde, edge_attr=self.E, size=None)
        #print(f"x_tilde shape = {x_tilde.shape}\nx shape = {x.shape}\nx_tilde = {x_tilde}\nx = {x}")
        #print(f"E shape = {self.E.shape}\nedge attr shape = {edge_attr.shape}\nE = {self.E}\nedge attr = {edge_attr}")
        
        m2 = torch.cat([x_tilde[edge_index[1]],
                        x_tilde[edge_index[0]],
                        self.E], dim=1)
        m2 = m2.clone().to(torch.float32) # Double -> Float conversion
        return self.R2(m2)#torch.sigmoid(self.R2(m2))

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        m1 = m1.clone().to(torch.float32) # Double -> Float conversion
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        c = c.clone().to(torch.float32) # Double -> Float conversion
        return self.O(c) 
