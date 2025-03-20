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

#We need to check what is the right number of turn. This information comes from the edges, not directly from hits.
max_n_turns = 5

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
    """
    In the paper "Charged Particle Tracking via Edge-Classifying Interation Network",
    The Interaction Network is built as:
    A relational model R1 acting to transform edge representation eij
    eij = R1 (eij, xi, xj)
    A object model O acts to transform node features xi
    xi = O (xi, Sum R1 (eij))
    A second relational model R2 acts to classify edges (output size = max_n_turns)
    w = R2 (eij, xi, xj)
    Using the variable T, the message passing can be performed T times.
    """
    def __init__(self,
                 hidden_size,
                 node_features_dim,
                 edge_features_dim,
                 time_steps=1):
        super(InteractionNetwork, self).__init__(aggr='mean', 
                                                 flow='source_to_target')
        #build update function for edges
        self.R1 = RelationalModel(2 * node_features_dim + edge_features_dim, edge_features_dim, hidden_size)
        #build update function for nodes
        self.O = ObjectModel(node_features_dim + edge_features_dim, node_features_dim, hidden_size)
        #build classifier function for edges: here change output dim from 1 to max_n_turns + 1 (accounting for the case 0 = noise)
        self.R2 = RelationalModel(2 * node_features_dim + edge_features_dim, max_n_turns + 1, hidden_size)
        
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
        
        #apply softmax to impose multiclass 
        #return torch.softmax(self.R2(m2), dim = 1)#torch.sigmoid(self.R2(m2))
        return self.R2(m2) #torch.sigmoid(self.R2(m2))
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
