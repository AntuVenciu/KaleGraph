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
from torch_geometric.nn.conv import HeteroConv
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid

#We need to check what is the right number of turn. This information comes from the edges, not directly from hits.
max_n_turns = 7

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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)






class HeterogenousInteractionNetwork(MessagePassing):
    def __init__(self,
                 hidden_size,
                 node_features_cdch_dim,
                 edge_features_dim,
                 node_features_spx_dim,
                 time_steps=1):
        super(InteractionNetwork, self).__init__(aggr='sum', 
                                                 flow='source_to_target')
        
        
        
        
        
        #build update function for edges        
        self.R1 = HeteroConv({
                             ('SPXHit', 'SPX_to_SPX_message', 'SPXHit'):RelationalModel(2 * node_features_spx_dim + edge_features_dim, hidden_size),
                             ('CDCHHit', 'CDCH_to_CDCH_message', 'CDCHHit'):RelationalModel(2 * node_features_cdch_dim + edge_features_dim, hidden_size),
                             ('CDCHHit', 'CDCH_to_SPX_message', 'SPXHit'):RelationalModel( node_features_cdch_dim +node_features_spx_dim+ edge_features_dim, hidden_size), 
                             ('SPXHit', 'SPX_to_CDCH_message', 'CDCHHit'):RelationalModel( node_features_cdch_dim +node_features_spx_dim+ edge_features_dim, hidden_size) 
                             })
        
        #build update function for nodes        
        self.O = HeteroConv({
                            ('SPXHit', 'SPX_to_SPX_message', 'SPXHit'):ObjectModel(node_features_spx_dim + edge_features_dim, node_features_spx_dim, hidden_size),
                            ('CDCHHit', 'CDCH_to_CDCH_message', 'CDCHHit'):ObjectModel(node_features_cdch_dim + edge_features_dim, node_features_cdch_dim, hidden_size),
                            ('CDCHHit', 'CDCH_to_SPX_message', 'SPXHit'):ObjectModel(node_features_cdch_dim + edge_features_dim, node_features_spx_dim, hidden_size) ,
                            ('SPXHit', 'SPX_to_CDCH_message', 'CDCHHit'):ObjectModel(node_features_spx_dim + edge_features_dim, node_features_cdch_dim, hidden_size) 
                            })
        
        
        
        #build classifier function for edges: here change output dim from 1 to max_n_turns + 1 (accounting for the case 0 = noise)
        self.R2 = HeteroConv({
                             ('SPXHit', 'SPX_to_SPX_message', 'SPXHit'):RelationalModel(2 * node_features_spx_dim + edge_features_dim, max_n_turns + 1, hidden_size),
                             ('CDCHHit', 'CDCH_to_CDCH_message', 'CDCHHit'):RelationalModel(2 * node_features_cdch_dim + edge_features_dim, max_n_turns + 1, hidden_size),
                             ('CDCHHit', 'CDCH_to_SPX_message', 'SPXHit'):RelationalModel( node_features_cdch_dim +node_features_spx_dim+ edge_features_dim, max_n_turns + 1, hidden_size),
                             ('SPXHit', 'SPX_to_CDCH_message', 'CDCHHit'):RelationalModel( node_features_cdch_dim +node_features_spx_dim+ edge_features_dim, max_n_turns + 1, hidden_size)                      
                             })
        
        
        
        
        self.E: Tensor = Tensor()

        self.T = time_steps
        
        
        
    def forward(self, x_dict: dict, edge_index_dict: dict, edge_attr_dict: dict) -> dict:



        # propagate_type: (x: Tensor, edge_attr: Tensor)
        
        
        x_tilde = x_dict
        self.E = edge_attr_dict
        for t in range(self.T):
            for edge_type, edge_index in edge_index_dict.items():
            
                starting_node_label, conn_type ,destination_node_label = edge_type
                x_src = x_tilde[starting_node_label]
                x_dst = x_tilde[destination_node_label]
                edge_attr = self.E[edge_type]

                # propagate
                new_x_tilde[dst] = self.propagate(edge_index, x=x_src, edge_attr=edge_attr, size=None)

        # update the values from the placeholder tensor
        for key in new_x_tilde:
            x_tilde[key] = new_x_tilde[key]

        #so we update the node features and the edges featuees: let us now evaluate the output.
        for edge_type, edge_index in edge_index_dict.items():
        # concatenating features of the nodes and of the edge.
            starting_node_label, conn_type ,destination_node_label = edge_type
            m2 = torch.cat([x_tilde[destination_node_label],
                            x_tilde[starting_node_label],
                            self.E[edge_type]], dim=1)
            m2 = m2.clone().to(torch.float32) # Double -> Float conversion
            self.R2(x_tilde, )
        # return the output of the last linear layer of R2: softmax is applied in the training by the CrossEntropy loss function
        return self.R2(m2) 

        
    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        m1 = m1.clone().to(torch.float32) # Double -> Float conversion
        

        # Compute attention scores
        #attention_scores = self.attention_mlp(m1).clone().to(torch.float32)  # Shape: (num_edges, 1)
        #attention_weights = torch.softmax(attention_scores, dim=0)  # Normalize across edges
        
        
        #self.E = self.R1(m1)*attention_weights
        self.E = self.R1(m1)
        return self.E



    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        c = c.clone().to(torch.float32) # Double -> Float conversion
        return self.O(c) 
        
        
