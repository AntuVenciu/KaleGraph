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
from torch_geometric.data import Data, Dataset, HeteroData
from torch_scatter import scatter_add
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






class HeterogenousInteractionNetwork(nn.Module):
    def __init__(self,
                 hidden_size,
                 node_features_cdch_dim,
                 node_features_spx_dim,
                 edge_features_dim,
                 time_steps=1):
        super().__init__()
        
        
        
        
        
        #build update function for edges        
        self.R1 = torch.nn.ModuleDict({
                             'SPX_to_SPX_edge':RelationalModel(2 * node_features_spx_dim + edge_features_dim, edge_features_dim,hidden_size),
                             'CDCH_to_CDCH_edge':RelationalModel(2 * node_features_cdch_dim + edge_features_dim, edge_features_dim,hidden_size),
                             'CDCH_to_SPX_edge':RelationalModel( node_features_cdch_dim +node_features_spx_dim+ edge_features_dim,edge_features_dim, hidden_size), 
                             'SPX_to_CDCH_edge':RelationalModel( node_features_cdch_dim +node_features_spx_dim+ edge_features_dim,edge_features_dim, hidden_size) 
                             })
        
        #build update function for nodes        
        self.O = torch.nn.ModuleDict({
                            'SPXHit':ObjectModel(node_features_spx_dim + edge_features_dim, node_features_spx_dim, hidden_size),
                            'CDCHHit':ObjectModel(node_features_cdch_dim + edge_features_dim, node_features_cdch_dim, hidden_size)
                            })
        
        
        
        #build classifier function for nodes: here change output dim from 1 to max_n_turns + 1 (accounting for the case 0 = noise)
        self.R2 = torch.nn.ModuleDict({
                             'SPXHit':RelationalModel( node_features_spx_dim + edge_features_dim, max_n_turns + 1, hidden_size),
                             'CDCHHit':RelationalModel(node_features_cdch_dim + edge_features_dim, max_n_turns + 1, hidden_size)
                             })
        
        
        
        
        self.E: Tensor = Tensor()

        self.T = time_steps
        
        
        
    def forward(self, data: HeteroData):
        x_dict = {k: v.to(torch.float32) for k, v in data.x_dict.items()}
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = {k: data[k].edge_attr.to(torch.float32) for k in edge_index_dict}

        edge_feature_dim = 3
        agg_msg_dict = {k: torch.zeros(x.shape[0], edge_feature_dim, device=x.device) for k, x in x_dict.items()}
	
        for t in range(self.T):
            # Step 1: compute messages (R1)
            for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
                
                #print()
                
                src_x = x_dict[src_type][edge_index[0]]
                dst_x = x_dict[dst_type][edge_index[1]]
                edge_attr = edge_attr_dict[(src_type, rel_type, dst_type)]
                
                
                msg = self.R1[rel_type](torch.cat([src_x, dst_x, edge_attr], dim=-1))
    
                # Step 2: aggregate messages using scatter
                dst_index = edge_index[1]
                
                # Non ci sono archi per questa relazione, salta l'aggiornamento
                
                if dst_index.numel() == 0:
                    #print(dst_index)
                    continue
                if(x_dict[dst_type].size(0)== 0):
                    continue
                #agg_msg_dict[dst_type] += scatter_add(msg, dst_index, dim=0, dim_size=x_dict[dst_type].size(0))
                agg_msg_dict[dst_type] = torch.scatter_add(agg_msg_dict[dst_type], 0, dst_index.unsqueeze(1).expand(-1, msg.size(1)), msg)
    	         
            # Step 3: O MLP update delle node features
            updated_x = {}
            for node_type in x_dict:
                updated_x[node_type] = self.O[node_type](torch.cat([x_dict[node_type].to(torch.float32), agg_msg_dict[node_type]], dim=-1))
    
            x_dict = updated_x  # aggiorna per iterazioni successive
    
        # Step 4: R2 MLP per output finale
        out_dict = {}
        for node_type in x_dict:
            out_dict[node_type] = self.R2[node_type](torch.cat([x_dict[node_type], agg_msg_dict[node_type]], dim=-1))
    
        return out_dict
            
                
