import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, knn

### helpers here

def gauss(x):
    return torch.exp(-1 * x * x)

def gauss_of_lin(x):
    return torch.exp(-1 * torch.abs(x))

def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.
    """
    sub_factor = -2 * torch.matmul(A, B.transpose(1, 2))  # -2ab term
    dotA = A.pow(2).sum(dim=2, keepdim=True)  # a^2 term
    dotB = B.pow(2).sum(dim=2, keepdim=True).transpose(1, 2)  # b^2 term
    return torch.abs(sub_factor + dotA + dotB)

def nearest_neighbor_matrix(spatial_features, k=10):
    """
    Nearest neighbors matrix given spatial features.

    :param spatial_features: Spatial features of shape [B, N, S] where B = batch size, N = max examples in batch,
                             S = spatial features
    :param k: Max neighbors
    :return:
    """
    D = euclidean_squared(spatial_features, spatial_features)
    topk = torch.topk(-D, k, dim=-1)
    return topk.indices, -topk.values

def indexing_tensor(spatial_features, k=10):
    shape_spatial_features = spatial_features.shape
    n_batch = shape_spatial_features[0]
    n_max_entries = shape_spatial_features[1]

    neighbor_matrix, distance_matrix = nearest_neighbor_matrix(spatial_features, k)

    batch_range = torch.arange(n_batch).view(n_batch, 1, 1, 1)
    batch_range = batch_range.expand(n_batch, n_max_entries, k, 1)
    expanded_neighbor_matrix = neighbor_matrix.unsqueeze(3)

    indexing_tensor = torch.cat([batch_range, expanded_neighbor_matrix], dim=3)
    return indexing_tensor.long(), distance_matrix

def high_dim_dense(inputs, nodes, **kwargs):
    if len(inputs.shape) == 3:
        return torch.nn.Conv1d(inputs.shape[2], nodes, kernel_size=1, **kwargs)(inputs.permute(0, 2, 1)).permute(0, 2, 1)
    elif len(inputs.shape) == 4:
        return torch.nn.Conv2d(inputs.shape[3], nodes, kernel_size=1, **kwargs)(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    elif len(inputs.shape) == 5:
        return torch.nn.Conv3d(inputs.shape[4], nodes, kernel_size=1, **kwargs)(inputs.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

def apply_edges(vertices, edges, reduce_sum=True, flatten=True, expand_first_vertex_dim=True, aggregation_function=torch.max):
    '''
    edges are naturally BxVxV'xF
    vertices are BxVxF'  or BxV'xF'
    This function returns BxVxF'' if flattened and summed
    '''
    edges = edges.unsqueeze(3)
    if expand_first_vertex_dim:
        vertices = vertices.unsqueeze(1)
    vertices = vertices.unsqueeze(4)

    out = edges * vertices  # [BxVxV'x1xF] x [Bx1xV'xF'x1] = [BxVxV'xFxF']

    if reduce_sum:
        out = aggregation_function(out, dim=2)[0]
    if flatten:
        out = out.view(out.shape[0], out.shape[1], -1)

    return out

### 
### 
### 
### 
### 
### 
### actual layers
### 
### 
### 
### 
### 
### 

class LayerGarNet(torch.nn.Module):
    def __init__(self, n_aggregators, n_filters, n_propagate, plus_mean=True):
        super(LayerGarNet, self).__init__()
        self.n_aggregators = n_aggregators
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.plus_mean = plus_mean

        self.dense_vertices_in = torch.nn.Linear(n_propagate, n_propagate)
        self.dense_agg_nodes = torch.nn.Linear(n_propagate, n_aggregators)
        self.dense_expanded_collapsed = high_dim_dense

    def forward(self, vertices_in):
        vertices_in_orig = vertices_in
        vertices_in = self.dense_vertices_in(vertices_in)

        agg_nodes = self.dense_agg_nodes(vertices_in_orig)
        agg_nodes = gauss_of_lin(agg_nodes)
        vertices_in = torch.cat([vertices_in, agg_nodes], dim=-1)

        edges = agg_nodes.unsqueeze(3)
        edges = edges.permute(0, 2, 1, 3)

        vertices_in_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True)
        vertices_in_mean_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True, aggregation_function=torch.mean)

        vertices_in_collapsed = torch.cat([vertices_in_collapsed, vertices_in_mean_collapsed], dim=-1)

        edges = edges.permute(0, 2, 1, 3)
        expanded_collapsed = apply_edges(vertices_in_collapsed, edges, reduce_sum=False, flatten=True)

        expanded_collapsed = torch.cat([vertices_in_orig, expanded_collapsed, agg_nodes], dim=-1)

        merged_out = self.dense_expanded_collapsed(expanded_collapsed, self.n_filters, activation=torch.nn.Tanh())
        return merged_out

class LayerGravNet(torch.nn.Module):
    def __init__(self, n_neighbours, n_dimensions, n_filters, n_propagate):
        super(LayerGravNet, self).__init__()
        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate

        self.dense_vertices_prop = high_dim_dense
        self.dense_neighb_dimensions = high_dim_dense

    def collapse_to_vertex(self, indexing, distance, vertices):
        neighbours = torch.gather(vertices, 1, indexing)  # BxVxNxF
        distance = distance.unsqueeze(3)
        distance = distance * 10.  # input is tanh activated or batch normed, allow for some more spread
        edges = gauss(distance)[:, :, 1:, :]
        neighbours = neighbours[:, :, 1:, :]
        scaled_feat = edges * neighbours
        collapsed = torch.max(scaled_feat, dim=2)[0]
        collapsed_mean = torch.mean(scaled_feat, dim=2)
        collapsed = torch.cat([collapsed, collapsed_mean], dim=-1)
        return collapsed

    def forward(self, vertices_in):
        vertices_prop = self.dense_vertices_prop(vertices_in, self.n_propagate, activation=None)
        neighb_dimensions = self.dense_neighb_dimensions(vertices_in, self.n_dimensions, activation=None)

        indexing, distance = indexing_tensor(neighb_dimensions, self.n_neighbours)
        collapsed = self.collapse_to_vertex(indexing, distance, vertices_prop)
        updated_vertices = torch.cat([vertices_in, collapsed], dim=-1)

        return self.dense_vertices_prop(updated_vertices, self.n_filters, activation=torch.nn.Tanh())

class LayerGlobalExchange(torch.nn.Module):
    def forward(self, vertices_in):
        global_summed = torch.mean(vertices_in, dim=1, keepdim=True)
        global_summed = global_summed.expand(-1, vertices_in.shape[1], -1)
        vertices_out = torch.cat([vertices_in, global_summed], dim=-1)
        return vertices_out
