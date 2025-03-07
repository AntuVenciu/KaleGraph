"""
Functions to build event sub-graphs corresponding to
a set of sectors.
We make use of a pandas DataFrame to store hit and informations
"""
import os

import numpy as np
import pandas as pd
import torch


def load_data(filename):
    """
    Load hits data from a file
    and transform them into a pandas dataframe.
    Separate events looking at hit ID
    """
    # Load the file
    data = np.loadtxt(filename)
    # Identify the indices where new events start
    event_starts = np.where(data[:, 0] == 0)[0]

    # Build a list of pd DataFrames
    feature_names = ['hitID', 'wireID', 't', 'layer', 'charge0', 'charge1', 'ampl0', 'ampl1', 'x0', 'y0', 'z0', 'theta', 'phi', 'z', 'sigmaz',  'truth', 'nextHit', 'trackID', 'mom', 'trackPhi', 'trackTheta']
    events = [
        pd.DataFrame(
            data[event_starts[i] : (event_starts[i + 1] if i + 1 < len(event_starts) else len(data))],
            columns=feature_names
        )
        for i in range(len(event_starts))
    ]

    return events

def filter_hits(hits, ampl_cut=-1, charge_cut=-1):
    """
    Apply simple cuts to filter noise hits on a DataFrame.

    Parameters:
    hits_df (pd.DataFrame): DataFrame containing hit information.
    ampl_cut (float): Minimum amplitude threshold.
    charge_cut (float): Minimum charge threshold.

    Returns:
    pd.DataFrame: Filtered DataFrame containing only hits passing the cuts.
    """
    # Apply filtering conditions
    filter_condition = (
        (hits['ampl0'] >= ampl_cut) | 
        (hits['charge0'] >= charge_cut)
    )
    
    # Return the filtered DataFrame
    return hits[filter_condition]

def split_cdch_sectors(hits, list_cdch_sectors=[[11, 0, 1], [2, 3, 4], [5, 6, 7]]):
    """
    Divide hits into a list of hits belonging to segments of the detector, identified by CDCH sectors.

    Parameters:
    hits_df (pd.DataFrame): DataFrame containing hit information, including 'wireID'.
    list_cdch_sectors (list of lists): Groups of sectors to filter hits by.

    Returns:
    list of pd.DataFrame: List of DataFrames, each containing hits belonging to a group of CDCH sectors.
    """
    hits_splitted = [
        hits[hits['wireID'].mod(192).floordiv(16).isin(cdch_sectors)]
        for cdch_sectors in list_cdch_sectors
    ]

    return hits_splitted

def build_edges_alternate_layers(hits, distance_same_layer=2):
    """
    We select all edges connecting
    a hit on layer i to a hit on layer i + 1.
    """
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    hits_layers = [hits[hits['wireID'].floordiv(192) == layer] for layer in layers]
    edge_index = []
    edge_attr = []
    
    for i, layer in enumerate(layers):

        # Get hits for the current layer
        hits_layer_i = hits_layers[i]

        # Make edges between hits on the i-th layer
        # up to a distance of distance_same_layer wires
        same_layer_pairs = []
        hitIDs = hits_layer_i['hitID'].values
        wireIDs = hits_layer_i['wireID'].values

        for j, hitID_1 in enumerate(hitIDs):
            for k, hitID_2 in enumerate(hitIDs):
                if k > j and abs(wireIDs[j] - wireIDs[k]) <= distance_same_layer:
                    same_layer_pairs.append((hitID_1, hitID_2))

        # Compute edge attributes (dx, dy, dt) for same-layer pairs
        if same_layer_pairs:
            same_layer_pairs = pd.DataFrame(same_layer_pairs, columns=['hitID_1', 'hitID_2'])
            same_layer_hits_pairs = same_layer_pairs.merge(hits_layer_i[['hitID', 'x0', 'y0', 't']],
                                                           left_on='hitID_1', right_on='hitID')
            same_layer_hits_pairs = same_layer_hits_pairs.merge(hits_layer_i[['hitID', 'x0', 'y0', 't']],
                                                                left_on='hitID_2', right_on='hitID',
                                                                suffixes=('_1', '_2'))

            dx_same = same_layer_hits_pairs['x0_2'] - same_layer_hits_pairs['x0_1']
            dy_same = same_layer_hits_pairs['y0_2'] - same_layer_hits_pairs['y0_1']
            dt_same = same_layer_hits_pairs['t_2'] - same_layer_hits_pairs['t_1']

            edge_index.append(same_layer_pairs.values.T)  # Shape: (2, num_same_layer_edges)
            edge_attr.append(np.stack((dx_same, dy_same, dt_same), axis=-1))  # Shape: (num_same_layer_edges, 3)

        # **Alternate-layer edges**
        
        if layer == 9:
            break  # Stop at the last layer

        # Get hits for the next layer
        hits_layer_i_plus_1 = hits_layers[i + 1]

        if len(hits_layer_i) == 0 or len(hits_layer_i_plus_1) == 0:
            continue  # Skip layers with no hits

        # Get hitIDs of hits in the two layers
        hitID_i = hits_layer_i['hitID'].values
        hitID_i_plus_1 = hits_layer_i_plus_1['hitID'].values

        # Create all possible pairs of hitIDs between the two layers
        pairs = pd.MultiIndex.from_product([hitID_i, hitID_i_plus_1]).to_frame(index=False)
        pairs.columns = ['hitID_1', 'hitID_2']

        # Compute edge attributes (dx, dy, dt)
        hits_pairs = pairs.merge(hits_layer_i[['hitID', 'x0', 'y0', 't']], left_on='hitID_1', right_on='hitID')
        hits_pairs = hits_pairs.merge(hits_layer_i_plus_1[['hitID', 'x0', 'y0', 't']], left_on='hitID_2', right_on='hitID', suffixes=('_1', '_2'))

        dx = hits_pairs['x0_2'] - hits_pairs['x0_1']
        dy = hits_pairs['y0_2'] - hits_pairs['y0_1']
        dt = hits_pairs['t_2'] - hits_pairs['t_1']

        # Append results to the edge list
        edge_index.append(pairs[['hitID_1', 'hitID_2']].values.T)  # Shape: (2, num_edges)
        edge_attr.append(np.stack((dx, dy, dt), axis=-1))  # Shape: (num_edges, 3)

    # Combine edge indices and attributes from all layers
    if len(edge_index) > 0:
        edge_index = np.hstack(edge_index)  # Shape: (2, total_num_edges)
        edge_attr = np.vstack(edge_attr)  # Shape: (total_num_edges, 3)

    return edge_index, edge_attr

def build_event_graphs(hits):
    """
    Build the graph.
    Input:
    param hits: matrix X of shape (n_hits, num_node_features)
    Output:
    graph: list.
    At index 0: matrix of hits features X
    At index 1: matrix of edge features R of shape = (num_edges, num_edge_features)
    At index 2: edge matrix
    """

    graphs = []
    
    # Filter hits first
    hits = filter_hits(hits)
    # Divide into sectors
    hits_sectors = split_cdch_sectors(hits)

    # Loop over sectors and create single graphs
    for sector_hits in hits_sectors:

        # X = (n_hits, n_features)
        feature_names = ['t', 'charge0', 'charge1', 'ampl0', 'ampl1', 'x0', 'y0', 'z', 'phi', 'theta']

        X = sector_hits[feature_names].values.astype(np.float32)  # Convert to NumPy array with float32 dtype
        # Check for zero-size array
        if X.size == 0 or X.shape[1] == 0:
            continue

        # Reset index for hits in a set of sectors to start from 0 in this sub-graph
        sector_hits = sector_hits.reset_index(drop=True)
        # Store old ID
        old_hits_id = sector_hits['hitID'].values
        map_abs_idx_sector_idx = dict(zip(sector_hits['hitID'].values, sector_hits.index))
        # Assign new "local" ID
        sector_hits['hitID'] = sector_hits.index
        
        # edge index and edge attributes
        # (2, n_edges) and (n_edges, n_features)
        edge_index, edge_attr = build_edges_alternate_layers(sector_hits)

        # Check that at least one graph exist
        if len(edge_index) < 1:
            continue

        # Evaluate edges truth label
        edge_truth = np.zeros(len(edge_index.T), dtype=np.float32)

        truth_hits = sector_hits['truth'].values
        nexthit_id = sector_hits['nextHit'].values

        for k, e in enumerate(edge_index.T):
            hit_i = int(e[0])
            hit_j = int(e[1])
            if nexthit_id[hit_i] in old_hits_id and nexthit_id[hit_j] in old_hits_id:
                if truth_hits[hit_i] and truth_hits[hit_j] and ( map_abs_idx_sector_idx[int(nexthit_id[hit_i])]==hit_j or map_abs_idx_sector_idx[int(nexthit_id[hit_j])]==hit_i) :
                    edge_truth[k] = 1

        graph = {'X' : X, 'edge_index' : edge_index, 'edge_attr' : edge_attr, 'truth' : edge_truth}
        graphs.append(graph)

    return graphs

if __name__ == "__main__" :

    PLOT = False
    TIME = True
    RECREATE = True
    
    if (TIME):
        """
        Time these functions
        """
        import time

        filename = "/home/antu/KaleGraph/dataset/1e6TrainSet_CDCH.txt"
        events = load_data(filename)

        # Define the output file
        output_dir = "/home/antu/KaleGraph/graph_files_train_1e6"
        # Should create an entry in a README file to write info about the
        # created graphs...

        # Loop over events
        for ev, hits_event in enumerate(events):

            t_start = time.time()

            graphs = build_event_graphs(hits_event)

            # Loop over sections in an event
            for sec, graph in enumerate(graphs):

                output_filename = os.path.join(output_dir, f"event{ev}_sectors{sec}.npz")

                # Save existing file only if RECREATE is true
                if RECREATE:
                    np.savez(output_filename, X=graph['X'], edge_attr=graph['edge_attr'], edge_index=graph['edge_index'], truth=graph['truth'])
                else:
                    if not os.path.isfile(output_filename):
                        np.savez(output_filename, X=graph['X'], edge_attr=graph['edge_attr'], edge_index=graph['edge_index'], truth=graph['truth'])

                if (PLOT):
                    """
                    Plot a graph
                    """
                    from plot_graph import plot

                
                    plot(graph['X'], graph['edge_index'], graph['truth'])

            t_stop = time.time()
            print(f"Time to make the graphs of an event = {(t_stop - t_start):.3f}.")
            
            
