"""
Functions to build event sub-graphs corresponding to
a set of sectors.
We make use of a pandas DataFrame to store hit and informations
"""
from collections import defaultdict
import os
import time

import numpy as np
import pandas as pd



def load_data(file_id, input_dir="/meg/data1/shared/subprojects/cdch/ext-venturini_a/GNN/NoPileUpMC"):
    """
    Load data for the events within sim file with id file_id
    and transform them into a pandas dataframe.
    Separate events looking at event id.
    """

    print("Loading data...")

    # Load the files
    data_mc = np.loadtxt(f'{input_dir}/{file_id}_MCTruth.txt')
    data_cdch = np.loadtxt(f'{input_dir}/{file_id}_CYLDCHHits.txt')
    data_spx = np.loadtxt(f'{input_dir}/{file_id}_SPXHits.txt')

    # Define features
    features_mc = ['event_id', 'xTGT', 'yTGT', 'zTGT', 'theta', 'phi', 'mom']
    features_cdch = ['event_id', 'wire_id', 'x0', 'y0', 'z0', 'theta', 'phi', 'ztimediff', 'time', 'ampl', 'truth', 'hit_id', 'next_hit_id']
    features_spx = ['event_id', 'pixel_id', 'x0', 'y0', 'z0', 'time', 'truth', 'hit_id', 'next_hit_id']

    # Create DataFrames
    df_mc_full = pd.DataFrame(data_mc, columns=features_mc)
    df_cdch_full = pd.DataFrame(data_cdch, columns=features_cdch)
    df_spx_full = pd.DataFrame(data_spx, columns=features_spx)

    # Group by event_id
    df_mc = df_mc_full.groupby('event_id')
    df_cdch = df_cdch_full.groupby('event_id')
    df_spx = df_spx_full.groupby('event_id')

    # Get the list of unique event IDs from MCTruth (assuming this is the master list)
    event_ids = df_mc_full['event_id'].unique()
    event_ids = event_ids.astype(int)
    event_ids.sort()

    print(f"Found {len(event_ids)} events.")

    # Collect events
    events = []
    for i_ev in event_ids:
        try:
            event_data = [
                df_mc.get_group(i_ev),
                df_cdch.get_group(i_ev),
                df_spx.get_group(i_ev)
            ]
        except KeyError as e:
            #print(f"Warning: Event {i_ev} missing in one of the datasets: {e}")
            continue  # skip this event if incomplete
        events.append(event_data)

    print(f"Loaded {len(events)} complete events.")
    return np.array(events, dtype=object)


def filter_hits(hits, feature, cut_low=-1e30, cut_high=1e30):
    """
    Apply cuts on a feature: cut_low < feature < cut_high to filter hits on a DataFrame.

    Parameters:
    hits_df (pd.DataFrame): DataFrame containing hits information.
    feature (string): feature name on which apply the filter
    cut_low (float): lower cut
    cut_high (float): higher cut

    Returns:
    pd.DataFrame: Filtered DataFrame containing only hits passing the cuts.
    """
    # Apply filtering conditions
    filter_condition = (
        (hits[feature] >= cut_low) & 
        (hits[feature] <= cut_high)
    )
    
    # Return the filtered DataFrame
    return hits[filter_condition]


#[0,1,2,3,4,5,6,7,8,9,10,11],
def split_cdch_sectors(cdch_hits,
                       list_cdch_sectors=[[11, 0, 1, 2, 3, 4, 5, 6, 7],
                                          #[11, 0, 1],
                                          #[1, 2, 3],
                                          #[3, 4, 5],
                                          #[5, 6, 7]
                                          ]):
    """
    Divide cdch_hits into a list of hits belonging to segments of the detector, identified by CDCH sectors.

    Parameters:
    hits_df (pd.DataFrame): DataFrame containing hit information, including 'wireID'.
    list_cdch_sectors (list of lists): Groups of sectors to filter hits by.
    Default list is made dividing sectors in group of 3 with a superposition of 1 sector
    in each group.
    Use [11, 0, 1, 2, 3, 4, 5, 6, 7] for a graph not segmented

    Returns:
    list of pd.DataFrame: List of DataFrames, each containing hits belonging to a group of CDCH sectors.
    """
    hits_splitted = [
        cdch_hits[cdch_hits['wire_id'].mod(192).floordiv(16).isin(cdch_sectors)]
        for cdch_sectors in list_cdch_sectors
    ]
    return hits_splitted

def build_edges_alternate_layers(cdch_hits,  n_successive_layer = 1,distance_same_layer=3 ):
    """
    We select all edges connecting
    a hit on layer i to a hit on layer i + n_successive_layer.
    """
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    hits_layers = [cdch_hits[cdch_hits['wire_id'].floordiv(192) == layer] for layer in layers]
    edge_index = []
    edge_attr = []
    
    for i, layer in enumerate(layers):

        # Get hits for the current layer
        hits_layer_i = hits_layers[i]

        # Make edges between hits on the i-th layer
        # up to a distance of distance_same_layer wires
        same_layer_pairs = []
        hitIDs = hits_layer_i['hit_id'].values
        wireIDs = hits_layer_i['wire_id'].values

        for j, hitID_1 in enumerate(hitIDs):
            for k, hitID_2 in enumerate(hitIDs):
                if k > j and abs(wireIDs[j] - wireIDs[k]) <= distance_same_layer:
                    same_layer_pairs.append((hitID_1, hitID_2))
        
        # Compute edge attributes (dx, dy, dt) for same-layer pairs
        if same_layer_pairs:
            same_layer_pairs = pd.DataFrame(same_layer_pairs, columns=['hit_id_a', 'hit_id_b'])
            same_layer_hits_pairs = same_layer_pairs.merge(hits_layer_i[['hit_id', 'x0', 'y0', 'time']],
                                                           left_on='hit_id_a', right_on='hit_id')
            same_layer_hits_pairs = same_layer_hits_pairs.merge(hits_layer_i[['hit_id', 'x0', 'y0', 'time']],
                                                                left_on='hit_id_b', right_on='hit_id',
                                                                suffixes=('_1', '_2'))

            dx_same = same_layer_hits_pairs['x0_2'] - same_layer_hits_pairs['x0_1']
            dy_same = same_layer_hits_pairs['y0_2'] - same_layer_hits_pairs['y0_1']
            dt_same = same_layer_hits_pairs['time_2'] - same_layer_hits_pairs['time_1']

            edge_index.append(same_layer_pairs.values.T)  # Shape: (2, num_same_layer_edges)
            edge_attr.append(np.stack((dx_same, dy_same, dt_same), axis=-1))  # Shape: (num_same_layer_edges, 3)

        # **Alternate-layer edges**
        
        if layer == 9:
            break  # Stop at the last layer
        for n in range(1,n_successive_layer+1):
            if(layer+n > 9):
                break
         
            # Get hits for the next layer
            hits_layer_i_plus_1 = hits_layers[i + n]

            if len(hits_layer_i) == 0 or len(hits_layer_i_plus_1) == 0:
                continue  # Skip layers with no hits

            # Get hit_id of hits in the two layers
            hitID_i = hits_layer_i['hit_id'].values
            hitID_i_plus_1 = hits_layer_i_plus_1['hit_id'].values
            # Create all possible pairs of hitIDs between the two layers
            pairs = pd.MultiIndex.from_product([hitID_i, hitID_i_plus_1]).to_frame(index=False)
            pairs.columns = ['hit_id_a', 'hit_id_b']

            # Compute edge attributes (dx, dy, dt)
            hits_pairs = pairs.merge(hits_layer_i[['hit_id', 'x0', 'y0', 'time']], left_on='hit_id_a', right_on='hit_id')
            hits_pairs = hits_pairs.merge(hits_layer_i_plus_1[['hit_id', 'x0', 'y0', 'time']], left_on='hit_id_b', right_on='hit_id', suffixes=('_1', '_2'))
    
            dx = hits_pairs['x0_2'] - hits_pairs['x0_1']
            dy = hits_pairs['y0_2'] - hits_pairs['y0_1']
            dt = hits_pairs['time_2'] - hits_pairs['time_1']

            # Append results to the edge list
            edge_index.append(pairs[['hit_id_a', 'hit_id_b']].values.T)  # Shape: (2, num_edges)
            edge_attr.append(np.stack((dx, dy, dt), axis=-1))  # Shape: (num_edges, 3)
            

    # Combine edge indices and attributes from all layers
    if len(edge_index) > 0:
        edge_index = np.hstack(edge_index)  # Shape: (2, total_num_edges)
        edge_attr = np.vstack(edge_attr)  # Shape: (total_num_edges, 3)

    return edge_index, edge_attr


def build_graph_spx(SPX_hits, index_start_at=0):
    # X = (n_hits, n_features)
    #feature_names = ['x0', 'y0', 'z0', 'time', 'ampl']
    # WARNING! Number of features of CDCH and SPX hit must be the same

    #print("Building SPX Graph")
    feature_names = ['x0', 'y0', 'z0', 'time', 'ampl', 'isSPX']
    # Add a place holder for ampl. Set to 1. This is not a problem for SPX hits,
    # which have low noise compared to CDCH.
    # Add also a flag to distinguish between CDCH and SPX
    SPX_hits['ampl'] = np.float32(1.)
    SPX_hits['isSPX'] = np.float32(1.)
    SPX_hits = SPX_hits.reset_index(drop = True)    
    edge_index = []
    edge_attr = []
    truth_hits = SPX_hits['truth'].values
    nexthit_id = SPX_hits['next_hit_id'].values


    couple_pixels = []

    
    
    
    map_idx = dict(zip(SPX_hits['hit_id'].values, SPX_hits.index))
    
    #Put this defaultdict: this is to avoid crash as the last hit which has nexthit = -1, also for noise
    map_idx = defaultdict(lambda:-100, map_idx)

    # Assign new "local" ID
    SPX_hits['hit_id'] = SPX_hits.index + index_start_at
    #save hits info
    X = SPX_hits[feature_names].values.astype(np.float32)
        

    
    #revert back to perform connections with map 
    SPX_hits['hit_id'] = SPX_hits.index 
    if(X.size == 0 or X.shape[1] == 0):
        return -1
    hitIDs = SPX_hits['hit_id']  



    #now we have to create all possible connections (we have few hits in the SPX hits)
    for j, hitID_1 in enumerate(hitIDs):
        for k, hitID_2 in enumerate(hitIDs): 
            if j <k: 
                couple_pixels.append((hitID_1, hitID_2))
    
    if couple_pixels:
        couple_pixels = pd.DataFrame(couple_pixels, columns=['hit_id_a', 'hit_id_b'])
        couple__hits_pixels = couple_pixels.merge(SPX_hits[['hit_id', 'x0', 'y0', 'time']],
                                                           left_on='hit_id_a', right_on='hit_id')

        couple__hits_pixels = couple__hits_pixels.merge(SPX_hits[['hit_id', 'x0', 'y0', 'time']],
                                                                left_on='hit_id_b', right_on='hit_id',
                                                                suffixes=('_1', '_2'))

        dx_same = couple__hits_pixels['x0_2'] - couple__hits_pixels['x0_1']
        dy_same = couple__hits_pixels['y0_2'] - couple__hits_pixels['y0_1']
        dt_same = couple__hits_pixels['time_2'] - couple__hits_pixels['time_1']
        # Attenzione! non è così che funziona la rete: la rete non usa la feature 'hit id'
        # per individuare l'hit corretto, ma bensì il suo indice all'interno del vettore X.
        # E' per questo che si fa il comando reset_index
        
        edge_index.append(couple__hits_pixels[['hit_id_a', 'hit_id_b']].values.T)  # Shape: (2, num_same_layer_edges)
        edge_attr.append(np.stack((dx_same, dy_same, dt_same), axis=-1))  # Shape: (num_same_layer_edges, 3)

    if len(edge_index) > 0:
        edge_index = np.hstack(edge_index)  # Shape: (2, total_num_edges)
        edge_attr = np.vstack(edge_attr)  # Shape: (total_num_edges, 3)
    else:
        return X, np.empty((2,0)),np.empty((0,3)),np.array([])
    # Evaluate edges truth label
    edge_truth = np.zeros(len(edge_index.T), dtype=np.float32)
    
    for k, e in enumerate(edge_index.T):
            hit_i = int(e[0])
            hit_j = int(e[1])
            if(truth_hits[hit_i] > 0 and truth_hits[hit_j] > 0 and (map_idx[nexthit_id[int(hit_i)]]==hit_j or map_idx[nexthit_id[int(hit_j)]]==hit_i)) :
                edge_truth[k] = truth_hits[hit_i]
    #correct for starting index.
    edge_index[0] +=index_start_at
    edge_index[1] +=index_start_at
    
    return X, edge_index, edge_attr, edge_truth


def build_graph_cdch(hits_cdch, sector_hits, depth_conn_cdch, same_layer_cdch_dist_conn):
    # X = (n_hits, n_features)
    #feature_names = ['x0', 'y0', 'ztimediff', 'time', 'ampl']
    # Warning! Number of features of CDCH and SPX hit must be the same
    

    sector_hits_placeholder = sector_hits
    # Store old ID
    old_hits_id = sector_hits_placeholder['hit_id'].values
    # Reset index for hits in a set of sectors to start from 0 in this sub-graph
    sector_hits_placeholder = sector_hits_placeholder.reset_index(drop=True)

    # X non ha bisogno della variabile hit_id.
    # hit_id equivale dopo il reset_index al numero di riga di ogni entry.
    # Questo è quanto basta per creare il grafo
    # Aggiungi una flag per tenere conto se la hit appartiene all'SPX o alla CDCH
    feature_names = ['x0', 'y0', 'ztimediff', 'time', 'ampl', 'isSPX']
    sector_hits_placeholder['isSPX'] = np.float32(0.)
    X = sector_hits_placeholder[feature_names]
    

    X = X.values.astype(np.float32)# Convert to NumPy array with float32 dtype    
    # Check for zero-size array
    if X.size == 0 or X.shape[1] == 0:
        return [],[],[],[]
        
    
    #print(f"\t\t\tNumber of hits in this CDCH subgraph ", X.shape[0])

    map_abs_idx_sector_idx = dict(zip(sector_hits_placeholder['hit_id'].values, sector_hits_placeholder.index))
    
    #Put this defaultdict: this is to avoid crash as the last hit which has nexthit = -1, also for noise
    map_abs_idx_sector_idx = defaultdict(lambda:-100, map_abs_idx_sector_idx)
    
    # Assign new "local" ID
    sector_hits_placeholder['hit_id'] = sector_hits_placeholder.index

    # edge index and edge attributes
    # (2, n_edges) and (n_edges, n_features)
    edge_index, edge_attr = build_edges_alternate_layers(sector_hits_placeholder, depth_conn_cdch, same_layer_cdch_dist_conn)

    # Check that at least one graph exist
    if len(edge_index) < 1:
        return [],[],[],[]
    
    # Evaluate edges truth label 
    edge_truth = np.zeros(len(edge_index.T), dtype=np.float32)
    truth_hits = sector_hits_placeholder['truth'].values
    nexthit_id = sector_hits_placeholder['next_hit_id'].values
    
    for k, e in enumerate(edge_index.T):
        hit_i = int(e[0])
        hit_j = int(e[1])
        if nexthit_id[hit_i] in old_hits_id and nexthit_id[hit_j] in old_hits_id:
            # La seconda condizione assicura la "direzionalità" del grafo
            # Se manca una hit però questo porta ad un grafo disconnesso, invece potrebbe essere utile avere quella connessione
            if truth_hits[hit_i] > 0 and truth_hits[hit_j] > 0 and ( map_abs_idx_sector_idx[int(nexthit_id[hit_i])]==hit_j or map_abs_idx_sector_idx[int(nexthit_id[hit_j])]==hit_i) :
                edge_truth[k] = truth_hits[hit_i]
    

    return X, edge_index, edge_attr, edge_truth           

def create_connection_between_cdchlayer_spx(hits_spx, hits_cdch, last_CDCH_layer_to_connect_spx = 1 ):
    #print("\t\t\t SPX hits = ", hits_spx)
    #print("\t\t\t CDCH hits = ", hits_cdch)
    edge_index = []
    edge_attr = []
    edge_truth = []

    
    
    truth_cdch = hits_cdch['truth'].values
    truth_spx = hits_spx['truth'].values
    # Store old ID
    hits_cdch_placeholder = hits_cdch.copy()
    old_hits_id = hits_cdch_placeholder['hit_id'].values
    hits_cdch_placeholder = hits_cdch_placeholder.reset_index(drop=True)
    hits_cdch_placeholder['hit_id'] = hits_cdch_placeholder.index
    


    hitIDs_cdch = hits_cdch_placeholder['hit_id'].values
    hitIDs_SPX = hits_spx['hit_id'].values
 
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    hits_layers = [hits_cdch_placeholder[hits_cdch_placeholder['wire_id'].floordiv(192) == layer] for layer in layers]
    first_non_empty_layer_index = next(index for index,layer in enumerate(hits_layers) if len(layer)!=0)
    #take beginning from the first non empty layer the next n layers, even if they are empty.
    if hits_layers:
        last_n_non_empty_cdch_layers = pd.concat([h for h in hits_layers[first_non_empty_layer_index:first_non_empty_layer_index + last_CDCH_layer_to_connect_spx]],ignore_index=True)

    hitIDs_cdch_last = last_n_non_empty_cdch_layers['hit_id'].values
    #create all possible couples between cdch layer and spx
    couple_cdchhit_spxhit = []
    for j, hitID_1 in enumerate(hitIDs_cdch_last):
        for k, hitID_2 in enumerate(hitIDs_SPX): 
                couple_cdchhit_spxhit.append((hitID_1, hitID_2))
    
    if(couple_cdchhit_spxhit):
    
        couple_cdchhit_spxhit = pd.DataFrame(couple_cdchhit_spxhit, columns=['hit_id_a', 'hit_id_b'])
        couple_cdchhit_spxhit = couple_cdchhit_spxhit.merge(last_n_non_empty_cdch_layers[['hit_id', 'x0', 'y0', 'time']],
                                                           left_on='hit_id_a', right_on='hit_id')

        couple_cdchhit_spxhit = couple_cdchhit_spxhit.merge(hits_spx[['hit_id', 'x0', 'y0', 'time']],
                                                                left_on='hit_id_b', right_on='hit_id',
                                                                suffixes=('_1', '_2'))

        dx_same = couple_cdchhit_spxhit['x0_2'] - couple_cdchhit_spxhit['x0_1']
        dy_same = couple_cdchhit_spxhit['y0_2'] - couple_cdchhit_spxhit['y0_1']
        dt_same = couple_cdchhit_spxhit['time_2'] - couple_cdchhit_spxhit['time_1']

        edge_index.append(couple_cdchhit_spxhit[['hit_id_a', 'hit_id_b']].values.T)  # Shape: (2, num_same_layer_edges)
        edge_attr.append(np.stack((dx_same, dy_same, dt_same), axis=-1))  # Shape: (num_same_layer_edges, 3)
    
    if len(edge_index) > 0:
        edge_index = np.hstack(edge_index)  # Shape: (2, total_num_edges)
        edge_attr = np.vstack(edge_attr)  # Shape: (total_num_edges, 3)
    else:
        return [],[],[]
    
    # The truth should be set to nturn and should not be the same for all CDCH and SPX connections
    edge_truth = np.zeros(len(edge_index.T), dtype=np.float32)

    for k, e in enumerate(edge_index.T):
        # Retrieve truth for hits in CDCH and SPX
        nturns_CDCH = 0
        nturns_SPX = 0
        i = int(e[0])
        j = int(e[1])
        #print(f"Connecting hits {i} and {j}.")

        
        if i < len(hitIDs_cdch):
            nturns_CDCH = truth_cdch[i]
        else:
            nturns_SPX = truth_spx[i - len(hitIDs_cdch)]

        if j < len(hitIDs_cdch):
            nturns_CDCH = truth_cdch[j]
        else:
            nturns_SPX = truth_spx[j - len(hitIDs_cdch)]

        # Set truth
        if nturns_CDCH == nturns_SPX and nturns_CDCH > 0:
            edge_truth[k] = nturns_SPX
        else:
            edge_truth[k] = 0

    return edge_index, edge_attr, edge_truth


def build_CDCH_graphs(hits_cdch, hits_spx, depth_conn_cdch, depth_conn_cdch_spx, same_layer_cdch_dist_conn):                  
    #print("\t\tBuilding the CDCH graphs")
    hits_sectors = split_cdch_sectors(hits_cdch)   
    X_sectors_CDCH = []
    edge_index_sectors_CDCH = []
    edge_attr_sectors_CDCH = []
    edge_truth_sectors_CDCH = []

    # Loop over sectors and create single graphs
    for i,sector_hits in enumerate(hits_sectors):
        #print(f"\t\tCDCH graph in Sector group {i}")
        #build connections inside the cdch
        X_cdch, edge_index_cdch,edge_attr_cdch, edge_truth_cdch = build_graph_cdch(hits_cdch, sector_hits, depth_conn_cdch, same_layer_cdch_dist_conn)
        #print(f"\t\tNumber of cdch hits in this graph = {len(X_cdch)}")

        #if there are no hits in CDCH, skip sector
        if(len(edge_index_cdch) == 0):
            continue;


        #build connection between cdch last layers (to be tuned) and spx.
        
        N_hits_CDCH = len(X_cdch)

        hits_spx_subgraph = hits_spx.copy()


        #shift spx indexes by N_hits_CDCH
        hits_spx_subgraph['hit_id'] = hits_spx.index+N_hits_CDCH

        
        
        #create connections between cdch and spx
        edge_index_cdch_spx,edge_attr_cdch_spx, edge_truth_cdch_spx = create_connection_between_cdchlayer_spx(hits_spx_subgraph, sector_hits, depth_conn_cdch_spx)

        #if there are no connections between spx and cdch, don't append result
        if len(edge_index_cdch_spx)!= 0:
            edge_index_cdch= np.concatenate((edge_index_cdch,edge_index_cdch_spx), axis =1)
            edge_attr_cdch = np.concatenate((edge_attr_cdch,edge_attr_cdch_spx ))
            edge_truth_cdch= np.concatenate((edge_truth_cdch, edge_truth_cdch_spx))
        
        X_sectors_CDCH.append(X_cdch)
        edge_index_sectors_CDCH.append(edge_index_cdch)
        edge_attr_sectors_CDCH.append(edge_attr_cdch)
        edge_truth_sectors_CDCH.append(edge_truth_cdch)    

    return X_sectors_CDCH, edge_index_sectors_CDCH, edge_attr_sectors_CDCH, edge_truth_sectors_CDCH
    
def build_event_graphs(hits_cdch, hits_spx, tune_cdch_connection_depth, same_layer_cdch_dist_conn, tune_cdch_and_spx_last_layers_connection_depth, normalize=True):
    """
    Build final graph from CDCH and SPX.
    Input:
    param hits: matrix X of shape (n_hits, num_node_features)
    Output:
    graph: list.
    At index 0: matrix of hits features X
    At index 1: matrix of edge features R of shape = (num_edges, num_edge_features)
    At index 2: edge matrix
    """
    
    #add N_hits_CDCH to hitids in spx
    graphs = []
    #
    hits_spx = hits_spx.reset_index(drop = True)
    
    #build both spx and cdch connections
    #print("\tBuild CDCH and CDCH - SPX graphs...")
    Vec_X_sector_CDCH, Vec_edge_index_CDCH, Vec_edge_attr_CDCH, Vec_edge_truth_CDCH = build_CDCH_graphs(hits_cdch,
                                                                                                        hits_spx,tune_cdch_connection_depth,
                                                                                                        tune_cdch_and_spx_last_layers_connection_depth,
                                                                                                        same_layer_cdch_dist_conn ) 

    for i, X_cdch in enumerate(Vec_X_sector_CDCH):
        
        N_hits_CDCH = len(X_cdch)
        hits_spx_subgraph = hits_spx.copy()
        hits_spx_subgraph['hit_id']  += N_hits_CDCH
        hits_spx_subgraph['next_hit_id'] += N_hits_CDCH
        
        # Build spx graph for this subgraph
        X_spx, edge_index_spx, edge_attr_spx , edge_truth_spx= build_graph_spx(hits_spx_subgraph, index_start_at=N_hits_CDCH)
        if(len(X_spx) != 0):
            X = np.concatenate((X_cdch, X_spx))
            edge_index =np.concatenate((Vec_edge_index_CDCH[i],edge_index_spx), axis = 1)
            edge_attr =np.concatenate((Vec_edge_attr_CDCH[i],edge_attr_spx))
            edge_truth = np.hstack((Vec_edge_truth_CDCH[i],edge_truth_spx))
        else:
            X = X_cdch
            edge_index = Vec_edge_index_CDCH[i]
            edge_attr = Vec_edge_attr_CDCH[i]
            edge_truth = Vec_edge_truth_CDCH[i]
          
        
        graph = {'X' : X, 'edge_index' : edge_index, 'edge_attr' : edge_attr, 'truth' : edge_truth}
        graphs.append(graph)

    return graphs

def build_dataset(file_ids,
                  input_dir="/meg/data1/shared/subprojects/cdch/ext-venturini_a/GNN/NoPileUpMC",
                  output_dir="./",
                  time_it=False,
                  plot_it=False):
    """
    Build all graphs in file_ids and save them to *.npz files.
    Only for cdch events at the moment.
    file_ids (array of string): numbers of the files with events to use for the graph creation, provided as string.
    """

    for file_id in file_ids:
        
        if time_it:
            t_start = time.time()

        # Loop over events
        events = load_data(file_id, input_dir=input_dir)

        for ev, event in enumerate(events):
            if(ev >=0):
                #if(ev > 72):
                #    break;
                mc_truth = event[0]
                cdch_event = event[1]
                spx_event = event[2]

                #print(f"Building graphs of event {ev}...")

                # Features are:
                # 1) cdch event, those are cdch hits
                # 2) spx event, those are spx hits
                # 3) depth_conn_cdch, this is to be tuned: sets the depth of connections between cdch layers in terms of cdch layers.  
                # 4) same_layer_cdch_dist_conn, this is to be tuned, set how far apart a connection is made ebtween wire of the same layer
                # 5) depth_conn_cdch_spx, this is to be tuned: sets how deep in the cdch connection are made with spx hits: ex 1 means only closest layer.
                #print("Starting to create graph for event ", ev)
                
                graphs = build_event_graphs(cdch_event, spx_event, 4, 3, 2)
    
                # Loop over sections in an event
                for sec, graph in enumerate(graphs):
    
                    output_filename = os.path.join(output_dir, f"{output_dir}/noNoisefile{file_id}_event{ev}_sectors{sec}.npz")
    
                    np.savez(output_filename, X=graph['X'], edge_attr=graph['edge_attr'], edge_index=graph['edge_index'], truth=graph['truth'])
    
                    if plot_it:
                        """
                        Plot a graph
                        """
                        from utils.plot_graph import plot
    
                        plot(graph['X'], graph['edge_index'], graph['truth'])
    
        if time_it:
            t_stop = time.time()
            print(f"{(t_stop - t_start) / len(events) :.3f} s per event to build {len(events)} events.")

    print(f"Dataset *.npz files created in {output_dir}")


if __name__ == "__main__" :

    import sys

    PLOT = False
    TIME = True
    input_dir = "/meg/data1/shared/subprojects/cdch/ext-venturini_a/GNN/NoPileUpMC"
    output_dir = input_dir
    #file_ids = [f'0{int(sys.argv[1])}']
    #file_ids = [f'0{int(idx)}' for idx in range(1001, 1010, 1)]
    file_ids = [f'0{int(sys.argv[1])}']
    build_dataset(file_ids, input_dir=input_dir, output_dir=input_dir, time_it=TIME, plot_it=PLOT)
