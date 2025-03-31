
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab10
from utils.tools import load_graph_npz
from utils.plot_graph import plot
import pandas as pd
import sys
from models.interaction_network import InteractionNetwork
from utils.dataset import GraphDataset

def plot_example(filename):
    graph = np.load(filename)
    
    print(graph['X'].shape)
    print(graph['edge_index'].shape)
    edge_index =graph['edge_index']

    startindex = np.min([np.min(edge_index[0]), np.min(edge_index[1])]) 
    
    edge_index[0] += -startindex
    edge_index[1] += -startindex

    print(graph['truth'].shape)
    print(graph['predicted'].shape)

    plot(graph['X'],edge_index, graph['truth'])
    
    plot(graph['X'],edge_index, graph['predicted'])
    
    
    
    
    
if __name__ == "__main__":
    """
    Test plot function
    """
   

    """
    Test plotting a graph from npz files
    """
    
    
    
    
    
    
    filename = fz"DataTruthPredicted/file01030_event{int(sys.argv[1])}_sectors0.npz_test_pred_truth.npz"
    
    
    plot_example(filename)
    
