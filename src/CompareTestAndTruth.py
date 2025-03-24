
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab10
from utils.tools import load_graph_npz
from utils.plot_graph import plot
import pandas as pd



    
if __name__ == "__main__":
    """
    Test plot function
    """
    #import build_graph as bg

    #hitIDs = [i for i in range(0, 1920 + 512) if np.random.uniform() > 0.9]
    #edges['edge_index'] = bg.build_graph(hitIDs)
    #plot(edges)

    """
    Test plotting a graph from npz files
    """

    filename = "DataTruthPredicted/3_test_pred_truth.npz"
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
