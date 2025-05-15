
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab10
from utils.tools import load_graph_npz


"""
Careful! Import plot either for utils.plot_graph or from utils.plot_graph_node_classification depending on task, this is temporary, will be solved.


"""

#from utils.plot_graph import plot
from utils.plot_graph_node_classification import plot
import pandas as pd
import sys
from models.interaction_network import InteractionNetwork
from utils.dataset import GraphDataset

def plot_example(filename):
    graph = np.load(filename)
    
   
    edge_index =graph['edge_index']

    startindex = np.min([np.min(edge_index[0]), np.min(edge_index[1])]) 
    
    edge_index[0] += -startindex
    edge_index[1] += -startindex


    plot(graph['X'],edge_index, graph['truth'])
    
    plot(graph['X'],edge_index, graph['predicted'])
    
    
    
    
    
if __name__ == "__main__":
    """
    Test plot function
    """
   

    """
    Test plotting a graph from npz files
    """
    
    
    
        
    filename = f"TruthWithNoise/{int(sys.argv[1])}_test_pred_truth.npz"
    
    
    #filename = f"TruthComparisonOldModel/{int(sys.argv[1])}_test_pred_truth.npz"
    
    
    plot_example(filename)
    
