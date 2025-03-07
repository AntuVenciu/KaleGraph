"""
Here goes some utils.
"""
import numpy as np


def load_graph_npz(file_path):
    """
    Loads a .npz file containing graph data and returns its contents as a dictionary.

    Parameters:
        file_path (str): Path to the .npz file.

    Returns:
        dict: A dictionary with keys ['x', 'edge_attr', 'edge_index', 'y', 'pid', 'pt', 'eta'].
              Each key maps to its corresponding numpy array.
    """
    # Load the .npz file
    with np.load(file_path) as data:
        # Extract elements
        graph_data = {
            'X': data['X'],               # Node features
            'edge_attr': data['edge_attr'],  # Edge attributes
            'edge_index': data['edge_index'],  # Edge connectivity
            'truth': data['truth'],               # Labels for edges
        }
    
    return graph_data

def load_graph_npz_as_variables(file_path):
    """
    Loads a .npz file containing graph data and returns its contents as separate variables.

    Parameters:
        file_path (str): Path to the .npz file.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Node features
            - edge_attr (numpy.ndarray): Edge attributes
            - edge_index (numpy.ndarray): Edge connectivity
            - y (numpy.ndarray): Labels for edges
            - pid (numpy.ndarray): Particle ID mapping
            - pt (numpy.ndarray): Transverse momentum
            - eta (numpy.ndarray): Pseudorapidity
    """
    # Load the .npz file
    with np.load(file_path) as data:
        X = data['X']               # Node features
        edge_attr = data['edge_attr']  # Edge attributes
        edge_index = data['edge_index']  # Edge connectivity
        truth = data['truth']               # Labels for edges
    
    return X, edge_attr, edge_index, truth
