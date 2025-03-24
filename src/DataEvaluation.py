
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab10

import pandas as pd
import sys
import torch
from models.interaction_network import InteractionNetwork
from utils.dataset import GraphDataset
from utils.tools import load_graph_npz
from utils.plot_graph import plot
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
import seaborn as sns
import joblib #we use this to load the scaler.
import glob
from collections import OrderedDict
def load_model(ModelState):





    #we need to remake the dict: there is a suffix in the model data.    
    new_state_dict = OrderedDict()
    for k, v in ModelState['model_state_dict'].items():
        new_key = k.replace("_orig_mod.", "")  
        new_state_dict[new_key] = v

    hyper_params = ModelState['hyper_params']
    hiddens_size = hyper_params['hidden_size']
    num_node_attr = hyper_params['node_features']
    num_edge_attr =hyper_params['edge_features']
    time_steps =hyper_params['time_steps']
    model = InteractionNetwork(hiddens_size, num_node_attr,num_edge_attr,time_steps)
    
    

    model.load_state_dict(new_state_dict)
    optimizer = optim.Adam(model.parameters())
    #i am not sure about this: one has to check if there is more stuff to set. Maybe setup values for all?
    scheduler = StepLR(optimizer, step_size = 1, gamma = 1)

    
    optimizer.load_state_dict(ModelState['optimizer_state_dict'])
    scheduler.load_state_dict(ModelState['scheduler_state_dict'])



    # 5. Metti il modello in modalità evaluation
    model.eval()
    return optimizer, scheduler,model
    
    
    
def evaluate_data(model, test_loader, scalers, name):
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)

            output = model(data.x,
                           data.edge_index,
                           data.edge_attr)
            y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32).to(device)
            y = y.to(torch.long)


            # denormalize output
            X_denormalized = torch.tensor(scalers['X'].inverse_transform(data.x), dtype=torch.float32).numpy()
            edge_attr_denormalized = torch.tensor(scalers['edge_attr'].inverse_transform(data.edge_attr), dtype=torch.float32).numpy()
            print(batch_idx)
            output_file = f'DataTruthPredicted/{name}_test_pred_truth.npz'
            np.savez(output_file, 
                         X=X_denormalized,
                         edge_attr=edge_attr_denormalized,
                         edge_index=data.edge_index.numpy(),
                         truth=y.cpu().numpy(),
                         predicted=torch.argmax(output.cpu(), dim =1).numpy())
                         
                         
def PlotMyModelResults(ModelState):                     
    train_loss = ModelState['train_loss'].numpy()
    val_loss = ModelState['val_loss'].numpy()
    test_loss = ModelState['test_loss'].numpy()
    Val_confusion_mat = ModelState['Val_Conf_matrix'].numpy()  
    Test_confusion_mat = ModelState['Test_Conf_matrix'].numpy()      
    
    plt.title("KaleGraph Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(np.linspace(1, len(train_loss), len(train_loss)), train_loss, label='Training', color='blue')
    plt.plot(np.linspace(1, len(test_loss), len(test_loss)), test_loss, label='Test', color='orange')
    plt.plot(np.linspace(1, len(val_loss), len(val_loss)), val_loss, label='Validation', color='red')

    plt.legend()
    plt.show()

    
    labels = ['Noise', 'Turn 1', 'Turn 2', 'Turn 3', 'Turn 4', 'Turn 5', 'Turn 6']
    
    #plot validation confusion matrix
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Validation Confusion Matrix")
    
    sns.heatmap(Val_confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.show()
    
    
    #plot test confusion matrix
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Test Confusion Matrix")
    
    sns.heatmap(Test_confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.show()
    
if __name__ == "__main__":
    """
        This script takes a .npz data and prints out a npz file with both predicted and truth index values.
        The npz file might be later be plotted with CompareTestAndTruth.py
    """
    
    
    PlotResults = False
    
    torch.set_float32_matmul_precision('high')
    
    #pass your model and scaler
    filepath_model = "ModelsMade/MessagePassSteps2/model1.pth"
    file_path_scaler = "ModelsMade/MessagePassSteps2/scaler1.pkl"
    
    
    
    ModelState = torch.load(filepath_model)
    optimizer, scheduler, myModel =  load_model(ModelState)
    MyScaler = joblib.load(file_path_scaler)
    #retrieve other information
    myModel = torch.compile(myModel)
    myModel.eval()
    
    
    
    if PlotResults:
        PlotMyModelResults(ModelState)
    
    name = "file01030_event1_sectors0.npz"
    
    test_file_name = "DataForTesting/" + name
    #graphTest = np.load(test_file_name)
    graph_files = glob.glob(test_file_name)
    
    test_set = GraphDataset(graph_files, scalers=MyScaler, fitted=True)

    use_cuda = False
    #now we have our graph: let us evaluate this graph.
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    myModel = myModel.to(device)
    params = {'batch_size':1 , 'shuffle' : False, 'num_workers' : 4}
    test_loader = DataLoader(test_set, **params)
    
    
    evaluate_data(myModel, test_loader, MyScaler, name)
    
    
