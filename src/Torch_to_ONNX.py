
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab10

import pandas as pd
import sys
import torch
import onnx
import onnxruntime as ort
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
import torch.nn as nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class ModelWithScaler(nn.Module):
    def __init__(self, scaler, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean_X", torch.tensor(scaler['X'].mean_, dtype=torch.float32))
        self.register_buffer("scale_X", torch.tensor(scaler['X'].scale_, dtype=torch.float32))
        self.register_buffer("mean_edge_attr", torch.tensor(scaler['edge_attr'].mean_, dtype=torch.float32))
        self.register_buffer("scale_edge_attr", torch.tensor(scaler['edge_attr'].scale_, dtype=torch.float32))
        
        
    def forward(self, X,edge_index,edge_attr):
        #scale our variables
        X = (X - self.mean_X) / self.scale_X
        edge_attr = (edge_attr - self.mean_edge_attr) / self.scale_edge_attr
        edge_index = edge_index.to(torch.int64)
        
        
        return self.model(X,edge_index,edge_attr)





def Import_to_ONNX(model, scaler):
    MyModel = ModelWithScaler(scaler,model)
    
    #now import scaled model to onnx.
    N, E = 5, 10
    x = torch.randn(N, 8)
    edge_index = torch.rand(2, E)
    edge_attr = torch.randn(E, 3)
    #torch.onnx_export()
    
    torch.onnx.export(
    MyModel,
    (x, edge_index, edge_attr),
    "gnn_mlp_message_with_scaler.onnx",
    input_names=["x", "edge_index", "edge_attr"],
    output_names=["y"],
    dynamic_axes={
        "x": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "edge_attr": {0: "num_edges"},
        "y": {0: "num_nodes"}
    },
    opset_version=17
    )





def UnPackModelAndScaler(filepath_model,file_path_scaler):

    #change path if using different architectures.

    from DataEvaluation import load_model
    
    ModelState = torch.load(filepath_model, map_location=torch.device(device))
    optimizer, scheduler, myModel =  load_model(ModelState)

    MyScaler = joblib.load(file_path_scaler)
    
    #retrieve other information
    
    return myModel, MyScaler


def TestONNXModelCreation(MyModel,MyScaler):
    onnx_model = onnx.load("gnn_mlp_message_with_scaler.onnx")
    onnx.checker.check_model(onnx_model)
    print("Il modello ONNX è valido 🎉")

   
    filename = "250kDatasetModel250Nodes/node_class_withNoiseMCfilewithNoise_01030_event999_sectors0.npz_test_pred_truth.npz"
    graph = np.load(filename)
    
    graph_files = glob.glob(filename)
    #scale dataset
    

    MyModel = ModelWithScaler(scaler,model)
    test_set = GraphDataset(graph_files)
    
    
    
    data = test_set.get(0)
    #the getter automatically scales the dataset.

    inputs = {
    'x': data.x.numpy().astype(np.float32),
    'edge_index': data.edge_index.numpy().astype(np.float32),
    'edge_attr': data.edge_attr.numpy().astype(np.float32)
    }
    
    np.savetxt("ONNXProva/x.txt",data.x, fmt='%.14f')
    np.savetxt("ONNXProva/e_i.txt",data.edge_index, fmt='%.1f')
    np.savetxt("ONNXProva/e_a.txt",data.edge_attr, fmt='%.14f',)

    
    session = ort.InferenceSession("gnn_mlp_message_with_scaler.onnx")
    
    
    
    outputs = session.run(None, inputs)
    y_torch = 0;
    with torch.no_grad():
        output = MyModel(data.x.to(torch.float32),
                           data.edge_index.to(torch.float32),
                           data.edge_attr.to(torch.float32))
        y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32).to(device)
        
        y_torch = torch.argmax(output.cpu(), dim =1).to(torch.long)

    #print(outputs)
    
    y = np.argmax(outputs[0], axis =1)
    print(y)
    
    print(y_torch.numpy())
    print(graph['truth'])
    print(graph['predicted'])
    #print(graph['predicted'])
    from utils.plot_graph_node_classification import plot

    #plot(graph['X'],graph['edge_index'], graph['truth'])
    #plot(graph['X'],graph['edge_index'], y)

if __name__ == '__main__':
    
    #pass your model and scaler
    filepath_model = "250kDatasetModel250Nodes/model_250k_dataset.pth"
    file_path_scaler = "250kDatasetModel250Nodes/scaler_250k_dataset.pkl"
    from models.interaction_network_node_classification import InteractionNetwork#SimpleMessagePassing
    
    device = 'cpu'
    torch.set_float32_matmul_precision('high')
    
    model, scaler = UnPackModelAndScaler(filepath_model,file_path_scaler)
    
    model = torch.compile(model)
    model.eval()
    model = model.to(device)
    
    
    Import_to_ONNX(model, scaler)
    
    TestONNXModelCreation(model, scaler)
    
    
    
    
