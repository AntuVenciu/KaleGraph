"""
Train your model
"""
import argparse
import glob
import os
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.profiler import profile, record_function, ProfilerActivity

from models.interaction_network import InteractionNetwork
from utils.dataset import GraphDataset
from build_graph_segmented import build_dataset
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler



#We need to check what is the right number of turn. This information comes from the edges, not directly from hits.
# WARNING: THIS NUMBER HAS TO BE EQUAL TO INTERACTION_NETWORK.PY
max_n_turns = 6

def train(args, model, device, train_loader, optimizer, epoch, scaler):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        #t0 = time()
        data = data.to(device)
        #output = model(data.x, data.edge_index, data.edge_attr)
        output = model(torch.tensor(scaler.fit_transform(data.x)),
                       data.edge_index,
                       torch.tensor(scaler.fit_transform(data.edge_attr)))
                       
                       
        y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32)
        # Have some problems here
        if max(y.numpy()) > max_n_turns:
            continue
        #convert to one hot encoding.
        y = y.to(torch.long)
        y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
        y_one_hot_encoding[torch.arange(len(y)), y] = 1
        
        #check if there are any true edges in the graph. (
        yn = y_one_hot_encoding.numpy()
        if yn.sum() == 0:
            continue
        # weight loss function by a factor = N_i / N_TOT to count the unbalance between classes.      
        class_weights = torch.sum(y_one_hot_encoding, dim = 0)/yn.sum()
         
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        loss = loss_fn(output, y)
        #print(y)
        #print(output)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        #print(f"time for the whole batch = {t1 - t0:.3f} s")
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        losses.append(loss.item())
        
    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, losses))
    return losses

def validate(model, device, val_loader, scaler):
    model.eval()
    losses = []
    y_pred_val = torch.empty(0, dtype=torch.long)
    y_true_val = torch.empty(0, dtype=torch.long)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            #output = model(data.x, data.edge_index, data.edge_attr)
            output = model(torch.tensor(scaler.transform(data.x)),
                           data.edge_index,
                           torch.tensor(scaler.transform(data.edge_attr)))
            y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32)
            #perform one hot encoding trasformation.    
            y = y.to(torch.long)
            
            # Have some problems here
            if max(y.numpy()) > max_n_turns:
                continue
            #if there are no edges.
            if y.numpy().sum() == 0:
                continue
            y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
            y_one_hot_encoding[torch.arange(len(y)), y] = 1
            yn = y_one_hot_encoding.numpy()
            
            
            #print(torch.argmax(output, dim = 1))
            #print(torch.argmax(y_one_hot_encoding, dim = 1))
            # weight loss function by a factor = N_i / N_TOT to count the unbalance between classes.      
            class_weights = torch.sum(y_one_hot_encoding, dim = 0)/yn.sum()
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(output, y)
            losses.append(loss.item())
            #let's build a confusion matrix for all turns.


            # we keep calculating those for all epochs since we need to keep account of accuracy throughout the training. At the moment we only see the last step to see if it converges to ill minimum.
            y_pred_val = torch.argmax(output, dim = 1)
            y_true_val = torch.argmax(y_one_hot_encoding, dim = 1)

            

    print('... val loss: {:.4f}\n'
          .format(np.mean(losses)))
    #print(y_pred_val)
    y_pred_val_np = y_pred_val.numpy()
    y_true_val_np = y_true_val.numpy()
    
    Confusion_mat = confusion_matrix(y_pred_val_np, y_true_val_np)      
          

    return  Confusion_mat, losses 

def test(model, device, test_loader, scaler, thld=0.5):
    model.eval()
    losses, accs = [], []
    y_pred_test = torch.empty(0, dtype=torch.long)
    y_true_test = torch.empty(0, dtype=torch.long)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            #output = model(data.x, data.edge_index, data.edge_attr)
            output = model(torch.tensor(scaler.transform(data.x)),
                           data.edge_index,
                           torch.tensor(scaler.transform(data.edge_attr)))

            

            y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32)
            
            #perform one hot encoding trasformation.
            y = y.to(torch.long)
            # Have some problems here
            if max(y.numpy()) > max_n_turns:
                continue
            y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
            y_one_hot_encoding[torch.arange(len(y)), y] = 1
            
            #let's build a confusion matrix for all turns.
            # we keep calculating those for all epochs since we need to keep account of accuracy throughout the training. At the moment we only see the last step to see if it converges to ill minimum.
            y_pred_test = torch.argmax(output, dim = 1)
            y_true_test = torch.argmax(y_one_hot_encoding, dim = 1)
            
            
            
            yn = y_one_hot_encoding.numpy()
            class_weights = torch.sum(y_one_hot_encoding, dim = 0)/yn.sum()
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(output, y)
            losses.append(loss.item())



    print('... test loss: {:.4f}\n'
          .format(np.mean(losses)))
    Confusion_mat = confusion_matrix(y_pred_test.numpy(), y_true_test.numpy())
    return Confusion_mat, losses #np.mean(losses), np.mean(accs)

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=1,
                        help='Learning rate step size')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--hidden-size', type=int, default=100,
                        help='Number of hidden units per layer')

    args = parser.parse_args()
    # Train on cpu for now
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

    # Load the dataset
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    #inputdir = "."
    inputdir = "/meg/data1/shared/subprojects/cdch/ext-venturini_a/GNN/NoPileUpMC"
    graph_files = glob.glob(os.path.join(inputdir, "*.npz"))

    # Check that the dataset has already been created
    
    if len(graph_files) < 15:
        print("Dataset not loaded correctly") 
        return

    # Split the dataset
    f_train = 0.75
    f_test = 0.15

    partition = {'train': graph_files[: int(f_train * len(graph_files))],
                 'test':  graph_files[int(f_train * len(graph_files)) : int((f_train + f_test)*len(graph_files))],
                 'val': graph_files[int((f_train + f_test)*len(graph_files)) : ]}

    params = {'batch_size': args.batch_size, 'shuffle' : False, 'num_workers' : 4}
    
    train_set = GraphDataset(partition['train'])
    train_set.plot(1)
    train_loader = DataLoader(train_set, **params)
    test_set = GraphDataset(partition['test'])
    test_loader = DataLoader(test_set, **params)
    val_set = GraphDataset(partition['val'])
    val_loader = DataLoader(val_set, **params)
    
    print(f"Number of train data samples : {train_set.len()}")
    print(f"Number of tests data samples : {test_set.len()}")
    print(f"Number of valid data samples : {val_set.len()}")

    # Set to the correct number of features
    NUM_NODE_FEATURES = train_set.get_X_dim()
    NUM_EDGE_FEATURES = train_set.get_edge_attr_dim()
    model = InteractionNetwork(args.hidden_size, NUM_NODE_FEATURES, NUM_EDGE_FEATURES, time_steps=1).to(device)
    model = torch.compile(model)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print('total trainable params:', total_trainable_params)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)

    scaler = StandardScaler() 
    output = {'train_loss': [], 'val_loss' : [], 'val_tpr' : [], 'val_tpr' : [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))

        train_loss = train(args, model, device, train_loader, optimizer, epoch, scaler)
        Val_confusion_mat, val_loss= validate(model, device, val_loader,scaler)

        #print('...optimal threshold', thld)
        Test_confusion_mat, test_loss = test(model, device, test_loader,scaler, thld = 0.5)
        scheduler.step()

        output['train_loss'] += train_loss
        output['val_loss'] += val_loss
        #output['val_tpr'] += val_tpr
        output['test_loss'] += test_loss
        #output['test_acc'] += test_acc
    
    # Plotting of history
    
    import matplotlib.pyplot as plt
    
    
    plt.figure(1)

    plt.title("KaleGraph Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(np.linspace(1, len(output['train_loss']), len(output['train_loss'])), output['train_loss'], label='Training', color='blue')
    plt.plot(np.linspace(1, len(output['test_loss']), len(output['test_loss'])), output['test_loss'], label='Test', color='orange')
    plt.plot(np.linspace(1, len(output['val_loss']), len(output['val_loss'])), output['val_loss'], label='Validation', color='red')

    plt.legend()
    plt.show()
    
    
    labels = ['Rumore', 'Giro 1', 'Giro 2', 'Giro 3', 'Giro 4', 'Giro 5', 'Giro 6']
    # Visualizza la matrice di confusione con un heatmap
    sns.heatmap(Val_confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    sns.heatmap(Test_confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
