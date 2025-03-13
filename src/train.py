"""
Train your model
"""
import argparse
import glob
import os
from time import time

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

max_n_turns = 5

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        #t0 = time()
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr)

        y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32)
        # Have some problems here
        if max(y.numpy()) > 5:
            continue
        #convert to one hot encoding.
        y = y.to(torch.int)
        y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
        y_one_hot_encoding[torch.arange(len(y)), y] = 1
        
        #check if there are any true edges in the graph. (
        yn = y_one_hot_encoding.numpy()
        if yn.sum() == 0:
            continue
        # weight loss function by a factor = N_i / N_TOT to count the unbalance between classes.      
        class_weights = torch.sum(y_one_hot_encoding, dim = 0)/yn.sum()
         
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(output, y_one_hot_encoding)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #t1 = time()
        #print(f"time for the whole batch = {t1 - t0:.3f} s")
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        losses.append(loss.item())
    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))
    return losses

def validate(model, device, val_loader):
    model.eval()
    losses = []
    y_pred_val = torch.empty(0, dtype=torch.long)
    y_true_val = torch.empty(0, dtype=torch.long)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32)
            #perform one hot encoding trasformation.    
            y = y.to(torch.int)
            # Have some problems here
            if max(y.numpy()) > 5:
                continue
            #if there are no edges.
            if y.numpy().sum() == 0:
                continue
            y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
            y_one_hot_encoding[torch.arange(len(y)), y] = 1
            yn = y_one_hot_encoding.numpy()
            # weight loss function by a factor = N_i / N_TOT to count the unbalance between classes.      
            class_weights = torch.sum(y_one_hot_encoding, dim = 0)/yn.sum()
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(output, y_one_hot_encoding)
            losses.append(loss)
            #let's build a confusion matrix for all turns.
            torch.cat((y_pred_val, torch.argmax(output, dim = 1)))
            torch.cat((y_true_val, torch.argmax(y_one_hot_encoding, dim = 1)))
            
    Confusion_mat = confusion_matrix(y_pred_val.to_numpy(), y_true_val.to_numpy())
    return  Confusion_mat, np.mean(losses) 

def test(model, device, test_loader, thld=0.5):
    model.eval()
    losses, accs = [], []
    y_pred_test = torch.empty(0, dtype=torch.long)
    y_true_test = torch.empty(0, dtype=torch.long)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)

            #let's build a confusion matrix for all turns.
            torch.cat((y_pred_test, torch.argmax(output, dim = 1)))
            torch.cat((y_true_test, data.y))
            y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32)

            #perform one hot encoding trasformation.
            y = y.to(torch.int)
            # Have some problems here
            if max(y.numpy()) > 5:
                continue
            y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
            y_one_hot_encoding[torch.arange(len(y)), y] = 1
            yn = y_one_hot_encoding.numpy()
            class_weights = torch.sum(y_one_hot_encoding, dim = 0)/yn.sum()
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(output, y_one_hot_encoding)
            losses.append(loss)
            #print(f"acc={TP+TN}/{TP+TN+FP+FN}={acc}")

    print('... test loss: {:.4f}\n'
          .format(np.mean(losses)))
    Confusion_mat = confusion_matrix(y_pred_test.to_numpy(), y_true_test.to_numpy())
    return Confusion_mat, np.mean(losses) #np.mean(losses), np.mean(accs)

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
    parser.add_argument('--hidden-size', type=int, default=50,
                        help='Number of hidden units per layer')

    args = parser.parse_args()
    # Train on cpu for now
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

    # Load the dataset
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    inputdir = "../dataset"
    graph_files = glob.glob(os.path.join(inputdir, "*.npz"))

    # Check that the dataset has already been created
    if len(graph_files) < 1000:
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
    train_set.plot(8)
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


    output = {'train_loss': [], 'val_loss' : [], 'val_tpr' : [], 'val_tpr' : [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        Val_confusion_mat, val_loss= validate(model, device, val_loader)
        #print('...optimal threshold', thld)
        Test_confusion_mat, test_loss = test(model, device, test_loader, thld=thld)
        scheduler.step()
        """
        if args.save_model:
            torch.save(model.state_dict(),
                       "trained_models/train{}_PyG_{}_epoch{}_{}GeV_redo.pt"
                       .format(args.sample, args.construction, epoch, args.pt))
        """
        output['train_loss'] += train_loss
        output['val_loss'] += val_loss
        #output['val_tpr'] += val_tpr
        output['test_loss'] += test_loss
        #output['test_acc'] += test_acc
        
        """
        np.save('train_output/train{}_PyG_{}_{}GeV_redo'
                .format(args.sample, args.construction, args.pt),
                output)
        """

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
    

if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
