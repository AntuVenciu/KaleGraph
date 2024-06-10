"""
Train your model
"""
import argparse
import os
from time import time

import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

from models.interaction_network import InteractionNetwork
from utils.dataset import GraphDataset, load_data
from utils.build_graph import build_adjacency_matrix


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y.clone().to(torch.float32), output.squeeze(1).clone().to(torch.float32)
        #print(f"y shape = {data.y.shape}\noutput shape = {output.shape}")
        #print(f"true = {y.numpy()}\npredicted = {output}")
        # weight loss function by a factor = N_0 / N_1 to count the unbalance beween 1 and 0'2
        #print(f"Number of zeros = {len(y < 0.5)}")
        #print(f"Number of ones = {len(y > 0.5)}")
        class_weight = torch.Tensor([len(y < 0.5) / len(y > 0.5)])
        #print(f"Fraction of edges with a good connection = {len(data.y < 0.5) / max(len(data.y > 0.5), 1)}")
        loss = F.binary_cross_entropy_with_logits(output, y, reduction='mean', pos_weight=class_weight)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        losses.append(loss.item())
    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))
    return np.mean(losses)

def validate(model, device, val_loader):
    model.eval()
    opt_thlds, accs, tps, fns = [], [], [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            y, output = data.y.clone().to(torch.float32), output.squeeze(1).clone().to(torch.float32)
            loss = F.binary_cross_entropy(output, y, reduction='mean').item()
        
            # define optimal threshold (thld) where TPR = TNR 
            diff, opt_thld, opt_acc = 100, 0, 0
            best_tpr, best_tnr = 0, 0
            for thld in np.arange(0.5, 1, 0.05):
                TP = torch.sum((y==1) & (output>thld)).item()
                TN = torch.sum((y==0) & (output<thld)).item()
                FP = torch.sum((y==0) & (output>thld)).item()
                FN = torch.sum((y==1) & (output<thld)).item()
                acc = (TP+TN)/(TP+TN+FP+FN)
                if TP + FN == 0:
                    TPR = 0
                else:
                    TPR = TP / (TP + FN)
                if TN + FP == 0:
                    TNR = 0
                else:
                    TNR = TN / (TN + FP)
                delta = abs(TPR-TNR)
                if (delta < diff):
                    diff, opt_thld, opt_acc, best_tpr = delta, thld, acc, TPR

            opt_thlds.append(opt_thld)
            accs.append(opt_acc)
            tps.append(best_tpr)
    
    print("... val accuracy = ", np.mean(accs))
    print("... val TPR = ", np.mean(tps))
    return np.mean(opt_thlds) 

def test(model, device, test_loader, thld=0.5):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            TP = torch.sum((data.y==1).squeeze() & 
                           (output>thld).squeeze()).item()
            TN = torch.sum((data.y==0).squeeze() & 
                           (output<thld).squeeze()).item()
            FP = torch.sum((data.y==0).squeeze() & 
                           (output>thld).squeeze()).item()
            FN = torch.sum((data.y==1).squeeze() & 
                           (output<thld).squeeze()).item()            
            acc = (TP+TN)/(TP+TN+FP+FN)
            y, output = data.y.clone().to(torch.float32), output.squeeze(1).clone().to(torch.float32)
            loss = F.binary_cross_entropy(output, y, 
                                          reduction='mean').item()
            accs.append(acc)
            losses.append(loss)
            #print(f"acc={TP+TN}/{TP+TN+FP+FN}={acc}")

    print('... test loss: {:.4f}\n... test accuracy: {:.4f}'
          .format(np.mean(losses), np.mean(accs)))
    return np.mean(losses), np.mean(accs)

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='Learning rate step size')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--hidden-size', type=int, default=40,
                        help='Number of hidden units per layer')

    args = parser.parse_args()
    # Train on cpu for now
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

    # Load adjacency matrix
    adj_matrix = build_adjacency_matrix()
    
    # Load the dataset
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    home_dir = "/meg/home/ext-venturini/meg2/analyzer/KaleGraph"
    inputdir = f"/meg/home/ext-venturini_a/meg2/analyzer"
    
    train_file = f"{inputdir}/1e6TrainSet_CDCH_SPX_noSelectedPositron.txt"
    test_file = f"{inputdir}/1e6TestSet_CDCH_SPX_noSelectedPositron.txt"
    val_file = f"{inputdir}/1e6ValSet_CDCH_SPX_noSelectedPositron.txt"

    partition = {'train': train_file,
                 'test':  test_file,
                 'val': val_file}

    params = {'batch_size': args.batch_size, 'shuffle' : True, 'num_workers' : 6}
    
    train_set = GraphDataset(partition['train'], adj_matrix)
    #train_set.plot(4)
    train_loader = DataLoader(train_set, **params)
    test_set = GraphDataset(partition['test'], adj_matrix)
    test_loader = DataLoader(test_set, **params)
    val_set = GraphDataset(partition['val'], adj_matrix)
    val_loader = DataLoader(val_set, **params)
    
    print(f"Number of train data samples : {train_set.len()}")
    print(f"Number of tests data samples : {test_set.len()}")
    print(f"Number of valid data samples : {val_set.len()}")

    # Set to the correct number of features in utils/dataset.py
    NUM_NODE_FEATURES = 6
    NUM_EDGE_FEATURES = 3

    model = InteractionNetwork(args.hidden_size, NUM_NODE_FEATURES, NUM_EDGE_FEATURES).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print('total trainable params:', total_trainable_params)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)


    output = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        thld = validate(model, device, val_loader)
        print('...optimal threshold', thld)
        test_loss, test_acc = test(model, device, test_loader, thld=thld)
        scheduler.step()
        """
        if args.save_model:
            torch.save(model.state_dict(),
                       "trained_models/train{}_PyG_{}_epoch{}_{}GeV_redo.pt"
                       .format(args.sample, args.construction, epoch, args.pt))
        """

        output['train_loss'].append(train_loss)
        output['test_loss'].append(test_loss)
        output['test_acc'].append(test_acc)

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
    plt.plot(args.epochs, output['train_loss'], label='Training', color='blue')
    plt.plot(args.epochs, output['test_loss'], label='Test', color='orange')
    plt.plot(args.epochs, output['val_loss'], label='Validation', color='red')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()


