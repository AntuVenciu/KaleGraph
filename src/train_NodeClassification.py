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

from models.interaction_network_node_classification import InteractionNetwork
from utils.dataset import GraphDataset
from build_graph_segmented import build_dataset
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import joblib #we use this to save the scaler.

from DataEvaluation import load_model


#We need to check what is the right number of turn. This information comes from the edges, not directly from hits.
# WARNING: THIS NUMBER HAS TO BE EQUAL TO INTERACTION_NETWORK.PY
max_n_turns = 7

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):

        #t0 = time()
        data = data.to(device)
        #output = model(data.x, data.edge_index, data.edge_attr)
        output = model(data.x,
                       data.edge_index,
                       data.edge_attr)
        
        y, output = data.y.clone().to(torch.float32), output.clone().to(device)
        
        # Have some problems here
        #if max(y.cpu().numpy()) > max_n_turns:
        #   continue
            
            
        #convert to one hot encoding.
        y = y.to(torch.long)
        y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
        y_one_hot_encoding[torch.arange(len(y)), y] = 1

        #check if there are any true edges in the graph. (
        yn = y_one_hot_encoding.numpy()
        if yn.sum() == 0:
            continue
        #weight loss function by a factor = 1 - N_i / N_TOT to count the unbalance between classes.      
        #class_weights = (yn.sum() - torch.sum(y_one_hot_encoding, dim = 0)*factor)/yn.sum()
        #class_weights = 1/ torch.sqrt(torch.sum(y_one_hot_encoding, dim=0))
        #loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
        
        
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        
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
    #print("...epoch {}: train loss={}".format(epoch, losses))
    return losses

def validate(model, device, val_loader):
    model.eval()
    losses = []
    y_pred_val = torch.empty(0, dtype=torch.long)
    y_true_val = torch.empty(0, dtype=torch.long)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            #output = model(data.x, data.edge_index, data.edge_attr)
            output = model(data.x,
                           data.edge_index,
                           data.edge_attr)
            y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32).to(device)
            #perform one hot encoding trasformation.    
            y = y.to(torch.long)
            
            #if max(y.cpu().numpy()) > max_n_turns:
            #	continue
            #if there are no edges.
            #if y.numpy().sum() == 0:
                #continue
            y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
            y_one_hot_encoding[torch.arange(len(y)), y] = 1
            yn = y_one_hot_encoding.numpy()
            
            
            #print(torch.argmax(output, dim = 1))
            #print(torch.argmax(y_one_hot_encoding, dim = 1))
            # weight loss function by a factor = N_i / N_TOT to count the unbalance between classes.      
            #class_weights = (yn.sum() - torch.sum(y_one_hot_encoding, dim = 0)*factor)/yn.sum()
            #class_weights = 1/ torch.sqrt(torch.sum(y_one_hot_encoding, dim=0))
            #loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
            
            loss_fn = torch.nn.CrossEntropyLoss().to(device)
            loss = loss_fn(output, y)
            losses.append(loss.item())
            #let's build a confusion matrix for all turns.


            # we keep calculating those for all epochs since we need to keep account of accuracy throughout the training. At the moment we only see the last step to see if it converges to ill minimum.
            y_pred_val = torch.cat((y_pred_val,torch.argmax(output.cpu(), dim = 1)), dim =0)
            y_true_val = torch.cat((y_true_val,torch.argmax(y_one_hot_encoding, dim = 1)), dim =0)

            

    print('... val loss: {:.4f}\n'
          .format(np.mean(losses)))
    #print(y_pred_val)
    y_pred_val_np = y_pred_val.cpu().numpy()
    y_true_val_np = y_true_val.cpu().numpy()
    
    Confusion_mat = confusion_matrix(y_pred_val_np, y_true_val_np)      
          

    return  Confusion_mat, losses 

def test(model, device, test_loader, scalers,isFinalEpoch,thld=0.5):
    model.eval()
    losses, accs = [], []
    y_pred_test = torch.empty(0, dtype=torch.long)
    y_true_test = torch.empty(0, dtype=torch.long)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            #output = model(data.x, data.edge_index, data.edge_attr)
            output = model(data.x,
                           data.edge_index,
                           data.edge_attr)

            

            y, output = data.y.clone().to(torch.float32), output.clone().to(torch.float32).to(device)
            
            #perform one hot encoding trasformation.
            y = y.to(torch.long)
            # Have some problems here
            #if max(y.cpu().numpy()) > max_n_turns:
            #    continue
            y_one_hot_encoding = torch.zeros(len(y), max_n_turns+1)
            y_one_hot_encoding[torch.arange(len(y)), y] = 1
            
            #let's build a confusion matrix for all turns.
            # we keep calculating those for all epochs since we need to keep account of accuracy throughout the training. At the moment we only see the last step to see if it converges to ill minimum.
            y_pred_test = torch.cat((y_pred_test,torch.argmax(output.cpu(), dim = 1)), dim =0)
            y_true_test = torch.cat((y_true_test,torch.argmax(y_one_hot_encoding, dim = 1)), dim =0)
            
            
            
            yn = y_one_hot_encoding.numpy()
            #class_weights = (yn.sum() - torch.sum(y_one_hot_encoding, dim = 0)*factor)/yn.sum()

            
            #class_weights = 1 / torch.sqrt(torch.sum(y_one_hot_encoding, dim=0))
            
            #loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
            
            loss_fn = torch.nn.CrossEntropyLoss().to(device)
            loss = loss_fn(output, y)
            losses.append(loss.item())

            
            if isFinalEpoch == True:  
                
                # Trova gli ID dei grafi nel batch
                unique_graph_ids = data.batch.unique().cpu().numpy()
                print(unique_graph_ids)
                #random_graph_id = unique_graph_ids[torch.randint(0, len(unique_graph_ids), (1,))].item()  # Scegli uno a caso
                for random_graph_id in unique_graph_ids:

                    # Crea una maschera per i nodi del grafo selezionato
                    node_mask = data.batch == random_graph_id

                    # Trova gli edge che collegano nodi appartenenti al grafo selezionato
                    edge_mask = (node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]])


                    # Seleziona gli edge e gli attributi corrispondenti
                    edge_index = data.edge_index[:, edge_mask].cpu()
                    edge_attr = data.edge_attr[edge_mask].cpu() if data.edge_attr is not None else None

                    X_subgraph = data.x[node_mask].cpu()
                    X_denormalized = torch.tensor(scalers['X'].inverse_transform(X_subgraph), dtype=torch.float32).numpy()
                    edge_attr_denormalized = torch.tensor(scalers['edge_attr'].inverse_transform(edge_attr), dtype=torch.float32).numpy()
                
                    
                    output_file = f'TruthWithNoise/{random_graph_id}_test_pred_truth.npz'
                    np.savez(output_file, 
                         X=X_denormalized,
                         edge_attr=edge_attr_denormalized,
                         edge_index=edge_index.numpy(),
                         truth=y[node_mask].cpu().numpy(),
                         predicted=torch.argmax(output[node_mask].cpu(), dim =1).numpy())
            
            
    print('... test loss: {:.4f}\n'
          .format(np.mean(losses)))
    Confusion_mat = confusion_matrix(y_pred_test.cpu().numpy(), y_true_test.cpu().numpy())
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
    parser.add_argument('--save-sample-test', action = 'store_true', default = False,
    			help = 'For saving the last batch of test dataset with both truth and prediction for visualization')
    parser.add_argument('--load-model', type = str , default = None,
    			help = 'For reloading a model, to retrain.')	
    parser.add_argument('--load-scaler', type = str , default = None,
    			help = 'For reloading a model scaler, to retrain.')
    args = parser.parse_args()

    use_cuda = False

    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)

    # Load the dataset
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    inputdir = "DataWithNoise"
    #inputdir = "/meg/data1/shared/subprojects/cdch/ext-venturini_a/GNN/NoPileUpMC"
    graph_files = glob.glob(os.path.join(inputdir, "*.npz"))

    # Check that the dataset has already been created
    
    if len(graph_files) < 1:
        print("Dataset not loaded correctly") 
        return

    # Limit while we wait for GPU training
    graph_files = graph_files[:]
    time_steps = 2
    # Split the dataset
    f_train = 0.75
    f_test = 0.15
    #graph_files = graph_files[0:50000]
    partition = {'train': graph_files[: int(f_train * len(graph_files))],
                 'test':  graph_files[int(f_train * len(graph_files)) : int((f_train + f_test)*len(graph_files))],
                 'val': graph_files[int((f_train + f_test)*len(graph_files)) : ]}

    params = {'batch_size': args.batch_size, 'shuffle' : True, 'num_workers' : 4}
   
    train_set = GraphDataset(partition['train'])
    #train_set.plot(1)
    
    

    
    


    
    
    

    # Set to the correct number of features
    NUM_NODE_FEATURES = train_set.get_X_dim()
    NUM_EDGE_FEATURES = train_set.get_edge_attr_dim()
    
    print(NUM_NODE_FEATURES)
    print(NUM_EDGE_FEATURES)
    #load model and scaler
    model = None
    scaler = None
    if args.load_model:
        
        if not args.load_scaler:
            raise OSError('Please specify a path for a scaler.')
        
        ModelState = torch.load(args.load_model)
        optimizer, scheduler,model = load_model(ModelState)
        scalers = joblib.load(args.load_scaler)
        if ModelState['hyper_params']['node_features'] != NUM_NODE_FEATURES:
            raise ValueError(
                      "The number of node features of the model ({model_features}) and of the data ({data_features}) do not match. "
                      "Please check you are using the correct dataset or loading the correct model.".format(
                                                                model_features=ModelState['hyper_params']['node_features'], 
                                                                data_features=NUM_NODE_FEATURES
                                                                )
                            )
        
        if ModelState['hyper_params']['edge_features'] != NUM_EDGE_FEATURES:
            raise ValueError(
                      "The number of node features of the model ({model_features}) and of the data ({data_features}) do not match. "
                      "Please check you are using the correct dataset or loading the correct model.".format(
                                                                model_features=ModelState['hyper_params']['edge_features'], 
                                                                data_features=NUM_EDGE_FEATURES
                                                                )
                            )
        train_set = GraphDataset(partition['train'], scalers = scalers, fitted = True)
        model = model.to(device)
        print("Model successfully loaded")
    else:
        model = InteractionNetwork(args.hidden_size, NUM_NODE_FEATURES, NUM_EDGE_FEATURES, time_steps).to(device)
        train_set.scale()
        scalers = train_set.scalers
    print(device)
    torch.set_float32_matmul_precision('high')

    #now that we have a scaler, we can scale the test and val data according to it
    test_set = GraphDataset(partition['test'], scalers=scalers, fitted=True)
    val_set = GraphDataset(partition['val'], scalers=scalers, fitted=True)
    
    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)
    val_loader = DataLoader(val_set, **params)

    print(f"Number of train data samples : {train_set.len()}")
    print(f"Number of tests data samples : {test_set.len()}")
    print(f"Number of valid data samples : {val_set.len()}")

    model = torch.compile(model)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print('total trainable params:', total_trainable_params)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = StepLR(optimizer, step_size=args.step_size,
    #                   gamma=args.gamma)
    
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-5, args.lr,  step_size_up=10, step_size_down=None, mode='triangular', gamma=args.gamma, scale_fn=None, scale_mode='cycle',last_epoch=-1)
    #def lr_lambda(epoch):
    #    return 1 if epoch < 10 else (1e-5 / args.lr)  # Dopo 10 epoche, diventa 1e-4
     
    def lr_lambda(epoch):
        if epoch < 5:
            return 1  # LR normale (3e-3)
        elif epoch < 10:
            return 1e-3 / args.lr  # Ridotto a 1e-3
        elif epoch < 20:
            return 1e-4 / args.lr  # Ridotto a 3e-4
        elif epoch < 40:
            return 0.7e-4 / args.lr
      

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output = {'train_loss': [], 'val_loss' : [], 'val_tpr' : [], 'val_tpr' : [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))

        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        Val_confusion_mat, val_loss= validate(model, device, val_loader)
        isFinalEpoch = False
        if epoch == args.epochs :
            isFinalEpoch =  True
        #print('...optimal threshold', thld)
        Test_confusion_mat, test_loss = test(model, device, test_loader, scalers,isFinalEpoch, thld = 0.5)
        scheduler.step()

        output['train_loss'].append(np.mean(train_loss))
        output['val_loss'].append(np.mean(val_loss))
        output['test_loss'].append(np.mean(test_loss))

    
    # Plotting of history
    
    import matplotlib.pyplot as plt
    
    
    plt.figure(1)

    plt.title("KaleGraph Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(np.linspace(1, len(output['train_loss']), len(output['train_loss'])), output['train_loss'], label='Training', color='blue')
    plt.plot(np.linspace(1, len(output['test_loss']), len(output['test_loss'])), output['test_loss'], label='Test', color='orange')
    plt.plot(np.linspace(1, len(output['val_loss']), len(output['val_loss'])), output['val_loss'], label='Validation', color='red')


    #filename = "Val_confusion_matrix.txt"

    # Salva la matrice di confusione in un file di testo
    #with open(filename, 'w') as f:
    #    f.write("Confusion Matrix (Normalized by rows):\n")
    #    for i, row in enumerate(Val_confusion_mat):
    #        f.write(f"Row {i}: {row}\n")
            
    #filename = "Test_confusion_matrix.txt"

    # Salva la matrice di confusione in un file di testo
    #with open(filename, 'w') as f:
    #   f.write("Confusion Matrix (Normalized by rows):\n")
    #    for i, row in enumerate(Test_confusion_mat):
    #        f.write(f"Row {i}: {row}\n")
            
    
    plt.legend()
    plt.show()
    #plt.savifig(f"loss_training_cuda_{time.struct_time()[0]}{time.struct_time()[1]}{time.struct_time()[2]}.png")
    #plt.show()
    #Val_confusion_mat = Val_confusion_mat / Val_confusion_mat.sum(axis=1, keepdims=True)
    #Test_confusion_mat = Test_confusion_mat / Test_confusion_mat.sum(axis=1, keepdims=True)
    labels = [ 'Noise','Turn 1', 'Turn 2', 'Turn 3', 'Turn 4', 'Turn 5', 'Turn 6']
    # Visualizza la matrice di confusione con un heatmap
    
    sns.heatmap(Val_confusion_mat, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    sns.heatmap(Test_confusion_mat, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    #plt.savefig(f"confusion_matrix_training_cuda_{time.struct_time()[0]}{time.struct_time()[1]}{time.struct_time()[2]}.png")
    plt.show()

    if args.save_model:
        torch.save({'epoch':args.epochs,
                    'train_loss':torch.tensor( output['train_loss'],dtype=torch.float32),
                    'val_loss':torch.tensor( output['val_loss'],dtype=torch.float32),
                    'test_loss':torch.tensor( output['test_loss'],dtype=torch.float32),
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'Val_Conf_matrix':torch.tensor(Val_confusion_mat, dtype=torch.int32),
                    'Test_Conf_matrix': torch.tensor(Test_confusion_mat, dtype=torch.int32),
                    'hyper_params': {'hidden_size':args.hidden_size, 'node_features':NUM_NODE_FEATURES, 'edge_features':NUM_EDGE_FEATURES, 'time_steps': time_steps}
                    }, f"{args.hidden_size}model1.pth"
                   )
        joblib.dump(scalers, f"{args.hidden_size}scaler1.pkl")

if __name__ == '__main__':
    # torch.set_num_threads(1)
    main()
