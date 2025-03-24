# Training Logbook

To be sure that work is done properly and that no effort is done in vain,
we shall write a logbook about training tests, in view of finding optimal
GNN architecture and training parameters.

## Rules for training

Each training should last at least 10 epochs.
Meaningful results should be saved in plot form and/or registered in this logbook.
The structure of the logbook should be:

date "command line with args to train" loss-after-10-epochs thrd TPR accuracy

If optimizer is changed, or major changes in the GNN architecture are made, they
should be properly documented before reporting the output of the training.
Unless overwise specified, new set-ups are assumed to be the new default.

## Logbook Entries

- 12062024 "python3 train.py --batch-size 32 --lr 0.002 --step-size 2 --hidden-size 40" 1.21 0.25 0.7 0.7

Use larger dataset with 200k events
(Takes too long, stopped after 2 epochs)
- 12062024 "python3 train.py --batch-size 128 --lr 0.002 --step-size 2 --hidden-size 40" 1.15 0.17 0.7 0.7

Make 2 iterations on node features and edges updates
-12062024 "python3 train.py --batch-size 128 --lr 0.001 --step-size 1 --hidden-size 80" 1.17 0.39 0.7 0.7

Add Dropout layers both in Relational and in Objective Model
-12062024 "python3 train.py --batch-size 128 --lr 0.001 --step-size 1 --hidden-size 80" 1.17 0.39 0.7 0.7

Change features of hits and nodes:
no more wireID, layers. Time in ns units.
deltaPhi, deltaZ for edges also.
In addition, we use the "max" aggregator
Train on smaller dataset for faster response.
Set dropout fraction to 0.1
Add ReLU at the beginning -> This is terrible
Set f_cdch and f_spx to 0.05

-12062024 "python3 train.py --batch-size 32 --lr 0.002 --step-size 1 --hidden-size 40" 1.15 0.5 0.68 0.68

### WARNING: Very important!

A new training goal is needed for the algorithm to converge:
1. Either we try to classify good nodes
2. Or we try to classify CONSECUTIVE edges between good nodes
We concentrate on 2. since it is more like what has been already developed.
We train with 5% cut on CDCH connection. NO SPX data
Add as a feature for the edges the average charge and amplitude of the two nodes

Try filtering only good hits
-14062024 "python3 train.py --batch-size 128 --lr 0.001 --step-size 1 --hidden-size 40" 1.29 0.16 0.68 0.68

Use a smaller dataset for quicker answers (10k events)
-14062024 "python3 train.py --batch-size 10 --lr 0.001 --step-size 5 --epochs 100 --hidden-size 60" 1.18 0.14 0.68 0.7
-14062024 "python3 train.py --batch-size 64 --epochs 100 --lr 0.005 --step-size 5 --gamma 0.9 --hidden-size 60" 0.98 0.15 0.78 0.78
-16062024 "python3 train.py --batch-size 64 --epochs 250 --lr 0.009 --step-size 10 --gamma 0.95 --hidden-size 60" 0.99 0.05 0.71 0.86

### WARNING: VERY IMPORTANT!

We log here results from a new approach, segmenting the CDCH
to have smaller graphs.

* 20032025 "python3 train.py --batch-size 128 --epochs 25 --lr 0.009 --step-size 10 --gamma 0.95 --hidden-size 140" 
	  * on 150k graphs with SPX and graph depth 4 (adjacent wires), 3 (adjacent layers), 3 (spx cdch connection)
	  * total number of trainable parameters ~60000


### 24 March 2025
New architecture: tried three different models:
* 24032025 Model 1: "python3 train.py --epochs 100 --lr 0.5e-4  --gamma 1 --hidden-size 250" 
* 24032025 Model 2: "python3 train.py --epochs 100 --lr 0.5e-4  --gamma 1 --hidden-size 500" 
* 24032025 Model 3: "python3 train.py --epochs 100 --lr 0.5e-4 --gamma 1 --hidden-size 1000" 

Model1 1st turn Sensitivity on test: 87%, 
Model2 1st turn Sensitivity on test: 89% 
Model3 1st turn Sensitivity on test: 92%   

The most performant model on testing seems to be the third with hidden size equal to 1000.

1)Aggregation function: 'sum'--> added before removing weighting of classes, it performed better compared to 'mean', 'max' aggregations. In order. 'sum'>'max'>'mean'
2)Message passing steps: 1
3)Feature used: 'x0', 'y0', 'zTimeDiff', 'isSPX', 'amplitude', 'thetaStereo', 'phiStereo', 'Time'
4)For SPXHits, phiStereo and thetaStereo are put to 0.
5)Training was performed on full graph
6)Removed weighting of classes


