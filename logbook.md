# Training Logbook
65;7006;1c
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

### ATTENTION: Very important!

A new training goal is needed for the algorithm to converge:
1. Either we try to classify good nodes
2. Or we try to classify CONSECUTIVE edges between good nodes
We concentrate on 2. since it is more like what has been already developed.
We train with 5% cut on CDCH connection. NO SPX data