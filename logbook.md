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

- 12062024 "python3 train.py --batch-size 128 --lr 0.002 --step-size 2 --hidden-size 40" 
