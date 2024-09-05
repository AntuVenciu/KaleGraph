# KaleGraph

This is an implementation of a GNN architecture
built to improve MEG track finidng algorithm.

## Repository organization

- src: The main functions to build the graph and train it;
  1. In utils: Functions to build the graph and plot it;
- notebook: A collection of test notebook to demonstrate the functionality of this architecture (to be started)
- src/models: Repository with different GNN architecture:
  1. interaction_network is a NN which makes edge classification. It is the only one already trainable
  2. gravnet is a NN which builds its own graph. To be implemented
- data: repository of training datasets based on 1e6 MEG data
