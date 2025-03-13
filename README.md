# KaleGraph

This is an implementation of a GNN architecture
built to improve MEGII track finding algorithm.

## The Architecture

We follow the idea from the article "Charged Particle Tracking via Edge-Classifaying interaction networks"
by G. DeZoort et al. Most of their code is also used.

Hits on a tracking detector are the nodes of the graph.
We want to identify the particle trajectory between the edges that connects signal hits through a multiclassification
network. Multiclassification because in MEGII  we have several turns per track, each turn with distinctive characteristics.

From an event in the MEG II spectrometer (composed of a CYLDCH and pTC, or SPX, detector) we build
the graph of the events connecting reconstructed hits with edges with a proximity criteria for
edge building (over ways may be found).
We then perform a classification task on the edges, exploiting the message passing technique with
a dense neural network to aggregate nodes, edges features.

## Notations

X = $(N_{nodes}, N_{features})$ matrix of features of the graph node
edge_attr = $(N_{edges}, N_{features})$ matrix of features of the edges
edge_index = $(2, N_{edges})$ indeces of the nodes in the graph connected by each edge
truth = $({N_edges})$ whether an edge is a signal (truth = NTurn in the CYLDCH) or noise (0)

The GNN acts like:

1. Update of the node features with message passing between connected nodes
   $$X_i \to O(X_i, \sum_j R_1(edge_{ij}, X_i, X_j))$$
   where O and $R_2$ are neural networks made of many dense layers
2. After passage 1. is iterated over a certain number of times (to be tuned), we draw the prediction
   $$w_{ij} = R_2(edge_{ij}, X_i, X_j)$$

## Repository organization

**NOTE** most of the following refers, at the moment, to the development in segmented_cdch branch

* src : The main functions to build the graph and train it;
  * train.py : trains the InteractionNetwork model
  * build_graph_segmented.py : build the graphs for training and validation etc. from data
  * utils/ :
    * plot_graph.py : plotting functions to aid in debugging
* dataset (untracked) : the directory where to store data for training
