# KaleGraph

This is an implementation of a GNN architecture
built to improve MEG II track finding algorithm.

## The Architecture

We follow the idea from the article "Charged Particle Tracking via Edge-Classifaying interaction networks"
by G. DeZoort et al. Most of their code is also used.

Hits on a tracking detector are the nodes of the graph.
We want to identify the particle trajectory between the edges that connects signal hits through a multiclassification
network. Multiclassification because in MEG II  we have several turns per track, each turn with distinctive characteristics.

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

* `src/` : The main functions to build the graph and train it;
  * `train.py` : trains the InteractionNetwork model
  * `build_graph_segmented.py` : build the graphs for training and validation etc. from data
  * `utils/` :
    * `plot_graph.py` : plotting functions to aid in debugging
* `dataset` (untracked) : the directory where to store data for training

### Running python on the cluster

Following some problems with MEG II cluster (ELog 1278 in Software Discussion 13 March 2025 Yusuke),
we need to activate ourselves a python working environment, waiting for a default one
suited for us to be deployed by the experiment.
This is how to do it:

1. `$ module load Python/3.9.10`
   This loads a Python environment with version `3.9.10` (enough for us).
   This command needs to be run everytime (I think)

2. `$ python3 -m venv ~/venv_gnn`
   This creates a virtual environment called `venv_gnn` in your working directory which you can reload everytime and modify as you please. We will then load libraries inside this environment

3. `$ source ~/venv_gnn/bin/activate`
   This is the command to access and work in this virtual environment.
   Run it everytime to reaccess it. No need to run again command 2.

Now if you ran command 3. you are inside `venv_gnn`. From here you can run your code and load your libraries *una tantum*.
For example:

4. `$ pip3 install pandas matplotlib torch`