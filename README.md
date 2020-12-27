# Data-Mining-Course-Project
Graph Representation Learning

## Problem Definition

This project focuses on Graph-structured Data based on AutoML. You should design a computer program capable of providing solutions to graph representation learning problems autonomously (without any human intervention).

You should propose automatic solutions that can effectively and efficiently learn high-quality representation for each node based on the given features, neighborhood and structural information underlying the graph. The solutions should be designed to automatically extract and utilize any useful signals in the graph.

The main task is to classify the given nodes

## Usage

Run ./code/main.py to run the training process as well as test process with argments:

* data - to choice a data set : 'a', 'b', 'c', 'd', 'e'

* overwrite_cache - command to overwrite cache

* additional_features - command to add adjacent matrix as features

* maxepoch - number of epochs in training process: default 200

* lr - learning rate: default 0.01

* n_layer - number of layers: default 2

* weight_decay - weight decay: default 1e-4

* drop - dropout rate: default 0.5

* hidden - number of channels of hidden layer: default 64

Example of local running:

```
python ./code/main.py --data a
```

## Ideas

* node2vec - failed
* nodes with degree 1 / 2