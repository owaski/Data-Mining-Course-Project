# used to read and process the data

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch

def preprocess(train_dir, test_label_file):

    # read config
    with open(os.path.join(train_dir, 'config.yml'), 'r') as r:
        config = yaml.safe_load(r)

    # read edges
    edges_tsv = pd.read_csv(os.path.join(train_dir, 'edge.tsv'), sep='\t', header=0)
    edges = []
    for idx, row in edges_tsv.iterrows():
        edges.append(row.to_list())
    config['n_edge'] = len(edges)
    edges=torch.Tensor(edges).long().t()
    edge_index=edges[[0,1]]
    edge_attr=edges[[2]].t()
    
    # read features
    features_tsv = pd.read_csv(os.path.join(train_dir, 'feature.tsv'), sep='\t', header=0)
    features = []
    for idx, row in features_tsv.iterrows():
        features.append(row.to_list()[1:]) # exclude the node_index
    assert len(features) == features_tsv.shape[0]
    config['n_feature'] = len(features)
    features = torch.Tensor(np.array(features, dtype=float))

    config['n_vertex'] = features.shape[0]

    # read labels
    train_labels_csv = pd.read_csv(os.path.join(train_dir, 'train.csv'), sep='\t', header=None)
    test_labels_csv = pd.read_csv(test_label_file, sep='\t', header=None)
    
    labels = np.zeros(config['n_vertex'], dtype=int)
    train_mask = np.zeros(config['n_vertex'], dtype=float)
    test_mask = np.zeros(config['n_vertex'], dtype=float)
    
    for idx, row in train_labels_csv.iterrows():
        vtx, clas = row.to_list()
        labels[vtx] = clas
        train_mask[vtx] = 1

    for idx, row in test_labels_csv.iterrows():
        vtx, clas = row.to_list()
        labels[vtx] = clas
        test_mask[vtx] = 1

    # config['n_class']=##to be added
        
    return config, features, edge_index, edge_attr, labels, train_mask, test_mask

def split(train_mask, ratio=0.1): # ratio: the ratio of eval set
    tmp = train_mask.copy()
    eval_mask = np.random.rand(train_mask.shape[0]) < ratio
    eval_mask = eval_mask * train_mask
    train_mask -= eval_mask
    assert all(tmp == train_mask + eval_mask)
    return train_mask, eval_mask
