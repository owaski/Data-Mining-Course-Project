import os
import sys
import argparse
from tqdm import tqdm

import json

import networkx as nx
import node2vec

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import preprocess, split, accuracy, DATASET_DIR
from models import FCNet, EdgeNet, GCNNet, GCN_Linear, GATNet, TAGNet, SAGENet
from torch_geometric.data import Data

MODEL_CLASS = {
    'FC' : FCNet,
    'Edge' : EdgeNet,
    'GCN' : GCNNet,
    'GCNL' : GCN_Linear,
    'GAT' : GATNet,
    'TAG' : TAGNet,
    'SAGE' : SAGENet,
}

def get_additional_features(config, edge_index, edge_attr, args):
    data = torch.sparse.FloatTensor(edge_index, edge_attr.squeeze(1))
    features = [[] for i in range(config['n_vertex'])]

    # for i in range(data.size(0)):
    #     features.append(data[i].to_dense().cpu().tolist())

    if args.node2vec:
        cache_path = os.path.join(DATASET_DIR, args.data, 'embedding.pt')
        if not os.path.exists(cache_path) or args.overwrite_cache:
            nx_G = node2vec.read_graph(config, edge_index, edge_attr)
            G = node2vec.Node2Vec(nx_G, True, 1., 1., args.verbose)
            G.preprocess_transition_probs()
            walks = G.simulate_walks(40, 10)
            embedding = node2vec.learn_embeddings(walks)
            embeddings = []
            for i in range(config['n_vertex']):
                embeddings.append(embedding.wv[str(i)].tolist())
            torch.save(embeddings, cache_path)
        else:
            embeddings = torch.load(cache_path)

        for i in range(config['n_vertex']):
            features[i] += embeddings[i]

    return torch.tensor(features).float()


def main(args):
    cache_path = os.path.join(DATASET_DIR, args.data, 'cache.pt')
    if not os.path.exists(cache_path) or args.overwrite_cache:
        config, features, edge_index, edge_attr, labels, train_mask, test_mask = preprocess(args.data)
        train_mask, eval_mask = split(train_mask, ratio=0.1)
        torch.save([config, features, edge_index, edge_attr, labels, train_mask, eval_mask, test_mask], cache_path)
    else:
        config, features, edge_index, edge_attr, labels, train_mask, eval_mask, test_mask = torch.load(cache_path)

    if args.additional_features:
        additional_features = get_additional_features(config, edge_index, edge_attr, args)
        features = torch.cat([features, additional_features], dim=-1)
        config['n_feature'] = features.size(1)

    data = Data(x=features,edge_index=edge_index, edge_attr=edge_attr, y=labels, train_mask=train_mask, test_mask=test_mask, eval_mask=eval_mask)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_class = MODEL_CLASS[args.model]
    model = model_class(num_feature=max(1, config['n_feature']), num_class=config['n_class'], num_layers=args.n_layer, hidden=args.hidden, drop=args.drop).to(device)
    model.reset_parameters()
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    max_eval_acc = 0.0
    max_eval_epoch = -1
    iterator = tqdm(range(args.maxepoch)) if args.verbose else range(args.maxepoch)
    for epoch in iterator:
        # print("range"+epoch)
        optimizer.zero_grad()
        out1 = model(data)
        # print(out, out[data.train_mask], data.y[data.train_mask])
        loss = F.nll_loss(out1[data.train_mask], data.y[data.train_mask])
        train_acc = accuracy(out1[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out2 = model(data)
            cur_eval_acc = accuracy(out2[data.eval_mask], data.y[data.eval_mask])
            if cur_eval_acc > max_eval_acc:
                max_eval_acc = cur_eval_acc
                max_eval_epoch = epoch
            elif epoch - max_eval_epoch >= 10:
                break
            model.train()
    
    with torch.no_grad():
        model.eval()
        out3 = model(data)
        train_acc = accuracy(out3[data.train_mask], data.y[data.train_mask])

    with torch.no_grad():
        model.eval()
        out3 = model(data)
        test_acc = accuracy(out3[data.test_mask], data.y[data.test_mask])

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        torch.save(model, os.path.join(args.save, 'model.pt'))
        with open(os.path.join(args.save, 'config.json'), 'w') as w:
            json.dump(vars(args), w, indent='\t')

    return train_acc, cur_eval_acc, test_acc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['FC', 'Edge', 'GCN', 'GCNL', 'GAT', 'SAGE', 'TAG'])
    parser.add_argument('--save', type=str, default=None) # save dir
    parser.add_argument('--data', type=str, choices=['a', 'b', 'c', 'd', 'e'])
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--maxepoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--additional_features', action='store_true')
    parser.add_argument('--node2vec', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(main(args))