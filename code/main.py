import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import preprocess, split, accuracy, DATASET_DIR
from models import GCNNet
from torch_geometric.data import Data

def get_additional_features(config, edge_index, edge_attr):
    data = torch.sparse.FloatTensor(edge_index, edge_attr.squeeze(1))
    features = []
    for i in range(data.size(0)):
        features.append(data[i].to_dense().cpu().tolist())
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
        additional_features = get_additional_features(config, edge_index, edge_attr)
        features = torch.cat([features, additional_features], dim=-1)
        config['n_feature'] = features.size(1)

    data = Data(x=features,edge_index=edge_index, edge_attr=edge_attr, y=labels, train_mask=train_mask, test_mask=test_mask, eval_mask=eval_mask)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNNet(num_feature=max(1, config['n_feature']), num_class=config['n_class'], num_layers=args.n_layer, hidden=args.hidden, drop=args.drop).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    max_eval_acc = 0.0
    max_eval_epoch = -1
    for epoch in range(args.maxepoch):
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
    
    with torch.no_grad():
        model.eval()
        out3 = model(data)
        test_acc = accuracy(out3[data.test_mask], data.y[data.test_mask])

    return train_acc, cur_eval_acc, test_acc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['a', 'b', 'c', 'd', 'e'])
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--maxepoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--additional_features', action='store_true')
    args = parser.parse_args()

    print(main(args))