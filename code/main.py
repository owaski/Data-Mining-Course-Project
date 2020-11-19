import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import preprocess, split
from models import GCNNet
from torch_geometric.data import Data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--test_label_file', type=str, required=True)
    parser.add_argument('--maxepoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()

    config, features, edge_index, edge_attr, labels, train_mask, test_mask = preprocess(args.train_data_dir, args.test_label_file)
    train_mask, eval_mask = split(train_mask, ratio=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNNet(config['n_feature'],config['n_class']).to(device)
    data = Data(x=features,edge_index=edge_index, edge_attr=edge_attr, y=labels, train_mask=train_mask,test_mask=test_mask, eval_mask = eval_mask)##to be modified
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for epoch in range(args.maxepoch):
        print("range"+epoch)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    
if __name__ == '__main__':
    main()