import argparse
import numpy as np
from tqdm import tqdm

from main import main

args = argparse.Namespace()
args.model = input('model: ')
args.data = input('data: ')
args.overwrite_cache = False
args.maxepoch = 200
args.additional_features = True if args.data == 'e' else False
args.save = None

max_acc = 0.0
max_args = None

for lr in tqdm(np.arange(-3, -0, 0.2)):
    for n_layer in [2]:
        for weight_decay in np.arange(-4, -1.9, 0.5):
            for drop in np.arange(0.1, 0.5, 0.1):
                for hidden in [64]:
                    args.lr = 10 ** lr
                    args.n_layer = n_layer 
                    args.weight_decay = 10 ** weight_decay      
                    args.drop = drop
                    args.hidden = hidden

                    train_acc, eval_acc, test_acc = main(args)
                    if eval_acc > max_acc:
                        max_acc = eval_acc
                        max_args = args
                        print(train_acc, max_acc, test_acc, max_args)
                    # print(lr, n_layer, weight_decay, drop, hidden)

# GCN
# a: acc = 0.872, lr = 0.0063, drop = 0.2, n_layer = 2, weight_decay = 1e-2
# b: acc = 0.736, lr = 0.0010, drop = 0.1, n_layer = 2, weight_decay = 3e-4
# c: acc = 0.970, lr = 0.0063, drop = 0.1, n_layer = 2, weight_decay = 1e-4
# d: acc = 0.936, lr = 0.0001, drop = 0.1, n_layer = 2, weight_decay = 1e-4
# e: acc = 0.870, lr = 0.0001, drop = 0.1, n_layer = 2, weight_decay = 1e-4

# SAGE
# a: acc = 0.880, lr = 0.0010, drop = 0.4, n_layer = 2, weight_decay = 1e-4
# b: acc = 0.737, lr = 0.00063, drop = 0.4, n_layer = 2, weight_decay = 1e-2
# c: acc = 0.899, lr = 0.0063, drop = 0.4, n_layer = 2, weight_decay = 3e-4
# 