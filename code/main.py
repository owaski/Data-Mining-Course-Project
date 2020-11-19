import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import preprocess, split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--test_label_file', type=str, required=True)
    args = parser.parse_args()

    config, features, edges, labels, train_mask, test_mask = preprocess(args.train_data_dir, args.test_label_file)
    train_mask, eval_mask = split(train_mask, ratio=0.1)

    
if __name__ == '__main__':
    main()