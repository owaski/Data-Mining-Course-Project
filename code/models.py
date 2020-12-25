import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GCNNet(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(GCNNet, self).__init__()
        self.conv0 = GCNConv(num_feature, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, num_class)
        self.n_layer = num_layers
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        for i in range(self.n_layer - 1):
            conv = self.conv0 if i == 0 else self.conv1
            x = conv(x, edge_index, edge_weight) if self.use_edge_weight else \
                conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index, edge_weight) if self.use_edge_weight else \
            self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN_Linear(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(GCN_Linear, self).__init__()
        self.conv0 = GCNConv(num_feature, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.n_layer = num_layers
        self.linear = Linear(hidden, num_class)
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        for i in range(self.n_layer):
            conv = self.conv0 if i == 0 else self.conv1
            x = conv(x, edge_index, edge_weight) if self.use_edge_weight else \
                conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class GATNet(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(GATNet, self).__init__()
        self.conv0 = GATConv(num_feature, hidden)
        self.conv1 = GATConv(hidden, hidden)
        self.conv2 = GATConv(hidden, num_class)
        self.n_layer = num_layers
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        for i in range(self.n_layer - 1):
            conv = self.conv0 if i == 0 else self.conv1
            x = conv(x, edge_index, edge_weight) if self.use_edge_weight else \
                conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index, edge_weight) if self.use_edge_weight else \
            self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT_Linear(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(GAT_Linear, self).__init__()
        self.conv0 = GATConv(num_feature, hidden)
        self.conv1 = GATConv(hidden, hidden)
        self.n_layer = num_layers
        self.linear = Linear(hidden, num_class)
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        for i in range(self.n_layer):
            conv = self.conv0 if i == 0 else self.conv1
            x = conv(x, edge_index, edge_weight) if self.use_edge_weight else \
                conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class SAGENet(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(SAGENet, self).__init__()
        self.conv0 = SAGEConv(num_feature, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, num_class)
        self.n_layer = num_layers
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        for i in range(self.n_layer - 1):
            conv = self.conv0 if i == 0 else self.conv1
            x = conv(x, edge_index, edge_weight) if self.use_edge_weight else \
                conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index, edge_weight) if self.use_edge_weight else \
            self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class SAGE_Linear(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(SAGE_Linear, self).__init__()
        self.conv0 = SAGEConv(num_feature, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.n_layer = num_layers
        self.linear = Linear(hidden, num_class)
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        for i in range(self.n_layer):
            conv = self.conv0 if i == 0 else self.conv1
            x = conv(x, edge_index, edge_weight) if self.use_edge_weight else \
                conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)
