import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TAGConv

class FCNet(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5):
        super(FCNet, self).__init__()
        self.linear0 = nn.Linear(num_feature, hidden)
        self.linear1s = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(num_layers - 2)])
        self.linear2 = nn.Linear(hidden, num_class)
        self.n_layer = num_layers
        self.drop = drop

    def reset_parameters(self):
        nn.init.normal_(self.linear0.weight)
        nn.init.normal_(self.linear0.bias)
        for linear in self.linear1s:
            nn.init.normal_(linear.weight)
            nn.init.normal_(linear.bias)
        nn.init.normal_(self.linear2.weight)
        nn.init.normal_(self.linear2.bias)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        x = self.linear0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        for linear in self.linear1s:
            x = linear(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.linear2(x)

        return F.log_softmax(x, dim=1)


class EdgeNet(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(EdgeNet, self).__init__()
        self.conv0 = GCNConv(num_feature, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.linear = nn.Linear(hidden, num_class)
        self.n_layer = num_layers
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.normal_(self.linear.weight)
        nn.init.normal_(self.linear.bias)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        x = torch.zeros_like(x)

        for i in range(self.n_layer - 1):
            conv = self.conv0 if i == 0 else self.conv1
            x = conv(x, edge_index, edge_weight) if self.use_edge_weight else \
                conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index, edge_weight) if self.use_edge_weight else \
            self.conv2(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)


class GCNNet(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(GCNNet, self).__init__()
        self.conv0 = GCNConv(num_feature, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.linear = nn.Linear(hidden, num_class)
        self.n_layer = num_layers
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.normal_(self.linear.weight)
        nn.init.normal_(self.linear.bias)

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

        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.linear(x)

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
        self.conv0 = GATConv(num_feature, hidden, heads=8, dropout=drop, concat=False)
        self.conv1 = GATConv(hidden, hidden, heads=8, dropout=drop, concat=False)
        self.conv2 = GATConv(hidden, num_class, heads=8, dropout=drop, concat=False)
        self.n_layer = num_layers
        self.use_edge_weight = use_edge_weight
        self.drop = drop

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data): # TODO: edge weight
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze(1)

        for i in range(self.n_layer - 1):
            conv = self.conv0 if i == 0 else self.conv1
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT_Linear(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(GAT_Linear, self).__init__()
        self.conv0 = GATConv(num_feature, hidden,heads=8)
        self.conv1 = GATConv(hidden, hidden,heads=8)
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
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index)

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

class TAGNet(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, k=3, hidden=64, drop=0.5, use_edge_weight=True):
        super(TAGNet, self).__init__()
        self.conv0 = TAGConv(num_feature, hidden, K=k)
        self.conv1 = TAGConv(hidden, hidden, K=k)
        self.conv2 = TAGConv(hidden, num_class, K=k)
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


class TAG_Linear(nn.Module):
    def __init__(self, num_feature, num_class, num_layers=2, hidden=64, drop=0.5, use_edge_weight=True):
        super(TAG_Linear, self).__init__()
        self.conv0 = TAGConv(num_feature, hidden,K=K)
        self.conv1 = TAGConv(hidden, hidden,K=K)
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
