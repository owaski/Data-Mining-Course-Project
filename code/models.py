import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNNet(torch.nn.Module, num_feature, num_class):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_feature, 16)
        self.conv2 = GCNConv(16, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

