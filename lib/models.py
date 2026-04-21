import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dgl.nn.pytorch import AvgPooling, GATConv, GraphConv, GCN2Conv, SAGEConv, EdgeWeightNorm

def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


class GATModel(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim_1=32, hidden_dim_2=64,
                 pred_hidden_dim=64, dropout=0.2):
        super(GATModel, self).__init__()
        # GAT
        self.GATconv1 = GATConv(input_dim, hidden_dim_1, num_heads=3)
        self.GATconv2 = GATConv(3*hidden_dim_1, hidden_dim_2, num_heads=3)

        self.gap = AvgPooling()

        self.dropout = dropout

        self.decoder1 = nn.Linear(3*hidden_dim_2, pred_hidden_dim)
        self.decoder2 = nn.Linear(pred_hidden_dim, 1)

        self.predict = nn.Sequential(
            self.decoder1,
            nn.ReLU(),
            self.decoder2
            # nn.Sigmoid()
        )

    def forward(self, g):
        # g = getGraph(adj)

        x1 = F.relu(self.GATconv1(g, g.ndata['feature']))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = torch.flatten(x1, 1)
        x2 = F.relu(self.GATconv2(g, x1))
        x2 = torch.flatten(x2, 1)

        # global average pooling
        x2 = self.gap(g, x2)
        o = self.predict(x2)

        return o


class GCNModel(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim_1=32, hidden_dim_2=64, gnn_dim_1=128,
                 gnn_dim_2=256, pred_hidden_dim=32, dropout=0.2):
        super(GCNModel, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout

        # GCN
        self.conv_1 = GraphConv(hidden_dim_2, gnn_dim_1)
        self.conv_2 = GraphConv(gnn_dim_1, gnn_dim_2)
        self.gap = AvgPooling()

        self.encoder1 = nn.Linear(input_dim, hidden_dim_1)
        self.encoder2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.encoder = nn.Sequential(
            self.encoder1,
            nn.ReLU(),
            self.encoder2,
            nn.ReLU(),
        )
        self.decoder1 = nn.Linear(gnn_dim_2, pred_hidden_dim)
        self.decoder2 = nn.Linear(pred_hidden_dim, 1)

        self.bn1 = nn.BatchNorm1d(gnn_dim_1)
        self.bn2 = nn.BatchNorm1d(gnn_dim_2)

        self.predict = nn.Sequential(
            self.decoder1,
            nn.ReLU(),
            self.decoder2
            # nn.Sigmoid()
        )

    def forward(self, graph, feat, eweight=None):
        x = self.encoder(feat)
        x = F.relu(self.bn1(self.conv_1(graph, x, edge_weight=eweight)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv_2(graph, x, edge_weight=eweight)))

        # global average pooling
        x = self.gap(graph, x)
        o = self.predict(x)

        return o


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, gnn_dim_1=128,
                 gnn_dim_2=128, pred_hidden_dim=32, dropout=0.2):
        super(GraphSAGEModel, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout

        # SAGE
        self.conv_1 = SAGEConv(hidden_dim, gnn_dim_1, 'mean')
        self.conv_2 = SAGEConv(gnn_dim_1, gnn_dim_2, 'mean')
        self.gap = AvgPooling()

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.decoder1 = nn.Linear(gnn_dim_2, pred_hidden_dim)
        self.decoder2 = nn.Linear(pred_hidden_dim, 1)

        self.bn1 = nn.BatchNorm1d(gnn_dim_1)
        self.bn2 = nn.BatchNorm1d(gnn_dim_2)

        self.predict = nn.Sequential(
            self.decoder1,
            nn.ReLU(),
            self.decoder2,
            # nn.Sigmoid()
        )


    def forward(self, graph, feat, eweight=None):
        x = F.relu(self.encoder(feat))
        x = F.relu(self.bn1(self.conv_1(graph, x, edge_weight=eweight)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv_2(graph, x, edge_weight=eweight)))

        # global average pooling
        x = self.gap(graph, x)
        o = self.predict(x)

        return o



class GCN2Model(torch.nn.Module):
    def __init__(self, input_dim=16, hidden_dim_1=16, hidden_dim_2=16,
                 pred_hidden_dim=32, dropout=0.2):
        super(GCN2Model, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout

        # GCN2
        self.conv_1 = GCN2Conv(input_dim, layer=1, alpha=0.5)
        self.conv_2 = GCN2Conv(hidden_dim_1, layer=2, alpha=0.5)
        self.gap = AvgPooling()

        self.decoder1 = nn.Linear(hidden_dim_2, pred_hidden_dim)
        self.decoder2 = nn.Linear(pred_hidden_dim, 1)

        self.predict = nn.Sequential(
            self.decoder1,
            nn.ReLU(),
            self.decoder2,
            nn.Sigmoid()
        )

    def forward(self, g):
        g = dgl.add_self_loop(g)

        x1 = F.relu(self.conv_1(g, g.ndata['feature'], g.ndata['feature']))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = F.relu(self.conv_2(g, x1, g.ndata['feature']))
        # x2 = torch.flatten(x2, 1)

        # global average pooling
        x2 = self.gap(g, x2)
        o = self.predict(x2)

        return o


class ClassifierModel(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim_1=16, hidden_dim_2=32,
                 pred_hidden_dim=16, dropout=0.2, kernal_size=5):
        super(ClassifierModel, self).__init__()
        self.encoder = nn.Linear(input_dim, input_dim, bias=False)

        self.conv1d_1 = nn.Conv1d(input_dim, hidden_dim_1, kernel_size=kernal_size)
        self.conv1d_2 = nn.Conv1d(hidden_dim_1, hidden_dim_2, kernel_size=kernal_size)
        self.decoder1 = nn.Linear(hidden_dim_2, pred_hidden_dim)
        self.decoder2 = nn.Linear(pred_hidden_dim, 1)

        self.conv = nn.Sequential(
            self.conv1d_1,
            nn.ReLU(),
            self.conv1d_2,
            nn.ReLU(),
        )
        self.predict = nn.Sequential(
            self.decoder1,
            nn.ReLU(),
            nn.Dropout(dropout),
            self.decoder2,
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = nn.AdaptiveMaxPool1d(1)(x)
        x = torch.squeeze(x)
        o = self.predict(x)

        return o


if __name__ == '__main__':
    testdata = torch.randn(16, 66, 10)
    model = ClassifierModel()
    output = model(testdata)
    print()