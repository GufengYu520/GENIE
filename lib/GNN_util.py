from dgl.data import DGLDataset
import dgl
from sklearn.preprocessing import Normalizer, StandardScaler
import numpy as np
import torch

def adj_to_weight(adj, attr=0.3):
    snp_count = len(adj)
    # adj = Normalizer().fit_transform(adj)
    weight = []
    u = []
    v = []
    for i in range(snp_count):
        for j in range(i + 1, snp_count):
            if (adj[i][j] + adj[j][i]) > attr:
                weight.append(adj[i][j] + adj[j][i])
                u.append(i)
                v.append(j)

    if u:
        return torch.tensor(np.array(weight), dtype=torch.float32), torch.tensor(np.array(u)), torch.tensor(np.array(v))
    else:
        return torch.tensor(np.array(weight), dtype=torch.float32), u, v


def sample_to_graph(features, adjs, graph_param):
    sample_count = len(features)
    graphs = []

    for i in range(sample_count):
        weight, u, v = adj_to_weight(adjs[i], graph_param['threshold_attr'])
        graph = dgl.graph((u, v), num_nodes=66)
        graph.edata['weight'] = weight
        if graph_param['bidirected']:
            graph = dgl.to_bidirected(graph)
            # u_, v_ = graph.edges()
        if graph_param['self_loop']:
            graph = dgl.add_self_loop(graph)
        if graph_param['nodeType'] == 'onehot':
            graph.ndata['feature'] = torch.tensor(features[i], dtype=torch.float32)
        else:
            graph.ndata['feature'] = torch.tensor(vec_to_node_feat(features[i], graph_param['nodeType']), dtype=torch.float32)

        graphs.append(graph)

    return graphs


def vec_to_node_feat(feat, vecPath='./data/kmer/SNP_9mer_30.npy'):
    vectors = np.load(vecPath)
    snp_count = len(feat)

    node_feat = np.zeros([snp_count, len(vectors[0][0])])

    for i in range(snp_count):
        one_pos = np.where(feat[i] == 1)[0][0]
        if one_pos == 0:
            node_feat[i] = vectors[i][0]
        elif one_pos == 1:
            node_feat[i] = (vectors[i][0] + vectors[i][3])/2
        elif one_pos == 2:
            node_feat[i] = (vectors[i][0] + vectors[i][2])/2
        elif one_pos == 3:
            node_feat[i] = (vectors[i][0] + vectors[i][1])/2
        elif one_pos == 4:
            node_feat[i] = vectors[i][3]
        elif one_pos == 5:
            node_feat[i] = (vectors[i][2] + vectors[i][3])/2
        elif one_pos == 6:
            node_feat[i] = (vectors[i][1] + vectors[i][3])/2
        elif one_pos == 7:
            node_feat[i] = vectors[i][2]
        elif one_pos == 8:
            node_feat[i] = (vectors[i][1] + vectors[i][2])/2
        elif one_pos == 9:
            node_feat[i] = vectors[i][1]


    return node_feat



class SNPDataset(DGLDataset):
    def __init__(self, X, y, adjs, graph_param):
        self.features = X
        self.label = y[:, None]
        self.adjs = adjs
        self.graph_param = graph_param
        self.graphs = sample_to_graph(self.features, self.adjs, self.graph_param)

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)


def getRank(array, sortindex):
    rank = np.zeros(array.shape)
    for i in range(len(sortindex)):
        rank[int(sortindex[i])] = i
    return rank


if __name__ == '__main__':
    test = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    pos = np.where(test == 1)[0][0]
    print(pos)

    print()