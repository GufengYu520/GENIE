import random

import numpy as np
import pandas as pd
import torch
from dgl.nn.pytorch import GNNExplainer
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import pickle
from tqdm.auto import tqdm

import lib.GNN_util as util

# params
seed = 88
device = 'cuda'
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 0,
          'drop_last': False}

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

same_seed(seed)


class TransferModel(torch.nn.Module):
    def __init__(self, model):
        super(TransferModel, self).__init__()
        self.model = model

    def forward(self, graph, feat, eweight=None):
        x = self.model(graph)
        return x


def explain(model, data):
    # explain_model = TransferModel(model)
    explain_model = model
    explainer = GNNExplainer(explain_model, num_hops=3, alpha2=3)
    g, label = data
    g = g.to(device)
    # print(label.item())
    # print(g.batch_size)

    model.eval()
    # with torch.no_grad():
    #     g = g.to(device)
    #     pred = model(g).item()
    #     if pred < 0:
    #         pred_int = 0
    #     else:
    #         pred_int = 1
    #
    # label = label.item()
    # if pred_int == label:
    #     pred_matrix = [1, label, pred_int]
    # else:
    #     pred_matrix = [0, label, pred_int]

    u, v = g.edges()
    features = g.ndata['feature']
    feat_mask, edge_mask = explainer.explain_graph(g, features)
    return feat_mask, edge_mask, u, v


def edge_mask_to_adj(edge_mask, u, v, snp_count=66):
    edge_mask_adj = np.zeros((snp_count, snp_count))
    for i in range(len(u)):
        edge_mask_adj[u[i]][v[i]] = edge_mask[i]
    return edge_mask_adj


def test_non_zero():
    # 加载数据
    print("Collecting data!")
    X_data, Y_label = np.load('./data/minidata_x_torch.npy'), \
        torch.tensor(np.load('./data/minilabel_y_torch.npy'), dtype=torch.float32)
    score_matrix = np.load('./data/score_matrix_mini_torch.npy')
    print("Data collected!")

    kmer = '7mer_50'
    kmerPath = './data/kmer/SNP_7mer_50.npy'
    graph_param = {
        "nodeType": kmerPath,
        'threshold_attr': 0.3,
        'bidirected': True,
        'self_loop': True,
    }

    raw_dataset_1 = util.SNPDataset(X_data[5000:], Y_label[5000:], score_matrix[5000:], graph_param)
    # raw_dataset_1 = util.SNPDataset(X_data, Y_label, score_matrix, graph_param)
    model = torch.load('./output/models/model_fold1_epoch64_0.7322.pth', map_location=device)
    model.eval()

    params_test = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0,
              'drop_last': False}

    data_loader = GraphDataLoader(raw_dataset_1, **params_test)
    y_pred = []
    y_label = []
    correct_non_zero_list = []
    cnt = 0
    with torch.no_grad():
        for batched_graph, labels in tqdm(data_loader):
            batched_graph, labels = batched_graph.to(device), labels.to(device)
            logits = model(batched_graph, batched_graph.ndata['feature'])
            y_pred = y_pred + logits.flatten().tolist()
            y_label = y_label + labels.flatten().tolist()
            pred_int = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0)])
            if int(logits.flatten()) > 0 and batched_graph.num_edges() > 66:
                correct_non_zero_list.append(cnt+5000)
            cnt = cnt + 1


    f1_val_test = f1_score(y_label, pred_int)
    acc_val_test = accuracy_score(y_label, pred_int)

    correct_non_zero_1 = {
        'list': correct_non_zero_list
    }
    with open('./data/explain_result_2023/correct_non_zero_1.pkl'.format(kmer), 'wb') as f:
        pickle.dump(correct_non_zero_1, f)


    print()



def main():
    # 加载数据
    print("Collecting data!")
    X_data, Y_label = np.load('./data/minidata_x_torch.npy'), \
                      torch.tensor(np.load('./data/minilabel_y_torch.npy'), dtype=torch.float32)
    score_matrix = np.load('./data/score_matrix_mini_torch.npy')
    print("Data collected!")


    kmer = '7mer_50'
    kmerPath = './data/kmer/SNP_7mer_50.npy'
    graph_param = {
        "nodeType": kmerPath,
        'threshold_attr': 0.3,
        'bidirected': True,
        'self_loop': True,
    }

    with open('./data/explain_result_2023/correct_non_zero_1.pkl', 'rb') as f:
        tmp = pickle.load(f)
        correct_non_zero_list = tmp['list']


    raw_dataset_1 = util.SNPDataset(X_data[correct_non_zero_list], Y_label[correct_non_zero_list], score_matrix[correct_non_zero_list], graph_param)
    model = torch.load('./output/models/model_fold1_epoch64_0.7322.pth', map_location=device)
    model.eval()

    # Explain the prediction for 1st graph
    # feat_mask, edge_mask = explain(model, raw_dataset[0])
    data_loader = GraphDataLoader(raw_dataset_1, **params)

    edge_mask_adjs_1 = np.zeros(raw_dataset_1.adjs.shape)
    pred_matrixes = []
    i = 0
    for batch_data in data_loader:
        print(i)
        feat_mask, edge_mask, u, v = explain(model, batch_data)
        u, v = u.tolist(), v.tolist()
        feat_mask, edge_mask = feat_mask.cpu().numpy(), edge_mask.cpu().numpy()
        edge_mask_adj = edge_mask_to_adj(edge_mask, u, v)
        edge_mask_adjs_1[i] = edge_mask_adj
        # pred_matrixes.append(pred_matrix)
        # if i == 0:
        #     feat_masks = feat_mask
        #     edge_masks = edge_mask
        # else:
        #     feat_masks = feat_masks + feat_mask
        #     edge_masks = torch.cat((edge_masks, edge_mask), dim=0)
        i = i + 1

    # np.save('./data/explain_result/feat_mask', (feat_masks/i).numpy())
    # np.save('./data/explain_result/edge_mask', edge_masks.numpy())
    explain_result = {
        'edge_mask_adjs': edge_mask_adjs_1
    }
    with open('./data/explain_result_2023/explain_{}.pkl'.format(kmer), 'wb') as f:
        pickle.dump(explain_result, f)

    print()



def test_svm(topEdge_index, k=5, node=False):

    kmer = '7mer_50'

    # 加载数据
    print("Collecting data!")
    Y_label = np.load('./data/minilabel_y_torch.npy')
    score_matrix = np.load('./data/score_matrix_mini_torch.npy')
    print("Data collected!")

    topNum = len(topEdge_index[:k])

    top_u = []
    top_v = []

    if node:
        for i in range(topNum):
            top_u.append(topEdge_index[i])
            top_v.append(topEdge_index[i])
    else:
        for i in range(topNum):
            top_u.append(topEdge_index[i][0])
            top_v.append(topEdge_index[i][1])
    top_edge_weight_z = score_matrix[:, top_u, top_v]
    top_edge_weight_t = score_matrix[:, top_v, top_u]

    top_edge_weight = top_edge_weight_z + top_edge_weight_t



    Kfold_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    model = SVC(random_state=seed, class_weight={0: 1, 1: 2})
    cv_result = cross_validate(model, top_edge_weight, Y_label, cv=Kfold_split, n_jobs=4)
    cv_score = cross_val_score(model, top_edge_weight, Y_label, cv=Kfold_split)
    cv_predict = cross_val_predict(model, top_edge_weight, Y_label, cv=Kfold_split)

    avg_acc = np.mean(cv_score)


    # print("AVG acc: {:.4f}, AVG f1: {:.4f}, AVG AUC: {:.4f}".format(avg_acc, avg_f1, avg_AUC))

    print("finished!")
    return avg_acc


def train_random(top_index, n=100, k=5, node=False):
    acc_list = []
    for i in range(n):
        random_top = random.sample(list(top_index), k)
        acc = test_svm(random_top, k, node)
        acc_list.append(acc)

    return acc_list




def getNetwork(topNum, Num_0=None):
    if Num_0:
        edge_mask = np.load('data/explain_result/edge_mask_single.npy')[Num_0:]
    else:
        edge_mask = np.load('data/explain_result/edge_mask_single.npy')
    edge_mask_sortindex = np.zeros(edge_mask.shape)
    edge_mask_rank = np.zeros(edge_mask.shape)

    for i in range(len(edge_mask)):
        edge_mask_sortindex[i] = np.argsort(-edge_mask[i])
        edge_mask_rank[i] = util.getRank(edge_mask[i], edge_mask_sortindex[i])

    edge_mask_rank_sum = np.sum(edge_mask_rank, axis=0)
    highlight_edge = np.argsort(edge_mask_rank_sum)[:topNum]

    u = []
    v = []
    h_u_SNP, h_u_Gene = [], []
    h_v_SNP, h_v_Gene = [], []
    SNP_Gene = pd.read_excel('./data/explain_result/SNP_Gene.xlsx')
    for i in range(66):
        for j in range(i + 1, 66):
            u.append(i)
            v.append(j)
    for i in range(len(highlight_edge)):
        h_u_SNP.append(SNP_Gene.loc[u[highlight_edge[i]], 'SNP'])
        h_u_Gene.append(SNP_Gene.loc[u[highlight_edge[i]], 'GENE'])
        h_v_SNP.append(SNP_Gene.loc[v[highlight_edge[i]], 'SNP'])
        h_v_Gene.append(SNP_Gene.loc[v[highlight_edge[i]], 'GENE'])

    highlight_node = pd.DataFrame(columns=['SNP1', 'Gene1', 'SNP2', 'Gene2', 'rank'])
    highlight_node.loc[:, 'SNP1'] = h_u_SNP
    highlight_node.loc[:, 'Gene1'] = h_u_Gene
    highlight_node.loc[:, 'SNP2'] = h_v_SNP
    highlight_node.loc[:, 'Gene2'] = h_v_Gene
    highlight_node.loc[:, 'rank'] = list(range(len(highlight_edge)))
    highlight_node.to_csv('./data/explain_result/highlight_SNP_0&1.csv', index=False)


def network_analysis():
    kmer = '7mer_50'

    with open('./data/explain_result_2023/explain_7mer_50.pkl', 'rb') as f:
        explain_result = pickle.load(f)

    edge_mask_adjs = explain_result["edge_mask_adjs"]
    edge_mask = np.sum(edge_mask_adjs, axis=0)
    top_node_mask = []
    tmp = []
    for i in range(66):
        top_node_mask.append(edge_mask[i][i])
        tmp.append(67*i)

    edge_mask_vector = []
    for i in range(66):
        for j in range(i+1, 66):
            edge_mask_vector.append(edge_mask[i][j]+edge_mask[j][i])


    top_edge_sort_index = np.argsort(np.array(edge_mask_vector))[::-1]
    top_node_sort_index = np.argsort(np.array(top_node_mask))[::-1]

    top_edge_sort_index_group = []

    for i in top_edge_sort_index:
        tmp_x = i//66
        tmp_y = i-tmp_x*66
        top_edge_sort_index_group.append((tmp_x, tmp_y))


    explain_result = {
        'node_mask': top_node_sort_index,
        'edge_mask': top_edge_sort_index_group
    }
    with open('./data/explain_result_2023/explain_{}_node_edge.pkl'.format(kmer), 'wb') as f:
        pickle.dump(explain_result, f)


    print()




if __name__ == '__main__':
    # getNetwork(50)
    # with open('./data/explain_result_2023/explain_9mer_30.pkl', 'rb') as f:
    #     explain_result = pickle.load(f)
    # topEdge_index = [(23, 45), (34, 61), (10, 15), (16, 27), (6, 58)]
    with open('./data/explain_result_2023/explain_7mer_50_node_edge.pkl', 'rb') as f:
        explain_result = pickle.load(f)
        topEdge_index = explain_result["edge_mask"]
        topNode_index = explain_result['node_mask']

    # acc_list_5 = train_random(topEdge_index)
    # acc_list_10 = train_random(topEdge_index, k=10)
    # acc_list_30 = train_random(topEdge_index, k=30)
    # acc_list_50 = train_random(topEdge_index, k=50)
    #
    # explain_result = {
    #     '5': acc_list_5,
    #     '10': acc_list_10,
    #     '30': acc_list_30,
    #     '50': acc_list_50
    # }
    # with open('./data/explain_result_2023/topN_edge_acc.pkl', 'wb') as f:
    #     pickle.dump(explain_result, f)

    acc_list_5_node = train_random(topNode_index, node=True)
    acc_list_10_node = train_random(topNode_index, k=10, node=True)
    acc_list_15_node = train_random(topNode_index, k=15, node=True)

    explain_result = {
        '5': acc_list_5_node,
        '10': acc_list_10_node,
        '15': acc_list_15_node
    }
    with open('./data/explain_result_2023/topN_node_acc.pkl', 'wb') as f:
        pickle.dump(explain_result, f)

    # avg_acc_5 = test_svm(topEdge_index)
    # avg_acc_10 = test_svm(topEdge_index, k=10)
    # avg_acc_30 = test_svm(topEdge_index, k=30)
    # avg_acc_50 = test_svm(topEdge_index, k=50)
    # avg_acc_5_node = test_svm(topNode_index, node=True)
    # avg_acc_10_node = test_svm(topNode_index, k=10, node=True)
    # avg_acc_15_node = test_svm(topNode_index, k=15, node=True)
    # test_non_zero()
    # main()
    # network_analysis()
    print()

