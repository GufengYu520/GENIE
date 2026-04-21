import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import copy
import pickle

from lib.models import ClassifierModel
from lib.OxfordDataGenerator import OxfordDataGenerator_train_test

# params
n_random_state = 88
seed = 88
epochs = 100
batch_size = 16
lr = 0.001
weight_decay = 0
dropout = 0
device = 'cuda:1'

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

same_seed(seed)

# Model and optimizer
model = ClassifierModel(dropout=dropout, kernal_size=3).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

loss_fct = torch.nn.BCELoss(weight=torch.tensor([4]).to(device))
loss_fct_train = torch.nn.BCELoss(weight=torch.tensor([2]).to(device))

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0,
          'drop_last': True}

same_seed(seed)

def compute_importance(model, sequences):
    params_attr = {'batch_size': 16,
              'num_workers': 0,
              'drop_last': False}
    baseline = np.array([1/16, 1/8, 1/8, 1/8, 1/16, 1/8, 1/8, 1/16, 1/8, 1/16])
    baselines = torch.tensor(np.expand_dims(baseline, 0).repeat(66, axis=0)).to(device).float()
    baselines = torch.unsqueeze(baselines, 0)
    model.eval()
    ig = IntegratedGradients(model)

    sequences = torch.tensor(sequences).float()
    data_loader = DataLoader(sequences, **params_attr)
    n = 0
    print("Start computing importance:")
    for batch_seqs in tqdm(data_loader):
        batch_seqs = batch_seqs.to(device)
        batch_attribution = ig.attribute(batch_seqs, baselines)
        if n == 0:
            attribution = torch.sum(batch_attribution, dim=2)
        else:
            attribution = torch.cat([attribution, torch.sum(batch_attribution, dim=2)], dim=0)
        n = n + 1
        # print(attribution.shape)
    # sequences = torch.tensor(sequences).to(device).float()
    # attribution = ig.attribute(sequences, baselines)

    return attribution.cpu().numpy()

def train(model, data, optimizer):
    train_loader = DataLoader(data, **params)

    model.train()

    y_pred = []
    y_label = []
    loss_history = 0
    cnt = 0

    for batched_data, labels in tqdm(train_loader):
        batched_data, labels = batched_data.to(device).float(), labels.to(device).float()
        logits = model(batched_data)
        loss = loss_fct(logits.flatten(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = y_pred + logits.flatten().tolist()
        y_label = y_label + labels.flatten().tolist()
        pred_int = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        loss_history = loss_history + loss.item()

        cnt = cnt + 1

    # roc_train = roc_auc_score(y_label, y_pred)
    f1_train = f1_score(y_label, pred_int)
    acc_train = accuracy_score(y_label, pred_int)

    return loss_history / cnt, f1_train, acc_train



def getTopSNP(X_data):
    sample_count = len(X_data)

    model_path = './output/models/model_test.pth'
    model = torch.load(model_path, map_location=device)
    # model = torch.nn.DataParallel(model)
    scores = compute_importance(model, X_data)
    print(scores.shape)

    abs_scores = np.abs(scores)
    sum_scores = np.sum(abs_scores, axis=0)

    sort_index_list = np.argsort(sum_scores)[::-1]

    return sort_index_list

def train_topN(topN):
    # 加载数据
    print("Collecting data!")
    X_data, Y_label = np.load('./data/minidata_x_torch.npy'), np.load('./data/minilabel_y_torch.npy')
    print("Data collected!")

    random_list = range(66)

    max_acc_list = []

    print("topN SNP: {}".format(topN))
    for i in range(100):
        print("No.{}".format(i+1))
        random_top_SNP = random.sample(random_list, topN)
        max_acc = 0.6
        model_f = copy.deepcopy(model)
        optimizer_f = optim.Adam(model_f.parameters(), lr=lr, weight_decay=weight_decay)

        X_data_N = X_data[:, random_top_SNP]

        traindata = OxfordDataGenerator_train_test(X_data_N, Y_label)

        for epoch in range(epochs):
            print('-------- Epoch ' + str(epoch + 1) + ' --------')
            loss_train, f1_train, acc_train = train(model_f, traindata, optimizer_f)
            print('Training Loss: {:.4f}'.format(loss_train))
            print('Training f1: {:.4f}'.format(f1_train))
            print('Training acc: {:.4f}'.format(acc_train))

            if acc_train > max_acc:
                max_acc = acc_train

        max_acc_list.append(max_acc)

    return max_acc_list






def train_topN_raw():
    # 加载数据
    print("Collecting data!")
    X_data, Y_label = np.load('./data/minidata_x_torch.npy'), np.load('./data/minilabel_y_torch.npy')
    print("Data collected!")

    top15_list = [10, 36, 13, 21, 12, 26, 22, 14, 9, 15, 34, 35, 18, 53, 28]

    max_acc_list = []

    for N in [5, 10, 15]:
        max_acc = 0.6
        print("top{} SNP: ".format(N))
        model_f = copy.deepcopy(model)
        optimizer_f = optim.Adam(model_f.parameters(), lr=lr, weight_decay=weight_decay)

        X_data_N = X_data[:, top15_list[:N]]

        traindata = OxfordDataGenerator_train_test(X_data_N, Y_label)

        for epoch in range(epochs):
            print('-------- Epoch ' + str(epoch + 1) + ' --------')
            loss_train, f1_train, acc_train = train(model_f, traindata, optimizer_f)
            print('Training Loss: {:.4f}'.format(loss_train))
            print('Training f1: {:.4f}'.format(f1_train))
            print('Training acc: {:.4f}'.format(acc_train))

            if acc_train > max_acc:
                max_acc = acc_train

        max_acc_list.append(max_acc)

    print(max_acc_list)





def main():
    # 加载数据
    print("Collecting data!")
    X_data, Y_label = np.load('./data/minidata_x_torch.npy'), np.load('./data/minilabel_y_torch.npy')
    print("Data collected!")

    sort_index_list = getTopSNP(X_data)
    top5_list = sort_index_list[:5]
    top10_list = sort_index_list[:10]
    top15_list = sort_index_list[:15]

    top15_list = [10, 36, 13, 21, 12, 26, 22, 14, 9, 15, 34, 35, 18, 53, 28]

    print()





if __name__ == '__main__':
    # main()
    # train_topN_raw()
    max_acc_list_5 = train_topN(5)
    max_acc_list_10 = train_topN(10)
    max_acc_list_15 = train_topN(15)
    max_acc_result = {
        "5": max_acc_list_5,
        "10": max_acc_list_10,
        "15": max_acc_list_15
    }
    with open('./data/topN_SNP_acc.pkl', 'wb') as f:
        pickle.dump(max_acc_result, f)
    print()