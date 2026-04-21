from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import copy
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from lib.models import ClassifierModel
from lib.OxfordDatasetAll import OxfordDatasetAll_v2
from lib.OxfordDataGenerator import OxfordDataGenerator, OxfordDataGenerator_train_test


# params
n_random_state = 88
seed = 88
epochs = 500
batch_size = 16
lr = 0.001
weight_decay = 5e-4
dropout = 0.05
device = 'cuda'

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

same_seed(seed)

# Model and optimizer
model = ClassifierModel(dropout=dropout).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

loss_fct = torch.nn.BCELoss()
loss_fct_train = torch.nn.BCELoss(weight=torch.tensor([2]).to(device))

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}


def train(model, data):
    train_loader = DataLoader(data, **params)

    model.train()

    y_pred = []
    y_label = []
    loss_history = 0
    cnt = 0

    for batched_data, labels in tqdm(train_loader):
        batched_data, labels = batched_data.to(device).float(), labels.to(device).float()
        logits = model(batched_data)
        loss = loss_fct_train(logits.flatten(), labels)
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


def val_test(model, data):
    val_test_dataloader = DataLoader(data, **params)

    model.eval()

    y_pred = []
    y_label = []
    loss_history = 0
    cnt = 0

    with torch.no_grad():
        for batched_data, labels in tqdm(val_test_dataloader):
            batched_data, labels = batched_data.to(device).float(), labels.to(device).float()
            logits = model(batched_data)
            loss = loss_fct(logits.flatten(), labels)

            y_pred = y_pred + logits.flatten().tolist()
            y_label = y_label + labels.flatten().tolist()
            pred_int = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
            loss_history = loss_history + loss.item()

            cnt = cnt + 1



    # roc_val_test = roc_auc_score(y_label, y_pred)
    f1_val_test = f1_score(y_label, pred_int)
    acc_val_test = accuracy_score(y_label, pred_int)

    return loss_history/cnt, f1_val_test, acc_val_test

def main():
    max_acc = 0.8
    model_max = copy.deepcopy(model)

    # logs
    writer = SummaryWriter('./output/record/logs')

    # load data
    print("Collecting data!")
    data_path = './data/oxford_imputed_ad.npz'
    outstanding_path = './data/snps_oxford'
    original_train_dataset = OxfordDatasetAll_v2(data_path, outstanding_path)
    print("Data collected!")

    # 0.8, 0.1, 0.1 splitting
    # x_train, x_test, y_train, y_test = train_test_split(original_train_dataset.data, original_train_dataset.label,
    #                                                     test_size=0.1, stratify=original_train_dataset.label, random_state=n_random_state)
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
    #                                                       test_size=0.1, stratify=y_train, random_state=n_random_state)
    x_train, y_train = copy.deepcopy(original_train_dataset.data), copy.deepcopy(original_train_dataset.label)


    print('Start Training...')
    for epoch in range(epochs):
        print('-------- Epoch ' + str(epoch + 1) + ' --------')

        sampling_rate = 1
        dataGenerator_train = OxfordDataGenerator(x_train, y_train, original_train_dataset.infos, sampling_rate)
        dataGenerator_train_x, dataGenerator_train_y = dataGenerator_train[0]
        dataGenerator_train = OxfordDataGenerator_train_test(dataGenerator_train_x, dataGenerator_train_y)
        loss_train, f1_train, acc_train = train(model, dataGenerator_train)

        # dataGenerator_val = OxfordDataGenerator_train_test(x_valid, y_valid)
        # loss_val, f1_val, acc_val = val_test(model, dataGenerator_val)

        print('Training Loss: {:.4f}'.format(loss_train))
        # print('Training AUC: {:.4f}'.format(roc_train))
        print('Training f1: {:.4f}'.format(f1_train))
        print('Training acc: {:.4f}'.format(acc_train))
        writer.add_scalar('Train Loss', loss_train, global_step=epoch)
        # print('Val Loss: {:.4f}'.format(loss_val))
        # # print('Val AUC: {:.4f}'.format(roc_val))
        # print('Val f1: {:.4f}'.format(f1_val))
        # print('Val acc: {:.4f}'.format(acc_val))
        # writer.add_scalar('Val Loss', loss_val, global_step=epoch)

        if acc_train > max_acc:
            model_max = copy.deepcopy(model)
            max_acc = acc_train

            # 保存
            model_max.eval()
            torch.save(model_max, './output/models/classify/model' + str(epoch) + '_' + str(max_acc) + '.pth')

    writer.close()

    # dataGenerator_test = OxfordDataGenerator_train_test(x_test, y_test)
    # loss_test, f1_test, acc_test = val_test(model_max, dataGenerator_test)
    # print('test Loss: {:.6f}'.format(loss_test))
    # # print('test AUC: {:.6f}'.format(roc_test))
    # print('test f1: {:.6f}'.format(f1_test))
    # print('test acc: {:.6f}'.format(acc_test))
    print("finished!")

if __name__ == '__main__':
    main()
