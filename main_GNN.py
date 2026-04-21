from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import copy
from tqdm.auto import tqdm
import numpy as np
import torch
import dgl
from dgl.dataloading import GraphDataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
import argparse

from lib.models import GATModel, GCN2Model, GCNModel, GraphSAGEModel
from lib import GNN_util as util
from lib.utils import same_seed, getWeight


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--drop_last', type=int, default=0)

    # graph data
    parser.add_argument('--kmer', type=str, default='9mer_mean_768')
    parser.add_argument('--threshold_attr', type=int, default=0.2)
    parser.add_argument('--bidirected', type=int, default=1)
    parser.add_argument('--self_loop', type=int, default=1)

    # model training (special)
    parser.add_argument('--class_weight', type=int, default=2)
    parser.add_argument('--max_acc', type=float, default=0.7)
    parser.add_argument('--model_type', type=str, default='graphsage', choices=['graphsage', 'gat', 'gcn', 'gcn2'])

    parser.add_argument('--kfold', type=int, default=5)



    args = parser.parse_args()
    args.drop_last = True if args.drop_last == 1 else False
    args.bidirected = True if args.drop_last == 1 else False
    args.self_loop = True if args.drop_last == 1 else False

    return args


args = get_args()

same_seed(args.seed)




def train(model, loss_fct, data, sample_weight, optimizer, params):
    sampler = WeightedRandomSampler(sample_weight, len(data))
    train_loader = GraphDataLoader(data, sampler=sampler, **params)

    model.train()

    y_pred = []
    y_label = []
    loss_history = 0
    cnt = 0

    for batched_graph, labels in tqdm(train_loader):
        batched_graph, labels = batched_graph.to(args.device), labels.to(args.device)
        logits = model(batched_graph, batched_graph.ndata['feature'])
        loss = loss_fct(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = y_pred + logits.flatten().tolist()
        y_label = y_label + labels.flatten().tolist()
        pred_int = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0)])
        loss_history = loss_history + loss.item()

        cnt = cnt + 1

    roc_train = roc_auc_score(y_label, y_pred)
    f1_train = f1_score(y_label, pred_int)
    acc_train = accuracy_score(y_label, pred_int)

    return loss_history / cnt, roc_train, f1_train, acc_train


def val_test(model, loss_fct, data,  params):
    val_test_dataloader = GraphDataLoader(data, **params)

    model.eval()

    y_pred = []
    y_label = []
    loss_history = 0
    cnt = 0

    with torch.no_grad():
        for batched_graph, labels in tqdm(val_test_dataloader):
            batched_graph, labels = batched_graph.to(args.device), labels.to(args.device)
            logits = model(batched_graph, batched_graph.ndata['feature'])
            loss = loss_fct(logits, labels)

            y_pred = y_pred + logits.flatten().tolist()
            y_label = y_label + labels.flatten().tolist()
            pred_int = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0)])
            loss_history = loss_history + loss.item()

            cnt = cnt + 1


    roc_val_test = roc_auc_score(y_label, y_pred)
    f1_val_test = f1_score(y_label, pred_int)
    acc_val_test = accuracy_score(y_label, pred_int)

    return loss_history/cnt, roc_val_test, f1_val_test, acc_val_test


def main():
    max_acc = args.max_acc
    # logs
    writer = SummaryWriter('./output/record/logs/GNN')

    # load data
    print("Collecting data!")
    X_data, Y_label = np.load('./data/minidata_x_torch.npy'), \
                      torch.tensor(np.load('./data/minilabel_y_torch.npy'), dtype=torch.float32)
    score_matrix = np.load('./data/score_matrix_mini_torch.npy')
    print("Data collected!")

    kmer = args.kmer
    kmerPath = './data/kmer/SNP_{}.npy'.format(kmer)
    graph_param = {
        "nodeType": kmerPath,
        'threshold_attr': args.threshold_attr,
        'bidirected': args.bidirected,
        'self_loop': args.self_loop,
    }

    raw_dataset = util.SNPDataset(X_data, Y_label, score_matrix, graph_param)
    # training_data = copy.deepcopy(raw_dataset)
    # 0.9, 0.1, 0 splitting
    training_data, validation_data, test_data = dgl.data.utils.split_dataset(raw_dataset, frac_list=[0.9, 0.1, 0],
                                                                             shuffle=True, random_state=args.seed)

    sample_weight = getWeight(training_data)

    # Model and optimizer
    kmer_data = np.load(kmerPath)
    input_dim = len(kmer_data[0][0])
    if args.model_type == 'graphsage':
        model = GraphSAGEModel(dropout=args.dropout, input_dim=input_dim).to(args.device)
    elif args.model_type == 'gat':
        model = GATModel(dropout=args.dropout, input_dim=input_dim).to(args.device)
    elif args.model_type == 'gcn':
        model = GCNModel(dropout=args.dropout, input_dim=input_dim).to(args.device)
    elif args.model_type == 'gcn2':
        model = GCN2Model(dropout=args.dropout, input_dim=input_dim).to(args.device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    loss_fct = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([args.class_weight])).to(args.device)
    params = {'batch_size': args.batch_size,
              # 'shuffle': True,
              'num_workers': args.num_workers,
              'drop_last': args.drop_last}

    print('Start Training...')
    for epoch in range(args.epochs):
        print('-------- Epoch ' + str(epoch + 1) + ' --------')

        loss_train, roc_train, f1_train, acc_train = train(model, loss_fct, training_data, sample_weight, optimizer, params)

        loss_val, roc_val, f1_val, acc_val = val_test(model, loss_fct, validation_data, params)

        print('Training Loss: {:.4f}'.format(loss_train))
        print('Training AUC: {:.4f}'.format(roc_train))
        print('Training f1: {:.4f}'.format(f1_train))
        print('Training acc: {:.4f}'.format(acc_train))
        writer.add_scalar('Train Loss', loss_train, global_step=epoch)
        writer.add_scalar('Train Acc', acc_train, global_step=epoch)
        print('Val Loss: {:.4f}'.format(loss_val))
        print('Val AUC: {:.4f}'.format(roc_val))
        print('Val f1: {:.4f}'.format(f1_val))
        print('Val acc: {:.4f}'.format(acc_val))
        writer.add_scalar('Val Loss', loss_val, global_step=epoch)
        writer.add_scalar('Val Acc', acc_val, global_step=epoch)

        if acc_val > max_acc:
            model_max = copy.deepcopy(model)
            max_acc = acc_val
            model_max.eval()
            torch.save(model_max, './output/models/GNN/{}/{}_epoch{}_{:.4f}.pth'.format(kmer, args.model_type, epoch, max_acc))

    writer.close()


    # loss_test, roc_test, f1_test, acc_test = val_test(model_max, test_data)
    # print('test Loss: {:.6f}'.format(loss_test))
    # print('test AUC: {:.6f}'.format(roc_test))
    # print('test f1: {:.6f}'.format(f1_test))
    # print('test acc: {:.6f}'.format(acc_test))
    print("finished!")


def main_kfold(k=5):
    max_acc = args.max_acc
    tmp = 0
    max_acc_Kfold = []
    max_f1_Kfold = []
    max_AUC_Kfold = []
    max_epoch_Kfold = []

    # logs
    writer = SummaryWriter('./output/record/logs/GNN_KFold')

    # load data
    print("Collecting data!")
    X_data, Y_label = np.load('./data/minidata_x_torch.npy'), \
        torch.tensor(np.load('./data/minilabel_y_torch.npy'), dtype=torch.float32)
    score_matrix = np.load('./data/score_matrix_mini_torch.npy')
    print("Data collected!")

    kmer = args.kmer
    kmerPath = './data/kmer/SNP_{}.npy'.format(kmer)
    graph_param = {
        "nodeType": kmerPath,
        'threshold_attr': args.threshold_attr,
        'bidirected': args.bidirected,
        'self_loop': args.self_loop,
    }
    Kfold_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)

    raw_dataset = util.SNPDataset(X_data, Y_label, score_matrix, graph_param)

    loss_fct = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([args.class_weight])).to(args.device)
    params = {'batch_size': args.batch_size,
              # 'shuffle': True,
              'num_workers': args.num_workers,
              'drop_last': args.drop_last}


    print('Start Training...')
    for i, (train_index, val_index) in enumerate(Kfold_split.split(X_data, Y_label)):
        print("Fold {}:".format(i+1))
        # training_data = util.SNPDataset(X_data[train_index], Y_label[train_index],
        #                                 score_matrix[train_index], kmerPath)
        # validation_data = util.SNPDataset(X_data[val_index], Y_label[val_index],
        #                                   score_matrix[val_index], kmerPath)

        training_data = dgl.data.utils.Subset(raw_dataset, train_index)
        training_data, _, _ = dgl.data.utils.split_dataset(training_data, frac_list=[1, 0, 0])
        validation_data = dgl.data.utils.Subset(raw_dataset, val_index)

        sample_weight = getWeight(training_data)
        # Model and optimizer
        kmer_data = np.load(kmerPath)
        input_dim = len(kmer_data[0][0])
        if args.model_type == 'graphsage':
            model = GraphSAGEModel(dropout=args.dropout, input_dim=input_dim).to(args.device)
        elif args.model_type == 'gat':
            model = GATModel(dropout=args.dropout, input_dim=input_dim).to(args.device)
        elif args.model_type == 'gcn':
            model = GCNModel(dropout=args.dropout, input_dim=input_dim).to(args.device)
        elif args.model_type == 'gcn2':
            model = GCN2Model(dropout=args.dropout, input_dim=input_dim).to(args.device)
        model_f = copy.deepcopy(model)

        optimizer_f = optim.AdamW(model_f.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            print('-------- Epoch ' + str(epoch + 1) + ' --------')

            loss_train, roc_train, f1_train, acc_train = train(model_f, loss_fct, training_data, sample_weight, optimizer_f, params)

            loss_val, roc_val, f1_val, acc_val = val_test(model_f, loss_fct, validation_data, params)

            print('Training Loss: {:.4f}'.format(loss_train))
            print('Training AUC: {:.4f}'.format(roc_train))
            print('Training f1: {:.4f}'.format(f1_train))
            print('Training acc: {:.4f}'.format(acc_train))
            writer.add_scalar('Train Loss (Fold {})'.format(i+1), loss_train, global_step=epoch)
            writer.add_scalar('Train Acc (Fold {})'.format(i+1), acc_train, global_step=epoch)
            print('Val Loss: {:.4f}'.format(loss_val))
            print('Val AUC: {:.4f}'.format(roc_val))
            print('Val f1: {:.4f}'.format(f1_val))
            print('Val acc: {:.4f}'.format(acc_val))
            writer.add_scalar('Val Loss (Fold {})'.format(i+1), loss_val, global_step=epoch)
            writer.add_scalar('Val Acc (Fold {})'.format(i+1), acc_val, global_step=epoch)

            if acc_val > tmp:
                tmp = acc_val
                tmp_f1 = f1_val
                tmp_AUC = roc_val
                tmp_epoch = epoch

            if acc_val > max_acc:
                model_max = copy.deepcopy(model_f)
                max_acc = acc_val
                model_max.eval()
                torch.save(model_max, './output/models/GNN_KFold/{}/{}_{}_fold{}_epoch{}_{:.4f}.pth'.format(kmer, args.threshold_attr, args.model_type, i+1, epoch, max_acc))

        max_acc_Kfold.append(tmp)
        max_f1_Kfold.append(tmp_f1)
        max_AUC_Kfold.append(tmp_AUC)
        max_epoch_Kfold.append(tmp_epoch)
        tmp = 0
        max_acc = args.max_acc

    writer.close()

    print('-------- Result: --------')
    for i in range(k):
        print("Fold {}, epoch{} : max acc: {:.4f}, max_f1: {:.4f}, max AUC: {:.4f}".format(i+1, max_epoch_Kfold[i],
                                                                                           max_acc_Kfold[i],
                                                                                           max_f1_Kfold[i],
                                                                                           max_AUC_Kfold[i]))
    avg_acc = sum(max_acc_Kfold)/k
    avg_f1 = sum(max_f1_Kfold) / k
    avg_AUC = sum(max_AUC_Kfold) / k

    print("AVG acc: {:.4f}, AVG f1: {:.4f}, AVG AUC: {:.4f}".format(avg_acc, avg_f1, avg_AUC))

    print("finished!")






if __name__ == '__main__':
    main_kfold()
