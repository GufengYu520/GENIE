import numpy as np
from random import sample
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lib.OxfordDatasetAll import OxfordDatasetAll_v2

data_path = './data/oxford_imputed_ad.npz'
outstanding_path = './data/snps_oxford'

device = 'cuda'
params = {'batch_size': 256,
          'num_workers': 6,
          'drop_last': False}

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
seed = 88
same_seed(seed)

def model_predict(model, data):
    data_loader = DataLoader(data, **params)
    model.eval()

    y_pred = []
    with torch.no_grad():
        for batched_data in tqdm(data_loader):
            batched_data = batched_data.to(device).float()
            logits = model(batched_data)
            y_pred = y_pred + logits.flatten().tolist()

    return np.array(y_pred)

def main():
    # load data
    print("Collecting data!")
    original_train_dataset = OxfordDatasetAll_v2(data_path, outstanding_path)
    print("Data collected!")

    X_data, Y_label = original_train_dataset.data, original_train_dataset.label

    # model_path = './checkpoints/model' + str(i) + '.h5'
    model_path = './output/models/model_test.pth'
    model = torch.load(model_path, map_location=device)
    pred = model_predict(model, X_data)
    # pred = model(X_data)

    pred_label = np.asarray([1 if i else 0 for i in (np.asarray(pred) >= 0.5)])

    # high score
    list0 = []
    list1 = []

    for i in range(len(Y_label)):
        if pred_label[i] == Y_label[i]:
            if pred_label[i] == 0 and pred[i] <= 1e-7:
                list0.append(i)
            elif pred_label[i] == 1:
                # print(pred[i])
                list1.append(i)

    print(len(list0))
    print(len(list1))
    listall = sample(list0, 5000) + list1
    print(len(listall))
    # minidata0, label0 = X_data[list0, :, :], Y_label[list0]
    # minidata1, label1 = X_data[list1, :, :], Y_label[list1]
    minidata, minilabel = X_data[listall, :, :], Y_label[listall]
    np.save('./data/minidata_x_torch', minidata)
    np.save('./data/minilabel_y_torch', minilabel)


if __name__ == '__main__':
    main()