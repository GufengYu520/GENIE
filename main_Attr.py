import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lib import Attr_util as util

device = 'cuda'
params = {'batch_size': 1000,
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

def compute_importance(model, sequences):
    baseline = np.array([1/16, 1/8, 1/8, 1/8, 1/16, 1/8, 1/8, 1/16, 1/8, 1/16])
    baselines = torch.tensor(np.expand_dims(baseline, 0).repeat(66, axis=0)).to(device).float()
    baselines = torch.unsqueeze(baselines, 0)
    model.eval()
    ig = IntegratedGradients(model)

    sequences = torch.tensor(sequences).float()
    data_loader = DataLoader(sequences, **params)
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


def main():
    print("Collecting data!")
    X_data, Y_label = np.load('./data/minidata_x_torch.npy'), np.load('./data/minilabel_y_torch.npy')
    print("Data collected!")
    sample_count = len(X_data)

    n = 9
    mutant_X_data = util.generate_mutants(X_data, n)
    print(mutant_X_data.shape)

    model_path = './output/models/model_tmp.pth'
    model = torch.load(model_path, map_location=device)
    model = torch.nn.DataParallel(model)
    scores = compute_importance(model, mutant_X_data)
    print(scores.shape)

    score_matrix = util.score_to_matrix(scores, sample_count)
    np.save('./data/score_matrix_mini_torch', score_matrix)
    print(score_matrix.shape)

def adj_to_weight():
    score_matrix = np.load('./data/score_matrix_mini_torch.npy')
    sample_count = len(score_matrix)
    snp_count = len(score_matrix[0])
    weights = np.zeros((sample_count, int(snp_count * (snp_count - 1) / 2)))

    for i in range(sample_count):
        weight = []
        for j in range(snp_count):
            for k in range(j + 1, snp_count):
                weight.append(score_matrix[i][j][k] + score_matrix[i][k][j])

        weights[i] = np.array(weight)

    data_df = pd.DataFrame(weights)
    data_df.to_csv('./data/edge_weights_torch_9.csv')
    np.save('data/edge_weights_torch_9', weights)

if __name__ == '__main__':
    main()
    adj_to_weight()