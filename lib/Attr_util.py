import numpy as np
import random
import copy
from tqdm.auto import tqdm

snp_count = 66
ori_list_one_hot_10 = list(range(10))

def generate_mutant(sequence, snp_pos, n=1):
    one_hot_position = np.where(sequence[snp_pos] == 1)[0].item()
    tmp_list = copy.deepcopy(ori_list_one_hot_10)
    tmp_list.remove(one_hot_position)
    # print(tmp_list)
    random.shuffle(tmp_list)
    mutant_pos_list = tmp_list[:n]

    temp = copy.deepcopy(sequence)
    temp[snp_pos, one_hot_position] = 0
    mutant_seq = np.expand_dims(temp, 0).repeat(n, axis=0)
    for i in range(n):
        mutant_seq[i, snp_pos, mutant_pos_list[i]] = 1

    del tmp_list
    return mutant_seq



def generate_mutants(sequences, n=1):
    sample_count = len(sequences)
    mutant_seqs = []

    print("Start mutant:")
    for i in tqdm(range(sample_count)):
        mutant_seqs.append(sequences[i])
        for j in range(snp_count):
            mutant_seq = generate_mutant(sequences[i], j, n)
            for k in range(n):
                mutant_seqs.append(mutant_seq[k])


    return np.array(mutant_seqs)


def score_to_matrix(scores, sample_count, n=1):
    score_matrix = np.zeros((sample_count, snp_count, snp_count))

    for i in range(sample_count):
        for j in range(snp_count):
            diff_all = np.zeros(snp_count)
            for k in range(n):
                diff = abs(scores[0 + i * (snp_count * n + 1) + j * n + k + 1]-scores[0 + i * (snp_count * n + 1)])
                diff_all = diff_all + diff

            score_matrix[i][j] = diff_all / n

    return score_matrix



if __name__ == '__main__':
    print("Collecting data!")
    X_data, Y_label = np.load('../data/minidata_x_torch.npy'), np.load('../data/minilabel_y_torch.npy')
    print("Data collected!")

    n = 3
    mutant_X_data = generate_mutants(X_data, n)
    test = mutant_X_data[1]

    print()