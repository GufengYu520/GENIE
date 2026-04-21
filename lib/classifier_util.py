import numpy as np
import random
import copy

snp_count = 66
ori_list_one_hot = list(range(16))


def generate_mutant(sequence, snp_pos, n=1):
    """对一条序列进行突变，产生n条突变序列，n最多为15"""
    one_hot_position = np.where(sequence[snp_pos] == 1)[0].item()
    tmp_list = copy.deepcopy(ori_list_one_hot)
    tmp_list.remove(one_hot_position)
    # print(tmp_list)
    random.shuffle(tmp_list)
    mutant_pos_list = tmp_list[:n]

    mutant_seq = np.zeros((n, 66, 16))
    for i in range(n):
        mutant_seq[i, snp_pos, mutant_pos_list[i]] = 1

    del tmp_list
    return mutant_seq



def generate_mutants(sequences, n=1):
    """改变SNP序列中的位点状态，16种情况中随机选择n种"""
    sample_count = len(sequences) # 总样本数/人数
    mutant_seqs = []

    for i in range(sample_count):
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

# ['AA','AC','AG','AT','CA','CC','CG','CT','GA','GC','GG','GT','TA','TC','TG','TT']
def get_orig_letter(one_hot_vec):
    try:
        assert(len(np.where(one_hot_vec!=0)[0]) == 1)
    except:
        return 'N'
    if one_hot_vec[0] != 0:
        return 'AA'
    elif one_hot_vec[1] != 0:
        return 'AC'
    elif one_hot_vec[2] != 0:
        return 'AG'
    elif one_hot_vec[3] != 0:
        return 'AT'
    elif one_hot_vec[4] != 0:
        return 'CA'
    elif one_hot_vec[5] != 0:
        return 'CC'
    elif one_hot_vec[6] != 0:
        return 'CG'
    elif one_hot_vec[7] != 0:
        return 'CT'
    elif one_hot_vec[8] != 0:
        return 'GA'
    elif one_hot_vec[9] != 0:
        return 'GC'
    elif one_hot_vec[10] != 0:
        return 'GG'
    elif one_hot_vec[11] != 0:
        return 'GT'
    elif one_hot_vec[12] != 0:
        return 'TA'
    elif one_hot_vec[13] != 0:
        return 'TC'
    elif one_hot_vec[14] != 0:
        return 'TG'
    elif one_hot_vec[15] != 0:
        return 'TT'


def possible_mutants(data) :
    possible_mutants_dict = {}
    for i in range(len(data)):
        for j in range(len(data[0])):
            cur_one_hot = data[i][j]
            cur_base = get_orig_letter(cur_one_hot)

            if (j not in possible_mutants_dict.keys()) or (cur_base not in possible_mutants_dict[j]):
                if j not in possible_mutants_dict.keys():
                    possible_mutants_dict[j] = [cur_base]
                else:
                    possible_mutants_dict[j].extend([cur_base])

    return possible_mutants_dict


def one_hot_encode_10(sequence):

    one_hot_array = np.zeros((int(len(sequence) / 2), 10))

    for i in range(0, len(sequence), 2):
        if sequence[i] == "A" and sequence[i + 1] == 'A':
            char_pos = 0
        elif (sequence[i] == "A" and sequence[i + 1] == 'C') or (sequence[i] == "C" and sequence[i + 1] == 'A'):
            char_pos = 1
        elif (sequence[i] == "A" and sequence[i + 1] == 'G') or (sequence[i] == "G" and sequence[i + 1] == 'A'):
            char_pos = 2
        elif (sequence[i] == "A" and sequence[i + 1] == 'T') or (sequence[i] == "T" and sequence[i + 1] == 'A'):
            char_pos = 3
        elif sequence[i] == "C" and sequence[i + 1] == 'C':
            char_pos = 4
        elif (sequence[i] == "C" and sequence[i + 1] == 'G') or (sequence[i] == "G" and sequence[i + 1] == 'C'):
            char_pos = 5
        elif (sequence[i] == "C" and sequence[i + 1] == 'T') or (sequence[i] == "T" and sequence[i + 1] == 'C'):
            char_pos = 6
        elif sequence[i] == "G" and sequence[i + 1] == 'G':
            char_pos = 7
        elif (sequence[i] == "G" and sequence[i + 1] == 'T') or (sequence[i] == "T" and sequence[i + 1] == 'G'):
            char_pos = 8
        elif sequence[i] == "T" and sequence[i + 1] == 'T':
            char_pos = 9
        else:
            raise RuntimeError("Unsupported character")

        one_hot_array[int(i/2), char_pos] = 1

    return one_hot_array