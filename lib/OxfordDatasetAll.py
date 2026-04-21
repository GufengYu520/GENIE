import torch.utils.data as data
import numpy as np
from lib import deeplift_util as util
from lib import classifier_util as util2


class OxfordDatasetAll(data.Dataset):
    def __init__(self, path, outstanding_path=""):
        file = np.load(path, allow_pickle=True)

        data = file['data']
        self.infos = file['infos']
        label = file['label']

        if outstanding_path != "":
            outstanding_snp_names = open(outstanding_path).read().strip().split('\n')
            idx = []
            for i in range(len(self.infos[0])):
                if self.infos[0][i] in outstanding_snp_names:
                    idx.append(i)

            infos_snp = []
            for i in range(len(self.infos[0])):
                if i in idx:
                    infos_snp.append(self.infos[0][i])

            self.infos[0] = infos_snp
            data = data[:, idx]

        self.data = []
        for d in data:
            d = np.array(d, dtype=str)
            sequence = ''.join(d)
            one_hot = util.one_hot_encode(sequence)
            self.data.append(np.array(one_hot))
        self.data = np.array(self.data)

        possible_mutants_dict = util.possible_mutants(self.data)
        self.possible_mutants_dict = possible_mutants_dict

        self.label = np.array(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __setitem__(self, key, value):
        self.label[key] = value

    def __len__(self):
        return len(self.label)



class OxfordDatasetAll_v2(data.Dataset):
    def __init__(self, path, outstanding_path=""):
        file = np.load(path, allow_pickle=True)

        data = file['data']
        self.infos = file['infos']
        label = file['label']

        if outstanding_path != "":
            outstanding_snp_names = open(outstanding_path).read().strip().split('\n')
            idx = []
            for i in range(len(self.infos[0])):
                if self.infos[0][i] in outstanding_snp_names:
                    idx.append(i)

            infos_snp = []
            for i in range(len(self.infos[0])):
                if i in idx:
                    infos_snp.append(self.infos[0][i])

            self.infos[0] = infos_snp
            data = data[:, idx]

        self.data = []
        for d in data:
            d = np.array(d, dtype=str)
            sequence = ''.join(d)
            one_hot = util2.one_hot_encode_10(sequence)
            self.data.append(np.array(one_hot))
        self.data = np.array(self.data)
        self.label = np.array(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __setitem__(self, key, value):
        self.label[key] = value

    def __len__(self):
        return len(self.label)

