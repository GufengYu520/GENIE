import random
import numpy as np
from torch.utils import data


class OxfordDataGenerator():
    def __init__(self, data, label, infos, sampling_rate=0.8):
        self.data, self.label, self.infos = data, label, infos
        self.sampling_rate = sampling_rate

        self.case_count = len(self.label[self.label == 1])
        self.sampling_case_count = int(self.case_count * self.sampling_rate)
        self.control_count = len(self.label) - self.case_count
        self.case_index = np.argwhere(self.label == 1).reshape(-1)

        self.sample_dict = self.split_sample()

    def __len__(self):
        return self.control_count // self.sampling_case_count

    def __getitem__(self, idx):
        data, label, index = self.down_sample()
        return data, label

    def split_sample(self):
        sample_dict = {
            "m0":[], # male age < 60
            "f0":[], # female age < 60
            "m1":[], # male 60 <= age < 70
            "f1":[], # female 60 <= age < 70
            "m2":[], # male age >= 70
            "f2":[], # female age >= 70
        }

        for i in range(len(self.label)):
            if self.label[i] == 0:
                if self.infos[4][i] == 1 and self.infos[3][i] < 60:
                    sample_dict['m0'].append(i)
                elif self.infos[4][i] == 0 and self.infos[3][i] < 60:
                    sample_dict['f0'].append(i)
                elif self.infos[4][i] == 1 and 60 <= self.infos[3][i] < 70:
                    sample_dict['m1'].append(i)
                elif self.infos[4][i] == 0 and 60 <= self.infos[3][i] < 70:
                    sample_dict['f1'].append(i)
                elif self.infos[4][i] == 1 and self.infos[3][i] >= 70:
                    sample_dict['m2'].append(i)
                elif self.infos[4][i] == 0 and self.infos[3][i] >= 70:
                    sample_dict['f2'].append(i)

        return sample_dict

    def down_sample(self):
        res_data = []
        res_label = []
        res_index = []

        '''
                case : 2224
                0 <=  age < 50 : 21
                50 <= age < 60 : 202
                60 <= age < 70 : 1943
                70 <= age < 80 : 58
        '''

        sampling_index = random.sample(list(self.case_index), int(self.sampling_case_count))

        for i in sampling_index:
            res_data.append(self.data[i])
            res_index.append(i)
            res_label.append(self.label[i])

            dict_key = None
            if self.infos[4][i] == 1 and self.infos[3][i] < 60:
                dict_key = 'm0'
            elif self.infos[4][i] == 0 and self.infos[3][i] < 60:
                dict_key = 'f0'
            elif self.infos[4][i] == 1 and 60 <= self.infos[3][i] < 70:
                dict_key = 'm1'
            elif self.infos[4][i] == 0 and 60 <= self.infos[3][i] < 70:
                dict_key = 'f1'
            elif self.infos[4][i] == 1 and self.infos[3][i] >= 70:
                dict_key = 'm2'
            elif self.infos[4][i] == 0 and self.infos[3][i] >= 70:
                dict_key = 'f2'

            idx = random.randint(0, len(self.sample_dict[dict_key]) - 1)
            sample_idx = self.sample_dict[dict_key][idx]
            res_data.append(self.data[sample_idx])
            res_index.append(sample_idx)
            res_label.append(self.label[sample_idx])

        return np.array(res_data), np.array(res_label), np.array(res_index)


class OxfordDataGenerator_train_test(data.Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]