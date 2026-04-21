import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
import random

n_random_state = 88
n_components = 10
path = './data/oxford_snp_two_dim_all.npz'


def get_data(data_path):
    file = np.load(data_path, allow_pickle=True)
    data = file['data']
    label = file['label']
    infos = file['infos']

    return np.array(data), np.array(label), np.array(infos)


def get_outstanding_index(infos):
    x_index = []
    lista = list(range(0, len(infos[0])))
    x_index.extend(lista)

    for p in range(len(infos[0]) - 1):
        for q in range(p + 1, len(infos[0])):
            x_index.append(str(p) + '*' + str(q))

    x_index = np.array(x_index)

    outstanding_index = []

    outstanding = np.load('data/outstanding_oxford_sampling_ad_sampling_30times_10_1.npy')
    for i in range(len(outstanding)):
        snp1 = outstanding[i][0]
        snp2 = outstanding[i][1]

        k = np.argwhere(x_index == str(min(snp1, snp2)) + '*' + str(max(snp1, snp2)))[0][0]
        outstanding_index.append(k)

    print(outstanding_index)

    return outstanding_index


def get_outstanding_id():
    outstanding = np.load('data/outstanding_oxford_sampling_ad_sampling_30times_10_1.npy')

    outstanding_id = []

    for i in range(len(outstanding)):
        snp1 = outstanding[i][0]
        snp2 = outstanding[i][1]
        outstanding_id.append(int(snp1))
        outstanding_id.append(int(snp2))

    outstanding_id = list(set(outstanding_id))

    print(outstanding_id)

    return outstanding_id


def get_sample_weight(y_train):
    sum = np.mean(y_train)
    weight_0 = sum
    weight_1 = 1 - sum
    weights_train = np.where(y_train == 0, weight_0, weight_1)
    return weights_train


def main():
    data, label, infos = get_data(path)
    outstanding_index = get_outstanding_index(infos)
    outstanding_id = get_outstanding_id()
    print("Data collecting!")
    # exit()

    # x_train = data[:, list(range(0, len(infos[0]))) + outstanding_index] # 10+66 错误
    x_train = data[:, outstanding_id + outstanding_index] # 10+x
    x_train = data # 66+...=2271
    y_train = label

    print("All data:{}".format(np.shape(x_train)))

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                        test_size=0.2, stratify=y_train, random_state=n_random_state)

    print("x_train shape:{}".format(np.shape(x_train)))
    print("y_train shape:{}".format(np.shape(y_train)))

    # model
    clf = Pipeline([("scaler", StandardScaler()),
                    ("logist", LogisticRegression(class_weight='balanced',
                                                  solver='saga',
                                                  max_iter=10000,
                                                  n_jobs=64,
                                                  random_state=n_random_state))])

    print("Start training:")
    sample_weights = get_sample_weight(y_train)
    # clf.fit(x_train, y_train, logist__sample_weight=sample_weights)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)

    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Score:{}".format(score))
    print("f1_score:{}".format(f1))
    print(report)

    print("params:")
    # print(clf.get_params())
    print(clf['logist'].densify())
    print(clf['logist'].coef_)
    params = clf['logist'].coef_
    params = np.array(params)
    np.save("./data/params_lr.npy", params)



if __name__ == '__main__':
    main()