import torch
import numpy as np

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def getWeight(data):
    n = len(data)
    labels = []
    for i in range(n):
        labels.append(data[i][1].item())
    n1 = sum(np.array(labels))
    n2 = n - n1

    w1 = 1/(n1/n)
    w0 = 1/(n2/n)
    w = [w0, w1]
    weight = [w[int(l)] for l in labels]

    return weight