import random

import numpy as np

def cv_split(X, Y, folds = 5):
    np.random.seed(12345)
    random.seed(12345)

    zipped = list(zip(X, Y))
    random.shuffle(zipped)
    X, Y = zip(*zipped)

    ret = [[[], [], [], []] for _ in range(folds)]


    for i in range(len(X)):
        for fold in range(folds):
            if i % folds == fold:
                ret[fold][2].append(X[i])
                ret[fold][3].append(Y[i])
            else:
                ret[fold][0].append(X[i])
                ret[fold][1].append(Y[i])

    ret = [[fold[0], np.array(fold[1]), fold[2], np.array(fold[3])] for fold in ret]

    return ret
