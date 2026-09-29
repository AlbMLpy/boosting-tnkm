import numpy as np


def mse(y, y_pred):
    return np.mean((y.squeeze() - y_pred.squeeze())**2)


def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))
