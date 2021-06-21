import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import BellMembFunc, make_bell_mfs, make_tri_mfs, make_gauss_mfs
from experimental import train_anfis, test_anfis, plot_all_mfs, train_anfis_cv

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
path = "my_data_set/week24"
file_name_xtrain = "anfis_c_x_train.csv"
file_name_ytrain = "anfis_c_y_train.csv"
file_name_xtest = "anfis_c_x_test.csv"
file_name_ytest = "anfis_c_y_test.csv"
file_name_xval = "anfis_c_x_val.csv"
file_name_yval = "anfis_c_y_val.csv"
file_name_xcus = "customer_label10_x.csv"
file_name_ycus = "customer_label10_y.csv"
df_x = pd.read_csv(f"{path}/{file_name_xtrain}")
label = list(df_x.columns)[1:]
dtype = torch.float64


def load_dataset(file_x, file_y, num_input, start, path, batch_size=65536):
    df_x = pd.read_csv(f"{path}/{file_x}")
    df_y = pd.read_csv(f"{path}/{file_y}")
    # data = np.concatenate((df_x.to_numpy(dtype=float)[:, 1:], df_y.to_numpy(dtype=float)[:, 1:]), axis=1)
    x = df_x.to_numpy(dtype=float)[:, start: start + num_input]
    y = df_y.to_numpy(dtype=float)[:, 1]
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def my_model(inp):
    """
    my model
    :param inp: [tensor], input data set.
    :return:
    """
    invardefs = [
        ('x0', make_bell_mfs(2, 2, np.linspace(min(inp[:, 0]), max(inp[:, 0]), 2))),
        ('x1', make_bell_mfs(2, 2, np.linspace(min(inp[:, 1]), max(inp[:, 1]), 2))),
        ('x2', make_bell_mfs(2, 2, np.linspace(min(inp[:, 2]), max(inp[:, 2]), 2)))]
    outvars = ['m1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars)
    # rules = [[0, 0, 0], [1, 0, 2], [2, 1, 0]]
    # model.set_rules(rules)
    return model


def my_model_m(inp):
    """
    my model
    :param inp: [tensor], input data set.
    :return:
    """
    invardefs = [
        ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
        ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
        ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
        ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
        ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
        ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
    ]
    outvars = ['m1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)
    # rules = [[1, 3, 3, 3, 3, 0, 0, 3, 3, 2], [3, 1, 1, 1, 1, 3, 3, 1, 1, 0],
    #          [0, 2, 2, 2, 2, 1, 1, 2, 2, 3], [2, 0, 0, 0, 0, 2, 2, 0, 0, 1]]
    # rules = [[0, 3, 9, 0, 8, 0, 9, 5, 9, 4], [8, 8, 0, 9, 0, 9, 4, 8, 4, 7],
    #          [4, 3, 4, 3, 5, 4, 2, 9, 2, 3], [2, 9, 4, 1, 8, 2, 8, 5, 8, 4],
    #          [0, 5, 9, 0, 9, 0, 7, 9, 7, 0], [9, 0, 0, 9, 2, 9, 0, 9, 0, 3],
    #          [1, 2, 4, 1, 9, 1, 4, 0, 4, 3], [4, 2, 4, 3, 3, 4, 6, 9, 6, 9]]
    # add v feature
    # rules = [[0, 3, 9, 0, 8, 0, 9, 0, 5, 0], [8, 8, 0, 9, 0, 9, 4, 9, 8, 9],
    #          [4, 3, 4, 3, 5, 4, 2, 3, 9, 3], [2, 9, 4, 1, 8, 2, 8, 1, 5, 1],
    #          [0, 5, 9, 0, 9, 0, 7, 0, 9, 0], [9, 0, 0, 9, 2, 9, 0, 9, 9, 9],
    #          [1, 2, 4, 1, 9, 1, 4, 1, 0, 1], [4, 2, 4, 3, 3, 4, 6, 3, 9, 3]]
    # re scaling
    # rules = [[0,	0,	2,	0,	1,	0,	3,	0,	0,	0], [0,	0,	2,	0,	1,	0,	3,	0,	0,	0],
    #          [1,	0,	1,	1,	1,	1,	2,	1,	1,	1], [2,	1,	0,	2,	0,	2,	1,	2,	0,	2],
    #          [3,	1,	0,	3,	0,	3,	1,	2,	0,	2], [1,	0,	1,	1,	1,	1,	0,	0,	0,	0],
    #          [1,	0,	1,	1,	1,	1,	0,	1,	0,	0], [0,	1,	1,	0,	1,	0,	1,	0,	1,	0]]
    # reduce MF
    rules = [[0, 2, 2, 0, 4, 0, 5, 0, 1, 0], [3, 4, 0, 3, 0, 3, 2, 3, 2, 3],
             [2, 2, 1, 2, 3, 2, 1, 2, 3, 2], [1, 5, 1, 1, 4, 1, 4, 1, 1, 1],
             [0, 3, 2, 0, 5, 0, 4, 0, 3, 0], [4, 0, 0, 4, 1, 4, 0, 3, 3, 3],
             [1, 1, 1, 1, 4, 1, 2, 1, 0, 1], [2, 1, 1, 2, 2, 2, 3, 2, 3, 2]]
    model.set_rules(rules)
    return model


def my_model_m1(inp, rules='less'):
    """
    my model
    :param inp: [tensor], input data set.
    :return:
    """
    if rules == 'less':
        invardefs = [
            ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
        ]
        # 17/06
        # rules = [[0,2,2,6,0,4,0,6,0,0], [3,3,0,2,3,0,3,2,3,3],
        #          [2,4,1,1,2,2,2,1,2,2], [1,0,1,5,1,3,1,5,1,1],
        #          [0,3,2,4,0,4,0,4,0,0], [4,2,0,0,4,1,4,0,4,4],
        #          [1,0,1,2,1,4,1,2,1,1], [2,1,1,3,2,2,2,3,2,2]]
        rules =  [[0,0,5,0,0,0,0,1,1,3], [2,3,3,6,1,2,3,6,3,1],
                 [3,2,0,3,2,3,2,4,6,0], [0,1,6,2,0,0,1,3,0,3],
                 [1,0,4,0,0,1,0,0,2,3], [3,4,1,5,2,3,4,6,5,0],
                 [1,1,4,1,0,1,1,2,2,2], [3,2,2,4,1,3,2,5,4,1]]
    elif rules == 'more':
        invardefs = [
            ('x0', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x1', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x2', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x3', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x4', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x5', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x6', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x7', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x8', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),
            ('x9', make_gauss_mfs(0.4, np.linspace(0, 1, 10))),]

        rules = [[1,8,5,2,7,4,6,2,7,7], [0,6,4,1,6,4,8,1,6,6],
                 [1,6,4,2,5,4,8,1,5,5], [1,7,5,2,6,4,7,1,6,6],
                 [1,7,5,2,6,4,7,2,6,6], [0,6,4,1,5,4,8,1,5,5],
                 [1,6,4,2,6,4,7,1,6,6], [1,7,4,2,6,4,7,1,6,6],
                 [0,5,1,1,4,1,0,1,4,4], [0,5,0,0,4,0,5,0,4,4],
                 [1,3,0,1,3,0,5,1,3,3], [0,3,1,0,2,1,1,0,2,2],
                 [0,2,1,1,2,0,3,1,2,2], [0,0,0,0,0,0,5,0,0,0],
                 [0,1,0,0,1,0,0,0,1,1], [1,6,0,1,5,0,5,1,5,5],
                 [6,9,9,7,9,9,9,7,9,9], [1,8,9,3,7,7,9,2,7,7],
                 [1,8,9,2,7,8,9,2,7,7], [4,9,9,6,9,8,9,5,9,9],
                 [4,9,9,5,9,7,9,5,9,9], [1,8,9,3,8,8,9,2,8,8],
                 [9,9,9,9,9,8,9,9,9,9], [1,7,8,2,7,8,9,2,7,7]]
    outvars = ['m1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)

    model.set_rules(rules)
    return model



def my_model_k(inp):
    invardefs = [
        ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
        ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
        ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
        ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
    ]
    outvars = ['k1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)
    # rules = [[8, 3, 8, 9, 4, 5, 0, 6, 5, 6], [0, 0, 0, 0, 0, 0, 9, 0, 9, 9],
    #          [8, 9, 8, 3, 9, 9, 2, 9, 2, 0], [0, 0, 0, 4, 0, 0, 7, 0, 9, 9],
    #          [9, 9, 9, 9, 9, 9, 0, 9, 0, 0], [0, 5, 0, 0, 5, 2, 6, 5, 8, 5],
    #          [0, 4, 0, 4, 5, 2, 3, 5, 7, 5], [8, 3, 8, 3, 4, 6, 2, 6, 3, 6]]
    # add v feature
    # rules = [[8, 3, 8, 9, 4, 5, 5, 0, 6, 9], [0, 0, 0, 0, 0, 0, 0, 9, 0, 1],
    #          [8, 9, 8, 3, 9, 9, 8, 2, 9, 4], [0, 0, 0, 4, 0, 0, 0, 7, 0, 4],
    #          [9, 9, 9, 9, 9, 9, 9, 0, 9, 9], [0, 5, 0, 0, 5, 2, 2, 6, 5, 0],
    #          [0, 4, 0, 4, 5, 2, 3, 3, 5, 3], [8, 3, 8, 3, 4, 6, 4, 2, 6, 5]]
    # reduce MF
    rules = [[1, 1, 1, 2, 1, 2, 4, 1, 2, 3], [0, 0, 0, 0, 0, 0, 0, 5, 1, 1],
             [1, 3, 1, 1, 2, 4, 5, 2, 3, 2], [0, 0, 0, 1, 0, 0, 0, 4, 0, 2],
             [2, 3, 2, 2, 2, 5, 6, 0, 3, 3], [0, 2, 0, 0, 1, 1, 1, 4, 2, 0],
             [0, 2, 0, 1, 1, 1, 2, 3, 2, 2], [1, 1, 1, 1, 1, 3, 3, 2, 2, 2]]
    model.set_rules(rules)
    return model


def my_model_k1():
    invardefs = [
        ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 2))),
        ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
        ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
    ]
    outvars = ['k1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)
    # rules = [[0,0,0,0,0,0,0,0,0,0], [2,2,3,3,3,3,3,3,3,3],
    #          [5,5,5,5,5,5,5,5,5,5], [1,1,2,2,2,2,2,2,2,2],
    #          [4,4,4,4,4,4,4,4,4,4], [3,3,6,6,4,4,7,7,6,6],
    #          [3,3,5,5,4,4,6,6,6,6], [1,1,1,1,1,1,1,1,1,1]]
    rules = [[1,1,2,2,4,1,4,4,1,5], [0,0,0,0,0,0,0,0,2,1],
             [1,3,2,1,5,2,4,5,0,2], [0,0,1,1,1,0,1,1,2,3],
             [1,3,2,2,5,2,5,5,0,4], [0,2,0,0,2,1,1,2,1,0],
             [0,2,1,1,3,1,2,3,1,2], [1,1,2,1,4,1,3,4,1,3]]
    model.set_rules(rules)
    return model


def my_model_c(inp):
    # invardefs = [
    #     ('x0', make_gauss_mfs(0.15, np.linspace(min(inp[:, 0]), max(inp[:, 0]), 10))),
    #     ('x1', make_gauss_mfs(0.15, np.linspace(min(inp[:, 1]), max(inp[:, 1]), 10))),
    #     ('x2', make_gauss_mfs(0.15, np.linspace(min(inp[:, 2]), max(inp[:, 2]), 10))),
    #     ('x3', make_gauss_mfs(0.15, np.linspace(min(inp[:, 3]), max(inp[:, 3]), 10))),
    #     ('x4', make_gauss_mfs(0.15, np.linspace(min(inp[:, 4]), max(inp[:, 4]), 10))),
    #     ('x5', make_gauss_mfs(0.15, np.linspace(min(inp[:, 5]), max(inp[:, 5]), 10))),
    #     ('x6', make_gauss_mfs(0.15, np.linspace(min(inp[:, 6]), max(inp[:, 6]), 10))),
    #     ('x7', make_gauss_mfs(0.15, np.linspace(min(inp[:, 7]), max(inp[:, 7]), 10))),
    #     ('x8', make_gauss_mfs(0.15, np.linspace(min(inp[:, 8]), max(inp[:, 8]), 10))),
    #     ('x9', make_gauss_mfs(0.15, np.linspace(min(inp[:, 9]), max(inp[:, 9]), 10))),
    # ]
    invardefs = [
        ('x0', make_gauss_mfs(0.15, np.linspace(0, 1, 3))),
        ('x1', make_gauss_mfs(0.15, np.linspace(0, 1, 3))),
        ('x2', make_gauss_mfs(0.15, np.linspace(0, 1, 7))),
        ('x3', make_gauss_mfs(0.15, np.linspace(0, 1, 7))),
        ('x4', make_gauss_mfs(0.15, np.linspace(0, 1, 7))),
        ('x5', make_gauss_mfs(0.15, np.linspace(0, 1, 7))),
        ('x6', make_gauss_mfs(0.15, np.linspace(0, 1, 7))),
        ('x7', make_gauss_mfs(0.15, np.linspace(0, 1, 7))),
        ('x8', make_gauss_mfs(0.15, np.linspace(0, 1, 6))),
        ('x9', make_gauss_mfs(0.15, np.linspace(0, 1, 6))),
    ]
    outvars = ['c1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)
    # rules = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #          [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    #          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    #          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    #          [8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]
    # rules = [[0, 2, 1, 2, 2, 1, 0, 2, 2, 0], [0, 2, 3, 3, 2, 0, 0, 3, 2, 2],
    #          [1, 0, 0, 0, 0, 2, 2, 0, 0, 1], [1, 1, 3, 1, 1, 1, 1, 1, 1, 2],
    #          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # rules = [[5, 0, 5, 9, 7, 9, 6, 4, 9, 7], [9, 6, 0, 4, 1, 4, 9, 7, 3, 1],
    #          [0, 7, 9, 1, 9, 1, 0, 4, 1, 9], [9, 2, 0, 7, 0, 7, 9, 3, 7, 0],
    #          [0, 2, 9, 7, 9, 7, 0, 0, 7, 9], [5, 9, 5, 0, 6, 0, 4, 4, 0, 6],
    #          [5, 5, 5, 3, 6, 3, 6, 4, 3, 6], [5, 4, 5, 5, 7, 5, 4, 9, 5, 6]]
    # best rule, input anfis_c1 with 10 MFs in range(0, 1), sigma = 0.5
    # rules = [[5, 5, 9, 9, 9, 9, 9, 9, 0, 0], [9, 0, 4, 4, 3, 3, 3, 3, 6, 6],
    #          [0, 9, 1, 1, 1, 1, 1, 1, 7, 7], [9, 0, 7, 7, 7, 7, 7, 7, 2, 2],
    #          [0, 9, 7, 7, 7, 7, 7, 7, 2, 2], [5, 5, 0, 0, 0, 0, 0, 0, 9, 9],
    #          [5, 5, 3, 3, 3, 3, 3, 3, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 4, 4]]
    # reduce mf
    rules = [[1, 1, 6, 6, 6, 6, 6, 6, 0, 0], [2, 0, 2, 2, 2, 2, 2, 2, 3, 3],
             [0, 2, 1, 1, 1, 1, 1, 1, 4, 4], [2, 0, 5, 5, 5, 5, 5, 5, 1, 1],
             [0, 2, 4, 4, 4, 4, 4, 4, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 5, 5],
             [1, 1, 2, 2, 2, 2, 2, 2, 3, 3], [1, 1, 3, 3, 3, 3, 3, 3, 2, 2]]
    model.set_rules(rules)
    return model


def my_model_c1(inp, rules='less'):
    if rules == 'less':
        invardefs = [
            ('x0', make_gauss_mfs(0.6, np.linspace(0, 1, 6))),
            ('x1', make_gauss_mfs(0.6, np.linspace(0, 1, 6))),
            ('x2', make_gauss_mfs(0.6, np.linspace(0, 1, 8))),
            ('x3', make_gauss_mfs(0.6, np.linspace(0, 1, 8))),
            ('x4', make_gauss_mfs(0.6, np.linspace(0, 1, 8))),
            ('x5', make_gauss_mfs(0.6, np.linspace(0, 1, 8))),
            ('x6', make_gauss_mfs(0.6, np.linspace(0, 1, 7))),
            ('x7', make_gauss_mfs(0.6, np.linspace(0, 1, 7))),
            ('x8', make_gauss_mfs(0.6, np.linspace(0, 1, 8))),
            ('x9', make_gauss_mfs(0.6, np.linspace(0, 1, 7))),
        ]
        # 09.06
        # rules = [[1,0,2,6,1,5,6,5,1,6], [2,4,0,2,0,2,2,2,0,2],
        #          [0,5,3,1,2,1,1,1,5,1], [2,1,1,5,0,5,5,4,0,5],
        #          [0,2,4,4,2,3,4,0,4,4], [1,6,0,0,1,0,0,2,3,0],
        #          [1,4,2,2,1,2,2,3,3,2], [1,3,1,3,1,4,3,4,2,3]]
        # rules = [[1,1,0,2,0,1,0,0,1,0], [2,3,3,0,3,3,3,3,3,4],
        #          [0,4,2,3,2,3,2,2,4,3], [2,3,1,1,1,4,1,1,3,1],
        #          [0,2,0,4,0,2,0,0,2,1], [1,5,4,0,4,5,4,4,5,5],
        #          [1,5,1,2,1,5,1,1,5,4], [1,0,2,1,2,0,2,2,0,2]]
        # rules = [[1,1,1,6,6,6,6,6,6,0], [2,0,0,2,2,2,2,2,2,4],
        #          [0,2,2,1,1,1,1,1,1,5], [2,0,0,5,5,5,5,5,5,1],
        #          [0,2,2,4,4,4,4,4,4,2], [1,1,1,0,0,0,0,0,0,6],
        #          [1,1,1,2,2,2,2,2,2,4], [1,1,1,3,3,3,3,3,3,3]]
        rules = [[1,1,0,0,0,0,0,0,0,0], [0,0,3,3,3,3,3,3,3,3],
                 [5,5,5,5,5,5,4,4,5,4], [0,0,2,2,2,2,2,2,2,2],
                 [4,4,4,4,4,4,3,3,4,3], [3,3,7,7,7,7,6,6,7,6],
                 [3,3,6,6,6,6,5,5,6,5], [2,2,1,1,1,1,1,1,1,1]]
    elif rules == 'more':
        invardefs = [
            ('x0', make_gauss_mfs(0.4, np.linspace(0, 1, 7))),
            ('x1', make_gauss_mfs(0.4, np.linspace(0, 1, 11))),
            ('x2', make_gauss_mfs(0.4, np.linspace(0, 1, 6))),
            ('x3', make_gauss_mfs(0.4, np.linspace(0, 1, 12))),
            ('x4', make_gauss_mfs(0.4, np.linspace(0, 1, 8))),
            ('x5', make_gauss_mfs(0.4, np.linspace(0, 1, 7))),
            ('x6', make_gauss_mfs(0.4, np.linspace(0, 1, 12))),
            ('x7', make_gauss_mfs(0.4, np.linspace(0, 1, 4))),
            ('x8', make_gauss_mfs(0.4, np.linspace(0, 1, 12))),
            ('x9', make_gauss_mfs(0.4, np.linspace(0, 1, 12))),]

        rules = [[3,2,0,9,4,5,9,2,6,9], [4,4,0,5,2,5,5,2,5,6],
                 [2,5,1,4,5,5,4,2,7,5], [4,2,0,8,2,5,8,2,5,8],
                 [2,2,1,7,5,5,7,2,7,8], [3,6,0,3,4,4,3,2,6,4],
                 [3,4,0,5,4,5,5,2,6,6], [3,3,0,6,4,5,6,2,6,7],
                 [3,0,0,5,3,2,5,1,1,6], [3,2,0,2,0,2,2,1,0,3],
                 [0,2,0,1,4,0,1,0,4,1], [3,0,0,4,0,3,4,1,1,5],
                 [0,1,0,2,4,2,2,1,3,3], [1,2,0,0,1,1,0,1,1,0],
                 [1,1,0,1,1,2,1,1,2,2], [3,2,0,4,3,2,4,1,1,5],
                 [3,4,0,11,4,6,11,3,9,11], [6,7,0,9,4,6,9,3,8,8],
                 [3,9,5,8,7,6,8,3,11,8], [6,5,0,11,4,6,11,3,8,11],
                 [3,7,4,10,7,6,10,3,11,10], [5,10,2,9,6,6,9,3,10,8],
                 [5,8,3,10,6,6,10,3,9,10], [3,5,0,9,4,6,9,3,9,8]]
    outvars = ['c1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)
    # rules = [[0,0,0,0,0,0,0,0,0,0], [2,2,3,3,3,3,3,3,3,3],
    #          [5,5,5,5,5,5,5,5,5,5], [1,1,2,2,2,2,2,2,2,2],
    #          [4,4,4,4,4,4,4,4,4,4], [3,3,6,6,4,4,7,7,6,6],
    #          [3,3,5,5,4,4,6,6,6,6], [1,1,1,1,1,1,1,1,1,1]]
    # rules = [[0,4,0,3,0,1,0,1,0,1], [1,4,1,0,1,0,2,2,2,0],
    #          [3,0,1,4,2,2,3,5,3,2], [1,5,1,0,1,0,2,0,2,0],
    #          [2,1,1,5,2,2,3,3,3,2], [2,1,2,1,3,1,3,4,3,1],
    #          [2,3,2,2,3,1,3,2,3,1], [1,2,0,2,0,1,1,3,1,1]]

    model.set_rules(rules)
    return model


def my_model_class(inp):
    """
    my model
    :param inp: [tensor], input data set.
    :return:
    """
    invardefs = [
        # ('x0', make_gauss_mfs(0.1, np.linspace(min(inp[:, 0]), max(inp[:, 0]), 10))),
        # ('x1', make_gauss_mfs(0.1, np.linspace(min(inp[:, 1]), max(inp[:, 1]), 10))),
        # ('x2', make_gauss_mfs(0.1, np.linspace(min(inp[:, 2]), max(inp[:, 2]), 10))),
        # ('x3', make_gauss_mfs(0.1, np.linspace(min(inp[:, 3]), max(inp[:, 3]), 10))),
        # ('x4', make_gauss_mfs(0.1, np.linspace(min(inp[:, 4]), max(inp[:, 4]), 10))),
        # ('x5', make_gauss_mfs(0.1, np.linspace(min(inp[:, 5]), max(inp[:, 5]), 10))),
        # ('x6', make_gauss_mfs(0.1, np.linspace(min(inp[:, 6]), max(inp[:, 6]), 10))),
        # ('x7', make_gauss_mfs(0.1, np.linspace(min(inp[:, 7]), max(inp[:, 7]), 10))),
        # ('x8', make_gauss_mfs(0.1, np.linspace(min(inp[:, 8]), max(inp[:, 8]), 10))),
        # ('x9', make_gauss_mfs(0.1, np.linspace(min(inp[:, 9]), max(inp[:, 9]), 10))),
        ('x0', make_gauss_mfs(0.2, np.linspace(0, 1, 5))),
        ('x1', make_gauss_mfs(0.2, np.linspace(0, 1, 4))),
        ('x2', make_gauss_mfs(0.2, np.linspace(0, 1, 5))),
        ('x3', make_gauss_mfs(0.2, np.linspace(0, 1, 5))),
        ('x4', make_gauss_mfs(0.2, np.linspace(0, 1, 4))),
        ('x5', make_gauss_mfs(0.2, np.linspace(0, 1, 4))),
        ('x6', make_gauss_mfs(0.2, np.linspace(0, 1, 5))),
        ('x7', make_gauss_mfs(0.2, np.linspace(0, 1, 5))),
        ('x8', make_gauss_mfs(0.2, np.linspace(0, 1, 3))),
        ('x9', make_gauss_mfs(0.2, np.linspace(0, 1, 5))),
    ]
    outvars = ['dc1', 'dc2', 'dc3', 'dc4', 'dc5', 'dc6', 'dc7', 'dc8']
    # outvars = ['dc1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, hybrid=False, grid=False)
    # rules = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #          [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    #          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    #          [6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    #          [8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]
    # rules = [[5, 0, 3, 0, 3, 0, 6, 5, 3, 8], [0, 9, 3, 9, 2, 9, 9, 5, 2, 0],
    #     #          [9, 4, 0, 4, 0, 4, 0, 1, 0, 8], [0, 1, 7, 2, 6, 2, 9, 9, 6, 0],
    #     #          [9, 0, 2, 0, 1, 0, 0, 0, 1, 9], [5, 9, 2, 9, 2, 9, 5, 1, 2, 0],
    #     #          [5, 1, 9, 1, 9, 1, 5, 7, 9, 0], [5, 3, 1, 4, 1, 4, 5, 1, 1, 8]]
    # rules = [[9, 0, 0, 3, 0, 0, 3, 0, 6, 0], [4, 9, 9, 3, 9, 9, 2, 9, 9, 9],
    #          [0, 3, 4, 0, 3, 3, 0, 4, 0, 4], [5, 1, 1, 7, 1, 2, 6, 2, 9, 2],
    #          [0, 0, 0, 2, 0, 0, 1, 0, 0, 0], [0, 9, 9, 2, 9, 9, 2, 9, 5, 9],
    #          [9, 1, 1, 9, 1, 1, 9, 1, 5, 1], [6, 3, 3, 1, 3, 3, 1, 4, 5, 4]]
    # reduce mf
    rules = [[4, 0, 0, 2, 0, 0, 2, 0, 1, 0], [1, 3, 3, 2, 3, 3, 1, 3, 2, 3],
             [0, 2, 2, 0, 2, 2, 0, 2, 0, 2], [1, 1, 1, 3, 1, 1, 3, 1, 2, 1],
             [0, 0, 0, 1, 0, 0, 1, 0, 0, 0], [0, 3, 4, 1, 3, 3, 1, 4, 1, 4],
             [3, 1, 1, 4, 1, 1, 4, 1, 1, 1], [2, 2, 2, 1, 2, 2, 1, 2, 1, 2]]

    model.set_rules(rules, hybrid=False)
    return model


def my_model_class1():
    """
    my model
    :param inp: [tensor], input data set.
    :return:
    """
    invardefs = [
        ('x0', make_gauss_mfs(1, np.linspace(0, 1, 3))),
        ('x1', make_gauss_mfs(1, np.linspace(0, 1, 4))),
        ('x2', make_gauss_mfs(1, np.linspace(0, 1, 5))),
        ('x3', make_gauss_mfs(1, np.linspace(0, 1, 5))),
        ('x4', make_gauss_mfs(1, np.linspace(0, 1, 5))),
        ('x5', make_gauss_mfs(1, np.linspace(0, 1, 5))),
        ('x6', make_gauss_mfs(1, np.linspace(0, 1, 5))),
        ('x7', make_gauss_mfs(1, np.linspace(0, 1, 5))),
        ('x8', make_gauss_mfs(1, np.linspace(0, 1, 5))),
        ('x9', make_gauss_mfs(1, np.linspace(0, 1, 5))),
    ]
    outvars = ['dc1', 'dc2', 'dc3', 'dc4', 'dc5', 'dc6', 'dc7', 'dc8']
    # outvars = ['dc1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, hybrid=False, grid=False)
    # 09.06
    # rules = [[1,1,0,0,0,1,0,0,1,0], [2,2,3,0,3,3,3,3,2,3],
    #          [0,3,2,1,2,3,2,2,3,3], [2,2,1,0,1,3,1,1,2,1],
    #          [0,2,0,2,0,2,0,0,2,1], [1,4,4,0,4,4,4,4,4,4],
    #          [1,4,1,0,1,4,1,1,4,3], [1,0,2,0,2,0,2,2,0,2]]
    # rules = [[1,1,1,0,0,0,1,0,6,0], [2,0,2,3,0,3,3,3,2,3],
    #          [0,2,3,2,1,2,3,2,0,2], [2,0,2,1,0,1,3,1,3,1],
    #          [0,2,2,0,2,0,2,0,1,0], [1,1,4,4,0,4,4,4,4,4],
    #          [1,1,4,1,0,1,4,1,5,1], [1,1,0,2,0,2,0,2,6,2]]
    rules = [[2,0,2,0,1,0,1,0,1,0], [0,0,1,3,1,3,2,3,3,3],
             [1,2,3,2,2,2,3,2,3,2], [1,1,3,1,3,1,2,1,3,1],
             [2,0,0,0,0,0,2,0,2,0], [0,1,2,4,2,4,4,4,4,4],
             [1,3,4,1,4,1,4,1,4,1], [1,2,3,2,2,2,0,2,0,2]]
    model.set_rules(rules, hybrid=False)
    return model


def my_plot(y_pre, y_tar, channel, title):
    plt.figure()
    plt.scatter(range(len(y_pre)), y_pre, label="prediction", color="b", marker="x")
    plt.scatter(range(len(y_tar)), y_tar, label="target", color="r", marker="o", alpha=0.6)
    plt.title(title)
    plt.ylabel(channel)
    plt.xlabel("index")
    plt.legend()
    plt.show()


def my_qq_plot(y_pre, y_tar, title):
    plt.figure()
    plt.scatter(y_tar, y_pre, label="prediction", color="b", marker="x")
    plt.plot(range(3), range(3), label="target", color="r", alpha=0.6)
    plt.xlim([np.min(np.vstack((y_pre, y_tar))) - 0.05, np.max(np.vstack((y_pre, y_tar))) + 0.05])
    plt.ylim([np.min(np.vstack((y_pre, y_tar))) - 0.05, np.max(np.vstack((y_pre, y_tar))) + 0.05])
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.legend()
    plt.show()


def my_confusion_matrix(y_actual, y_pre, title, normalize="true"):
    confusion = confusion_matrix(y_actual, y_pre, labels=np.arange(0, 8), normalize=normalize)
    acc = np.sum(np.diag(confusion)) / np.sum(confusion, axis=(0, 1))
    label = np.array(["nc", "dc1", "dc2", "dc3", "dc4", "dc5", "dc6", "dc7"])
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=label)
    disp.plot()
    plt.title(title)
    plt.show()
    print(f"{title} total accuracy: {100 * acc:.3f}%")


if __name__ == "__main__":
    mode = "r"
    save = True
    load = False
    customer = False
    load_path = "my_model/regression/Class_17_06"

    # Initialization
    training_data = load_dataset(file_name_xtrain, file_name_ytrain, num_input=10, start=1, path=path)
    test_data = load_dataset(file_name_xtest, file_name_ytest, num_input=10, start=1, path=path)
    if not customer:  # validation set
        val_data = load_dataset(file_name_xval, file_name_yval, num_input=10, start=1, path=path)
    else:  # customer set
        val_data = load_dataset(file_name_xcus, file_name_ycus, num_input=10, start=1, path=path)
    x_train, _ = training_data.dataset.tensors
    if not load:
        # my_model = my_model_class1()
        my_model = my_model_c1(x_train, rules='less')
        # my_model = my_model_k1()
        print("mf before:", my_model.layer.fuzzify)
        # plot_all_mfs(my_model, x_train)

        # Training
        _, y_train_tar, y_train_pre, y_test_tar, y_test_pre = \
            train_anfis_cv(my_model, [training_data, test_data], 3089, show_plots=True, \
                           metric="rmse", mode=mode, save=save, name="Reg_c_21_06", detail=False)
        show_plot = True

        # Test result
        # my_plot(y_test_pre.detach().numpy(), y_test_tar.detach().numpy(), "k1", title="Test")
        # print("mf after:", my_model.layer.fuzzify)juc

    else:
        my_model = torch.load(load_path)

        # my_model = torch.load(load_path)["model_info"]
        show_plot = False

    # Visualization of validation set
    y_val_pre, y_val_tar = test_anfis(my_model, val_data, None, show_plot, mode=mode)
    a = torch.sum(my_model.raw_weights[0])
    b = my_model.raw_weights[0] / a
    print("a:", a)
    print("b:", b)
    print("norm:", my_model.weights[0])
    print("raw:", my_model.raw_weights[0])
    if mode == "r":  # regression
        my_plot(y_val_pre.detach().numpy(), y_val_tar.detach().numpy(), "m1", title="Validation")
        my_qq_plot(y_val_pre.detach().numpy(), y_val_tar.detach().numpy(), title="QQ plot")
        # plt.scatter(range(y_val_tar.size()[0]), y_val_tar.detach().numpy(), color='r', marker='o', label="target")
        # plt.scatter(range(y_val_pre.size()[0]), y_val_pre.detach().numpy(), color='b', marker='x', label="prediction")
    else:  # classification
        if not load:
            my_confusion_matrix(y_train_tar.numpy(), np.vstack(torch.argmax(y_train_pre, dim=1).numpy()), "Training",
                                None)
            my_confusion_matrix(y_train_tar.numpy(), np.vstack(torch.argmax(y_train_pre, dim=1).numpy()), "Training",
                                "true")
            my_confusion_matrix(y_test_tar.numpy(), np.vstack(torch.argmax(y_test_pre, dim=1).numpy()), "Testing", None)
            my_confusion_matrix(y_test_tar.numpy(), np.vstack(torch.argmax(y_test_pre, dim=1).numpy()), "Testing",
                                "true")
        my_confusion_matrix(y_val_tar.numpy(), np.vstack(torch.argmax(y_val_pre, dim=1).numpy()), "Validation", None)
        my_confusion_matrix(y_val_tar.numpy(), np.vstack(torch.argmax(y_val_pre, dim=1).numpy()), "Validation", "true")
