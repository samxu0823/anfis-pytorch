import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import BellMembFunc, make_bell_mfs, make_tri_mfs, make_gauss_mfs
from experimental import train_anfis, test_anfis, plot_all_mfs, train_anfis_cv


# Load dataset
path = "my_data_set/week16"
file_name_xtrain = "anfis_c_x_train.csv"
file_name_ytrain = "anfis_c_y_train.csv"
file_name_xtest = "anfis_c_x_test.csv"
file_name_ytest = "anfis_c_y_test.csv"
file_name_xval = "anfis_c_x_val.csv"
file_name_yval = "anfis_c_y_val.csv"
df_x = pd.read_csv(f"{path}/{file_name_xtrain}")
label = list(df_x.columns)[1:]
dtype = torch.float64


def load_dataset(file_x, file_y, num_input, start, path, batch_size=1024):
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
        ('x0', make_gauss_mfs(4, np.linspace(min(inp[:, 0]), max(inp[:, 0]), 4))),
        ('x1', make_gauss_mfs(4, np.linspace(min(inp[:, 1]), max(inp[:, 1]), 4))),
        ('x2', make_gauss_mfs(4, np.linspace(min(inp[:, 2]), max(inp[:, 2]), 4))),
        ('x3', make_gauss_mfs(4, np.linspace(min(inp[:, 3]), max(inp[:, 3]), 4))),
        ('x4', make_gauss_mfs(4, np.linspace(min(inp[:, 4]), max(inp[:, 4]), 4))),
        ('x5', make_gauss_mfs(4, np.linspace(min(inp[:, 5]), max(inp[:, 5]), 4))),
        ('x6', make_gauss_mfs(4, np.linspace(min(inp[:, 6]), max(inp[:, 6]), 4))),
        ('x7', make_gauss_mfs(4, np.linspace(min(inp[:, 7]), max(inp[:, 7]), 4))),
        ('x8', make_gauss_mfs(4, np.linspace(min(inp[:, 8]), max(inp[:, 8]), 4))),
        ('x9', make_gauss_mfs(4, np.linspace(min(inp[:, 9]), max(inp[:, 9]), 4))),
    ]
    outvars = ['m1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars)
    rules = [[1, 3, 3, 3, 3, 0, 0, 3, 3, 2], [3, 1, 1, 1, 1, 3, 3, 1, 1, 0],
             [0, 2, 2, 2, 2, 1, 1, 2, 2, 3], [2, 0, 0, 0, 0, 2, 2, 0, 0, 1]]
    model.set_rules(rules)
    return model


def my_model_k(inp):
    invardefs = [
        ('x0', make_gauss_mfs(3, np.linspace(min(inp[:, 0]), max(inp[:, 0]), 3))),
        ('x1', make_gauss_mfs(3, np.linspace(min(inp[:, 1]), max(inp[:, 1]), 3))),
        ('x2', make_gauss_mfs(3, np.linspace(min(inp[:, 2]), max(inp[:, 2]), 4))),
        ('x3', make_gauss_mfs(3, np.linspace(min(inp[:, 3]), max(inp[:, 3]), 4))),
        ('x4', make_gauss_mfs(3, np.linspace(min(inp[:, 4]), max(inp[:, 4]), 3))),
        ('x5', make_gauss_mfs(3, np.linspace(min(inp[:, 5]), max(inp[:, 5]), 3))),
        ('x6', make_gauss_mfs(3, np.linspace(min(inp[:, 6]), max(inp[:, 6]), 2))),
        ('x7', make_gauss_mfs(3, np.linspace(min(inp[:, 7]), max(inp[:, 7]), 4))),
        ('x8', make_gauss_mfs(3, np.linspace(min(inp[:, 8]), max(inp[:, 8]), 3))),
        ('x9', make_gauss_mfs(3, np.linspace(min(inp[:, 9]), max(inp[:, 9]), 3))),
    ]
    outvars = ['m1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars)
    rules = [[2, 1, 1, 2, 1, 2, 1, 2, 1, 1], [2, 2, 2, 3, 2, 1, 1, 3, 2, 0],
             [0, 0, 0, 0, 0, 2, 1, 0, 0, 0], [1, 1, 3, 1, 0, 0, 0, 1, 0, 2]]
    model.set_rules(rules)
    return model


def my_model_c(inp):
    invardefs = [
        ('x0', make_gauss_mfs(4, np.linspace(min(inp[:, 0]), max(inp[:, 0]), 2))),
        ('x1', make_gauss_mfs(4, np.linspace(min(inp[:, 1]), max(inp[:, 1]), 3))),
        ('x2', make_gauss_mfs(4, np.linspace(min(inp[:, 2]), max(inp[:, 2]), 4))),
        ('x3', make_gauss_mfs(4, np.linspace(min(inp[:, 3]), max(inp[:, 3]), 4))),
        ('x4', make_gauss_mfs(4, np.linspace(min(inp[:, 4]), max(inp[:, 4]), 3))),
        ('x5', make_gauss_mfs(4, np.linspace(min(inp[:, 5]), max(inp[:, 5]), 3))),
        ('x6', make_gauss_mfs(4, np.linspace(min(inp[:, 6]), max(inp[:, 6]), 3))),
        ('x7', make_gauss_mfs(4, np.linspace(min(inp[:, 7]), max(inp[:, 7]), 4))),
        ('x8', make_gauss_mfs(4, np.linspace(min(inp[:, 8]), max(inp[:, 8]), 3))),
        ('x9', make_gauss_mfs(4, np.linspace(min(inp[:, 9]), max(inp[:, 9]), 3))),
    ]
    outvars = ['m1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars)
    rules = [[0, 2, 1, 2, 2, 1, 0, 2, 2, 0], [0, 2, 3, 3, 2, 0, 0, 3, 2, 2],
             [1, 0, 0, 0, 0, 2, 2, 0, 0, 1], [1, 1, 3, 1, 1, 1, 1, 1, 1, 2],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    model.set_rules(rules)
    return model


def my_plot(y_pre, y_tar, channel, title):
    plt.figure()
    plt.scatter(range(len(y_pre)), y_pre, label="prediction", color="b", marker="x")
    plt.scatter(range(len(y_tar)), y_tar, label="target", color="r", marker="o", alpha=0.6)
    plt.title = title
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
    plt.title = title
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    training_data = load_dataset(file_name_xtrain, file_name_ytrain, num_input=10, start=1, path=path)
    test_data = load_dataset(file_name_xtest, file_name_ytest, num_input=10, start=1, path=path)
    val_data = load_dataset(file_name_xval, file_name_yval, num_input=10, start=1, path=path)
    x_train, _ = training_data.dataset.tensors
    my_model = my_model_c(x_train)
    print("mf before:", my_model.layer.fuzzify)
    # plot_all_mfs(my_model, x_train)
    train_anfis_cv(my_model, [training_data, test_data], 754, show_plots=True, metric="rmse")
    y_test_pre, y_tar_pre = test_anfis(my_model, val_data, training_data, True)
    print("mf after:", my_model.layer.fuzzify)
    my_plot(y_test_pre.detach().numpy(), y_tar_pre.detach().numpy(), "c1", "prediction")
    my_qq_plot(y_test_pre.detach().numpy(), y_tar_pre.detach().numpy(), "QQ plot")
    # plt.scatter(range(y_tar_pre.size()[0]), y_tar_pre.detach().numpy(), color='r', marker='o', label="target")
    # plt.scatter(range(y_test_pre.size()[0]), y_test_pre.detach().numpy(), color='b', marker='x', label="prediction")
