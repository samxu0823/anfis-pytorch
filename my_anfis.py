import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import BellMembFunc, make_bell_mfs
from experimental import train_anfis, test_anfis, plot_all_mfs, train_anfis_cv


# Load dataset
path = "my_data_set/week15"
file_name_xtrain = "m1_x_train.csv"
file_name_ytrain = "m1_y_train.csv"
file_name_xtest = "m1_x_test.csv"
file_name_ytest = "m1_y_test.csv"
file_name_xval = "m1_x_val.csv"
file_name_yval = "m1_y_val.csv"
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
        ('x0', make_bell_mfs(2, 2, np.linspace(min(inp[:, 0]), max(inp[:, 0]), 3))),
        ('x1', make_bell_mfs(2, 2, np.linspace(min(inp[:, 1]), max(inp[:, 1]), 3))),
        ('x2', make_bell_mfs(2, 2, np.linspace(min(inp[:, 2]), max(inp[:, 2]), 3)))]
    outvars = ['m1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars)
    rules = [[0, 0, 0], [1, 0, 2], [2, 1, 0]]
    model.set_rules(rules)
    return model


def my_plot(y_pre, y_tar, title):
    plt.figure()
    plt.scatter(range(len(y_pre)), y_pre, label="prediction", color="b", marker="x")
    plt.scatter(range(len(y_tar)), y_tar, label="target", color="r", marker="o", alpha=0.6)
    plt.title = title
    plt.legend()
    plt.show()


if __name__ == "__main__":
    training_data = load_dataset(file_name_xtrain, file_name_ytrain, num_input=3, start=1, path=path)
    test_data = load_dataset(file_name_xtest, file_name_ytest, num_input=3, start=1, path=path)
    val_data = load_dataset(file_name_xval, file_name_yval, num_input=3, start=1, path=path)
    x_train, _ = training_data.dataset.tensors
    my_model = my_model(x_train)
    # plot_all_mfs(my_model, x_train)
    train_anfis_cv(my_model, [training_data, test_data], 300, show_plots=True, metric="rmse")
    y_test_pre, y_tar_pre = test_anfis(my_model, val_data, True)
    plt.figure()
    plt.scatter(range(y_tar_pre.size()[0]), y_tar_pre.detach().numpy(), color='r', marker='o', label="target")
    plt.scatter(range(y_test_pre.size()[0]), y_test_pre.detach().numpy(), color='b', marker='x', label="prediction")
    plt.legend()
    plt.show()