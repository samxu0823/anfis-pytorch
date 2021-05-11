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
path = "my_data_set/week18"
file_name_xtrain = "anfis_class_x_train.csv"
file_name_ytrain = "anfis_class_y_train.csv"
file_name_xtest = "anfis_class_x_test.csv"
file_name_ytest = "anfis_class_y_test.csv"
file_name_xval = "anfis_class_x_val.csv"
file_name_yval = "anfis_class_y_val.csv"
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


def my_model_class(inp):
    """
    my model
    :param inp: [tensor], input data set.
    :return:
    """
    invardefs = [
        ('x0', make_gauss_mfs(0.1, np.linspace(min(inp[:, 0]), max(inp[:, 0]), 10))),
        ('x1', make_gauss_mfs(0.1, np.linspace(min(inp[:, 1]), max(inp[:, 1]), 10))),
        ('x2', make_gauss_mfs(0.1, np.linspace(min(inp[:, 2]), max(inp[:, 2]), 10))),
        ('x3', make_gauss_mfs(0.1, np.linspace(min(inp[:, 3]), max(inp[:, 3]), 10))),
        ('x4', make_gauss_mfs(0.1, np.linspace(min(inp[:, 4]), max(inp[:, 4]), 10))),
        ('x5', make_gauss_mfs(0.1, np.linspace(min(inp[:, 5]), max(inp[:, 5]), 10))),
        ('x6', make_gauss_mfs(0.1, np.linspace(min(inp[:, 6]), max(inp[:, 6]), 10))),
        ('x7', make_gauss_mfs(0.1, np.linspace(min(inp[:, 7]), max(inp[:, 7]), 10))),
        ('x8', make_gauss_mfs(0.1, np.linspace(min(inp[:, 8]), max(inp[:, 8]), 10))),
        ('x9', make_gauss_mfs(0.1, np.linspace(min(inp[:, 9]), max(inp[:, 9]), 10))),
    ]
    outvars = ['dc1', 'dc2', 'dc3', 'dc4', 'dc5', 'dc6', 'dc7', 'dc8']
    # outvars = ['dc1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, hybrid=False, grid=False)
    rules = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
             [6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
             [8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]
    # rules = [[5, 0, 3, 0, 3, 0, 6, 5, 3, 8], [0, 9, 3, 9, 2, 9, 9, 5, 2, 0],
    #          [9, 4, 0, 4, 0, 4, 0, 1, 0, 8], [0, 1, 7, 2, 6, 2, 9, 9, 6, 0],
    #          [9, 0, 2, 0, 1, 0, 0, 0, 1, 9], [5, 9, 2, 9, 2, 9, 5, 1, 2, 0],
    #          [5, 1, 9, 1, 9, 1, 5, 7, 9, 0], [5, 3, 1, 4, 1, 4, 5, 1, 1, 8]]
    model.set_rules(rules, hybrid=False)
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
    mode = "c"
    training_data = load_dataset(file_name_xtrain, file_name_ytrain, num_input=10, start=1, path=path)
    test_data = load_dataset(file_name_xtest, file_name_ytest, num_input=10, start=1, path=path)
    val_data = load_dataset(file_name_xval, file_name_yval, num_input=10, start=1, path=path)
    x_train, _ = training_data.dataset.tensors
    my_model = my_model_class(x_train)
    print("mf before:", my_model.layer.fuzzify)
    plot_all_mfs(my_model, x_train)
    _, y_train_tar, y_train_pre, y_test_tar, y_test_pre = \
        train_anfis_cv(my_model, [training_data, test_data], 5000, show_plots=True, metric="ce", mode="c")
    y_val_pre, y_val_tar = test_anfis(my_model, val_data, training_data, True, mode="c")
    print("mf after:", my_model.layer.fuzzify)
    if mode == "r":
        my_plot(y_val_pre.detach().numpy(), y_val_tar.detach().numpy(), "c1", "prediction")
        my_qq_plot(y_val_pre.detach().numpy(), y_val_tar.detach().numpy(), "QQ plot")
        # plt.scatter(range(y_val_tar.size()[0]), y_val_tar.detach().numpy(), color='r', marker='o', label="target")
        # plt.scatter(range(y_val_pre.size()[0]), y_val_pre.detach().numpy(), color='b', marker='x', label="prediction")
    else:
        my_confusion_matrix(y_train_tar.numpy(), np.vstack(torch.argmax(y_train_pre, dim=1).numpy()), "Training", None)
        my_confusion_matrix(y_train_tar.numpy(), np.vstack(torch.argmax(y_train_pre, dim=1).numpy()), "Training", "true")
        my_confusion_matrix(y_test_tar.numpy(), np.vstack(torch.argmax(y_test_pre, dim=1).numpy()), "Testing", None)
        my_confusion_matrix(y_test_tar.numpy(), np.vstack(torch.argmax(y_test_pre, dim=1).numpy()), "Testing", "true")
        my_confusion_matrix(y_val_tar.numpy(), np.vstack(torch.argmax(y_val_pre, dim=1).numpy()), "Validation", None)
        my_confusion_matrix(y_val_tar.numpy(), np.vstack(torch.argmax(y_val_pre, dim=1).numpy()), "Validation", "true")

