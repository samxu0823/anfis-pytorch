#!/usr/bin/env python3
# @ -*- coding: utf-8 -*-
# @Time:   2021/4/22 00:54
# @Author:  Wei XU <samxu0823@gmail.com>

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import BellMembFunc, make_bell_mfs, make_tri_mfs, make_gauss_mfs
from experimental import plot_all_mfs
from train_anfis import train_anfis_cv, plot_results, test_anfis
from ini_my_model import my_model_m1, my_model_k1, my_model_c1, my_model_class1, classifier_rig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_dataset(file_x, file_y, num_input, start, path, batch_size=65536):
    """
    Load the data set generated from MatLab.
    :param file_x: File name of input
    :param file_y: File name of target
    :param num_input: No. of the feature
    :param start: Start point of the feature
    :param path: Path of folder
    :param batch_size: Training batch size
    :return:
    DataLoader: Data set in form of DataLoader
    """
    df_x = pd.read_csv(f"{path}/{file_x}")
    df_y = pd.read_csv(f"{path}/{file_y}")
    # data = np.concatenate((df_x.to_numpy(dtype=float)[:, 1:], df_y.to_numpy(dtype=float)[:, 1:]), axis=1)
    x = df_x.to_numpy(dtype=float)[:, start: start + num_input]
    y = df_y.to_numpy(dtype=float)[:, 1]
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def my_plot(y_pre, y_tar, channel, title):
    """
    Plot prediction - target result.
    :param y_pre: Prediction
    :param y_tar: Target
    :param channel: Output channel
    :param title: Title of the plot
    :return:
    """
    plt.figure()
    plt.scatter(range(len(y_pre)), y_pre, label="prediction", color="b", marker="x")
    plt.scatter(range(len(y_tar)), y_tar, label="target", color="r", marker="o", alpha=0.6)
    plt.title(title)
    plt.ylabel(channel)
    plt.xlabel("index")
    plt.legend()
    plt.show()


def my_qq_plot(y_pre, y_tar, title):
    """
    Plot qq - plot.
    :param y_pre: Prediction
    :param y_tar: Target
    :param title: Title of the plot
    :return:
    """
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


def my_confusion_matrix(y_actual, y_pre, title, normalize="true", mode="sim"):
    """
    Confusions matrix for evaluating the performance of the classifier.
    :param y_actual: Target
    :param y_pre: Prediction
    :param title: Title of the plot
    :param normalize: Normalized by all predictions with respect to one true label (over column)
    :param mode: Output with 8 classes when "sim", Output with 7 classes when "rig"
    :return:
    """
    if mode == "sim":
        label = np.array(["nc", "dc1", "dc2", "dc3", "dc4", "dc5", "dc6", "dc7"])
        n_class = 8
    else:
        label = np.array(["nc", "dc1", "dc2", "dc3", "dc4", "dc5", "dc6"])
        n_class = 7
    confusion = confusion_matrix(y_actual, y_pre, labels=np.arange(0, n_class), normalize=normalize)
    form = '.2g' if normalize else 'd'
    acc = np.sum(np.diag(confusion)) / np.sum(confusion, axis=(0, 1))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=label)
    disp.plot(values_format=form)
    plt.title(title)
    plt.show()
    print(f"{title} total accuracy: {100 * acc:.3f}%")


if __name__ == "__main__":
    # Basic set up
    mode = "c"  # "c" for classifier and "r" for regressor
    load = True  # Load the external trained model (True) or train a model (False)
    save = False  # Save the trained model
    customer = False  # Use customer data
    load_path = "my_model/classification/classifier/Class_final_100"  # External model path: Classifier
    # load_path = "my_model/regression/m1/Reg_m1_final"     # External model path: Regressor
    num_input = 10
    model_type = "class"  # Select the desired model type for training
    save_path = "/classifier/class_final"  # Set save path and name of trained model
    conf_exp = "sim" if model_type == "class" else "rig"    # Different confusion matrix due to generic and rig

    # Load training dataset
    path = "my_data_set/generic/training"   # Path of folder (Training data)
    name1 = "anfis_classifier_100"               # Name of training data set
    file_name_xtrain = f"{name1}_x_train.csv"
    file_name_ytrain = f"{name1}_y_train.csv"
    file_name_xtest = f"{name1}_x_test.csv"
    file_name_ytest = f"{name1}_y_test.csv"
    file_name_xval = f"{name1}_x_val.csv"
    file_name_yval = f"{name1}_y_val.csv"

    # Customer dataset
    customer_path = "my_data_set/generic/robustness_test"       # Path of folder (Customer data)
    name = "customer_label085"                                  # Name of customer data set
    file_name_xcus = f"{name}_x.csv"
    file_name_ycus = f"{name}_y.csv"
    df_x = pd.read_csv(f"{path}/{file_name_xtrain}")
    label = list(df_x.columns)[1:]
    dtype = torch.float64

    # Initialization
    training_data = load_dataset(file_name_xtrain, file_name_ytrain, num_input=num_input, start=1, path=path)
    test_data = load_dataset(file_name_xtest, file_name_ytest, num_input=num_input, start=1, path=path)
    if not customer:  # validation set
        val_data = load_dataset(file_name_xval, file_name_yval, num_input=num_input, start=1, path=path)
    else:  # customer set
        val_data = load_dataset(file_name_xcus, file_name_ycus, num_input=num_input, start=1, path=customer_path)
    x_train, _ = training_data.dataset.tensors
    if not load:
        if model_type == "class":
            my_model = my_model_class1()
        elif model_type == "m1":
            my_model = my_model_m1()
        elif model_type == "k1":
            my_model = my_model_k1()
        elif model_type == "c1":
            my_model = my_model_c1()
        elif model_type == "class_rig":
            my_model = classifier_rig(window="small")
        print("mf before:", my_model.layer.fuzzify)
        with open("my_model/membership/before/premise.txt", "w") as text_file:
            print(f"{my_model.layer.fuzzify}", file=text_file)
        plot_all_mfs(my_model, x_train, save=True, path="before")

        # Training
        epoch = 2000
        metric = "ce" if mode == "c" else "rmse"
        _, y_train_tar, y_train_pre, y_test_tar, y_test_pre = \
            train_anfis_cv(my_model, [training_data, test_data], epoch=epoch, show_plots=True, \
                           metric=metric, mode=mode, save=save, name=save_path, detail=False)
        show_plot = True

    else:
        my_model = torch.load(load_path)
        show_plot = False

    # Visualization of validation set
    print("norm:", my_model.weights[2])
    print("raw:", my_model.raw_weights[2])
    y_val_pre, y_val_tar = test_anfis(my_model, val_data, None, show_plot, mode=mode)
    with open("my_model/membership/after/premise.txt", "w") as text_file:
        print(f"{my_model.layer.fuzzify}", file=text_file)

        # Information flow: Explainability
    # print("Input:", val_data.dataset.tensors[0][2])
    # print("Target:", val_data.dataset.tensors[1][2])
    # c = torch.exp(y_val_pre)
    # d = torch.reshape(torch.sum(torch.exp(y_val_pre), dim=1), (c.shape[0], 1))
    # e = c / d
    # print("Probability:", e[2])
    # print("logSoftmax:", y_val_pre / torch.sum(y_val_pre))

    if mode == "r":  # regression
        plot_results(y_val_tar, y_val_pre)
        my_plot(y_val_pre.detach().numpy(), y_val_tar.detach().numpy(), model_type, title="Validation")
        my_qq_plot(y_val_pre.detach().numpy(), y_val_tar.detach().numpy(), title="QQ plot")
        # plt.scatter(range(y_val_tar.size()[0]), y_val_tar.detach().numpy(), color='r', marker='o', label="target")
        # plt.scatter(range(y_val_pre.size()[0]), y_val_pre.detach().numpy(), color='b', marker='x', label="prediction")
    else:  # classification
        if not load:
            exp = "rig"
            my_confusion_matrix(y_train_tar.numpy(), np.vstack(torch.argmax(y_train_pre, dim=1).numpy()), "Training",
                                None, exp)
            my_confusion_matrix(y_train_tar.numpy(), np.vstack(torch.argmax(y_train_pre, dim=1).numpy()), "Training",
                                "true", exp)
            my_confusion_matrix(y_test_tar.numpy(), np.vstack(torch.argmax(y_test_pre, dim=1).numpy()), "Testing", None,
                                exp)
            my_confusion_matrix(y_test_tar.numpy(), np.vstack(torch.argmax(y_test_pre, dim=1).numpy()), "Testing",
                                "true", exp)
        exp = "rig"
        my_confusion_matrix(y_val_tar.numpy(), np.vstack(torch.argmax(y_val_pre, dim=1).numpy()), "Validation", None,
                            exp)
        my_confusion_matrix(y_val_tar.numpy(), np.vstack(torch.argmax(y_val_pre, dim=1).numpy()), "Validation", "true",
                            exp)
