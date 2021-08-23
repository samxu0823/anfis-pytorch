#!/usr/bin/env python3
# @ -*- coding: utf-8 -*-
# @Time:   2021/4/21 23:00
# @Author:  Wei XU <samxu0823@gmail.com>    

from matplotlib import pyplot as plt
from experimental import plot_all_mfs
import torch
import time
import os
import torch.nn.functional as F


def train_anfis_cv(model, data, epochs=500, show_plots=False, metric="rmse", mode="r", \
                   save=False, name="my_model", detail=False):
    """
    Train the given model using the given (x,y) data.
    :param model: [AnfisNet], the given model.
    :param data: [DataLoader] or [list], training data or list of training data and test data.
    :param epochs: [int], no. of training epoch.
    :param show_plots: [boolean], show the learning curve.
    :param metric: [str], error used to plot curve.
    :param mode: [str], choose "r" train the anfis regressor, choose "c" train the anfis classifier.
    :param save: [boolean], save the model or not.
    :param name: [str], name of the saving file.
    :param detail: [boolean], save the model with detailed or simplified information.
    :return:
    see return of train_anfis_with_cv.
    """
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss(reduction='sum') if mode == "r" else torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000], gamma=0.6)
    return train_anfis_with_cv(model, data, optimizer, criterion, scheduler, epochs, show_plots, metric, save, name,
                               detail)


def train_anfis_with_cv(model, data, optimizer, criterion, scheduler,
                        epochs=500, show_plots=False, metric="mse", save=False, name="my_model", detail=False):
    """
    Train the given model using training data, meanwhile use test data to test the model in every epoch.
    :return:
    :param model: [AnfisNet], the given model.
    :param data: [DataLoader] or [list], training data or list of training data and test data.
    :param optimizer: [optim], optimizer.
    :param criterion: [Loss], loss function.
    :param scheduler: [lr_scheduler], adaptive learning rate.
    :param epochs: [int], no. of training epoch.
    :param show_plots: [boolean], show the learning curve.
    :param metric: [str],  error used to plot curve.
    :param save: [boolean], save the model or not.
    :param name: [str], name of the saving file.
    :param detail: [boolean], save the model with detailed or simplified information.
    :return:
    x_train: [tensor], x value from data set.
    y_actual: [tensor], Target y value.
    y_pred: [tensor], Prediction by x_train using trained model.
    """
    time_start = time.time()
    test = False  # Using test data during training or not
    classifier = False  # regression or classification

    if isinstance(criterion, torch.nn.MSELoss):  # regression
        errors = {"mse": [], "rmse": [], "PE": []}  # Keep a dict of these for plotting afterwards
    elif isinstance(criterion, torch.nn.CrossEntropyLoss):  # Classification
        classifier = True
        errors = {"ce": [], "acc": []}  # Keep a dict of Cross Entropy for plotting afterwards

    if isinstance(data, list):
        test = True
        train_data, test_data = data
        print('### Training for {} epochs, training size = {} cases, test size = {}'.
              format(epochs, train_data.dataset.tensors[0].shape[0], test_data.dataset.tensors[0].shape[0]))
    else:
        train_data = data
        print('### Training for {} epochs, training size = {} cases'.
              format(epochs, train_data.dataset.tensors[0].shape[0]))

    for t in range(epochs):
        # Process each mini-batch in turn:
        for x_train, y_train_actual in train_data:  # Random x: data loader will shuffle the x, y pairs in every epoch
            y_train_pred = model(x_train)
            # Compute and print loss
            loss = criterion(y_train_pred, y_train_actual) if not classifier else \
                criterion(y_train_pred, y_train_actual.squeeze().long())
            # Zero gradients, perform a backward pass, and update the weights(Premise parameter).
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        # Epoch ending, so now fit the coefficients(Consequent parameter) based on all data:
        x_train, y_train_actual = train_data.dataset.tensors  # Training data in sequence
        with torch.no_grad():
            model.fit_coeff(x_train, y_train_actual)
        # Get the error rate for the whole batch:
        y_train_pred = model(x_train)
        if not classifier:
            mse, rmse, perc_loss, _ = calc_error(y_train_pred, y_train_actual)
        else:
            ce, accuracy = calc_error_class(y_train_pred, y_train_actual)
        if test:
            x_test, y_test_actual = test_data.dataset.tensors  # test data in sequence
            # for x_test, y_test_actual in test_data:   random test data set
            y_test_pred = model(x_test)
            if not classifier:
                test_mse, test_rmse, test_perc_loss, _ = calc_error(y_test_pred, y_test_actual)
                mse = (mse, test_mse)
                rmse = (rmse, test_rmse)
                perc_loss = (perc_loss, test_perc_loss)
                errors["mse"].append(mse)
                errors["rmse"].append(rmse)
                errors["PE"].append(perc_loss)
            else:
                test_ce, test_accuracy = calc_error_class(y_test_pred, y_test_actual)
                ce = (ce, test_ce)
                accuracy = (accuracy, test_accuracy)
                errors["ce"].append(ce)
                errors["acc"].append(accuracy)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            if not test:
                if not classifier:
                    print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f}, PE ={:.2f}%'
                          .format(t, mse, rmse, perc_loss))
                elif classifier:
                    print('epoch {:4d}: CrossEntropy={:.5f}, Accuracy={:.3f}%'
                          .format(t, ce, accuracy))
            else:
                train_error = errors[metric][-1][0]
                test_error = errors[metric][-1][1]
                print(f'epoch {t}: Train {metric}={train_error:.5f}, Test {metric}={test_error:.5f}')
            print("Current LR:", scheduler.get_last_lr()[0])
            print("-----------------")
    test_err = [x for _, x in errors[metric]]
    time_end = time.time()
    print("Training time: %.2fs" % (time_end - time_start))
    if metric == "acc":
        m_test_err = max(test_err)
        m_epoch = test_err.index(max(test_err)) + 1
        print("max. test accuracy:", m_test_err)
        print("epoch:", m_epoch)
    else:
        m_test_err = min(test_err)
        m_epoch = test_err.index(min(test_err)) + 1
        print("min. test error:", m_test_err)
        print("epoch:", m_epoch)
    x_train = train_data.dataset.tensors[0]
    y_train_actual = train_data.dataset.tensors[1]
    x_test = test_data.dataset.tensors[0]
    y_test_actual = test_data.dataset.tensors[1]
    y_train_pred = model(x_train)
    y_test_pred = model(x_test)

    # show result plot and error plot
    if show_plots:
        plot_errors(errors, metric, test)
        if classifier:
            plot_results_class(y_test_actual, torch.argmax(y_test_pred, dim=1))
        else:
            plot_results(y_test_actual, y_test_pred)

    # save optimal model
    if save:
        other_dict = {"epoch": m_epoch, "model_info": model, "optimizer": optimizer.state_dict(), \
                      "best_loss": m_test_err} if detail else {}  # parameter list if detail is needed
        save_model(model=model, classifier=classifier, name=name, detail=detail, **other_dict)
    return x_train, y_train_actual, y_train_pred, y_test_actual, y_test_pred


def save_model(model, classifier, name="my_model", detail=False, **other_dict):
    """
    Model saving module.
    :param model: [AnfisNet], trained Anfis model.
    :param classifier: [boolean], True is classification, False is regression.
    :param name: [str], file name.
    :param detail: [boolean], save the model with detailed or simplified information.
    :param other_dict: [dict], contain model, optimizer, epoch and loss information.
    :return:
    void
    """
    # Defined path and create folder
    model_type = "classification" if classifier else "regression"
    model_path = f"my_model/{model_type}"
    folder = os.path.exists(model_path)
    if not folder:
        os.makedirs(model_path)
    if not detail:
        # default save module
        torch.save(model, f"{model_path}/{name}")
    else:
        # customized save_module
        torch.save(other_dict, f"{model_path}/{name}")


def test_anfis(model, data, train=None, show_plots=False, mode="r"):
    """
    Do a single forward pass with x and compare with y_actual.
    :param model: [AnfisNet], anfis model after training.
    :param data: [DataLoader], test data set.
    :param train: Use training data or test data to plot membership function
    :param show_plots: [boolean], show membership function after training and result.
    :param mode: [str], choose "r" train the anfis regressor, choose "c" train the anfis classifier.
    :return:
    """

    x_test, y_actual = data.dataset.tensors
    if show_plots:
        if train is not None:
            x_train, _ = train.dataset.tensors
            plot_all_mfs(model, x_train, save=True, path="after")
        else:
            plot_all_mfs(model, x_test, save=True, path="after")
    print('### Testing for {} cases'.format(x_test.shape[0]))
    y_pred = model(x_test)
    if mode == "r":
        mse, rmse, perc_loss, r2 = calc_error(y_pred, y_actual)
        print('R2 = {:.4f}, MS error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
              .format(r2, mse, rmse, perc_loss))
    else:
        ce, accuracy = calc_error_class(y_pred, y_actual)
        print('Cross Entropy = {:.4f}, accuracy = {:.4f}'
              .format(ce, accuracy))
    if show_plots:
        if mode == "r":
            plot_results(y_actual, y_pred)
        else:
            plot_results_class(y_actual, torch.argmax(y_pred, dim=1))
    return y_pred, y_actual


def calc_error(y_pred, y_actual):
    """
    Calculate the error of regressor.
    :param y_pred: Prediction
    :param y_actual: Target
    :return:
    tot_loss: Mean square error
    rmse: Root mean square error
    perc_loss: Percentage error
    r2: Determination coefficient
    """
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
                                                / y_actual))
        ss_tot = torch.sum((y_pred - torch.mean(y_pred)).pow(2))
        ss_res = torch.sum((y_pred - y_actual).pow(2))
        r2 = 1 - ss_res / ss_tot
    return tot_loss, rmse, perc_loss, r2


def calc_error_class(y_pred, y_actual):
    """
    Calculate the error of classifier.
    :param y_pred: Prediction
    :param y_actual: Target
    :return:
    current: Current error
    accuracy: Accuracy
    """
    with torch.no_grad():
        error = torch.nn.CrossEntropyLoss()
        current = error(y_pred, y_actual.squeeze().long())
        num_right = torch.sum(torch.argmax(y_pred, dim=1) == y_actual.squeeze().long())
        accuracy = num_right / y_pred.size()[0]
    return current, accuracy


def plot_errors(err, metric, test_data=False):
    """
    Plot the given dict of error rates against no. of epochs.
    Currently support mse, rmse, PE.
    :param err: [dict], (train_err, test_err), contain the different types of error
    :param metric: [str], plot certain error against no. of epochs
    :param test_data: [boolean], if true, plot test error along with training error
    :return:
    void
    """
    if not test_data:
        plt.figure()
        error = err[metric]
        plt.plot(range(len(error)), error, '-ro', label='errors')
    else:
        plt.figure()
        train_error = [x for x, _ in err[metric]]
        test_error = [x for _, x in err[metric]]
        plt.plot(range(len(train_error)), train_error, marker='o', color='r', label='Training')
        plt.plot(range(len(test_error)), test_error, marker='x', color='b', label='Test')
    plt.ylabel(f"{metric}")
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def plot_errors_class(err):
    """
    Plot the error against no. of epoch when train a classifier
    :param err: [dict], cross entropy
    :return:
    """
    plt.figure()
    plt.plot(range(len(err)), err, '-ro')
    plt.ylabel("Cross Entropy")
    plt.xlabel('Epoch')
    plt.show()


def plot_results(y_actual, y_predicted, title="Validation", channel="m1"):
    """
    Plot the target vs. prediction result of regressor.
    :param y_actual: Target
    :param y_predicted: Prediction
    :param title: Title of the plot
    :param channel: Output channel
    :return:
    """
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='Prediction', linewidth=0.8, linestyle='-.')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='Target', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.ylabel(channel)
    plt.xlabel("Index")
    plt.show()


def plot_results_class(y_actual, y_predicted):
    """
    Plot the target vs. prediction result of classifier.
    :param y_actual: Target
    :param y_predicted: Prediction
    :return:
    """
    plt.scatter(range(len(y_predicted)), y_predicted.detach().numpy(), color='r', marker='o', alpha=0.5,
                label='trained')
    plt.scatter(range(len(y_actual)), y_actual.numpy(), color='b', marker='x', label='original')
    plt.legend(loc='upper left')
    plt.show()
