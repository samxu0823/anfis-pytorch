#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple functions to supply data and plot results.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

dtype = torch.float


class TwoLayerNet(torch.nn.Module):
    '''
        From the pytorch examples, a simjple 2-layer neural net.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    def __init__(self, d_in, hidden_size, d_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, d_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def linear_model(x, y, epochs=200, hidden_size=10):
    '''
        Predict y from x using a simple linear model with one hidden layer.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    assert x.shape[0] == y.shape[0], 'x and y have different batch sizes'
    d_in = x.shape[1]
    d_out = y.shape[1]
    model = TwoLayerNet(d_in, hidden_size, d_out)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    errors = []
    for t in range(epochs):
        y_pred = model(x)
        tot_loss = criterion(y_pred, y)
        perc_loss = 100. * torch.sqrt(tot_loss).item() / y.sum()
        errors.append(perc_loss)
        if t % 10 == 0 or epochs < 20:
            print('epoch {:4d}: {:.5f} {:.2f}%'.format(t, tot_loss, perc_loss))
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    return model, errors


def plot_errors(err, metric, test_data=False):
    """
    Plot the given dict of error rates against no. of epochs.
    Currently support mse, rmse, PE.
    :param err: [dict], (train_err, test_err), contain the different types of error.
    :param metric: [str], plot certain error against no. of epochs
    :return:
    void
    """
    if not test_data:
        plt.figure()
        error = err[metric]
        plt.plot(range(len(error)), error, '-ro', label='errors')
        plt.ylabel(f"{metric}")
        plt.xlabel('Epoch')
    else:
        plt.figure()
        train_error = [x for x, _ in err[metric]]
        test_error = [x for _, x in err[metric]]
        plt.plot(range(len(train_error)), train_error, marker='o', color='r', label='Training')
        plt.plot(range(len(test_error)), test_error, marker='x', color='b', label='Test')
    plt.legend()
    plt.show()


def plot_results(y_actual, y_predicted):
    '''
        Plot the actual and predicted y values (in different colours).
    '''
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()


def _plot_mfs(var_name, fv, x):
    """
    A simple utility function to plot the MFs for each feature in interval [x_min, x_max].
    Supply the variable name, MFs and a set of x values to plot.
    :param var_name: [Str], name of the input feature.
    :param fv: [FuzzyfiVariable], Membership function.
    :param x: [tensor], m * 1 tensor, single input feature as x.
    :return:
    """
    # Sort x so we only plot each x-value once:
    min = torch.min(x)
    max = torch.max(x)
    num = 100
    x_in = torch.arange(min, max, (max - min) / num)
    for mfname, yvals in fv.fuzzify(x_in):
        plt.plot(x_in.tolist(), yvals.tolist(), label=mfname)
    plt.xlabel('Values for variable {} ({} MFs)'.format(var_name, fv.num_mfs))
    plt.ylabel('Membership')
    # plt.legend(bbox_to_anchor=(1., 0.95))
    plt.legend()
    plt.show()


def plot_all_mfs(model, x):
    """
    Plot membership function. Detail see _plot_mfs.
    Customize the x interval: tensor([[min, min], [max, max]]) for 2 input features.
    :param model: [AnfisNet], anfis model.
    :param x: [tensor], m * n tensor, a set of input. n is no. of input feature.
    :return:
    """
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        _plot_mfs(var_name, fv, x[:, i])


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
                               / y_actual))
    return(tot_loss, rmse, perc_loss)


def test_anfis(model, data, show_plots=False):
    """
    Do a single forward pass with x and compare with y_actual.
    :param model: [AnfisNet], anfis model after training.
    :param data: [DataLoader], test data set.
    :param show_plots: [boolean], show membership function after training and result.
    :return:
    """

    x_test, y_actual = data.dataset.tensors
    if show_plots:
        plot_all_mfs(model, x_test)
    print('### Testing for {} cases'.format(x_test.shape[0]))
    y_pred = model(x_test)
    mse, rmse, perc_loss = calc_error(y_pred, y_actual)
    print('MS error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
          .format(mse, rmse, perc_loss))
    if show_plots:
        plot_results(y_actual, y_pred)


def train_anfis_with(model, data, optimizer, criterion,
                     epochs=500, show_plots=False, metric="rmse"):
    '''
        Train the given model using the given (x,y) data.
    '''
    errors = {"mse": [], "rmse": [], "PE": []}  # Keep a dict of these for plotting afterwards
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data.dataset.tensors[0].shape[0]))
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:    # Random x: data loader will shuffle the x, y pairs in every epoch
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors  # Sequential x: x, y pairs in original sequence.
        with torch.no_grad():
            model.fit_coeff(x, y_actual)
        # Get the error rate for the whole batch:
        y_pred = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        errors["mse"].append(mse)
        errors["rmse"].append(rmse)
        errors["PE"].append(perc_loss)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f} ={:.2f}%'
                  .format(t, mse, rmse, perc_loss))
    # End of training, so graph the results:
    if show_plots:
        plot_errors(errors, metric)
        y_actual = data.dataset.tensors[1]
        y_pred = model(data.dataset.tensors[0])
        plot_results(y_actual, y_pred)
    return x, y_pred, y_actual


def train_anfis_with_cv(model, data, optimizer, criterion,
                     epochs=500, show_plots=False, metric="mse"):
    """
    Train the given model using training data, meanwhile use test data to test the model in every epoch.
    :param model: [AnfisNet], the given model.
    :param data: [DataLoader] or [list], training data or list of training data and test data.
    :param optimizer: [optim], optimizer.
    :param criterion: [Loss], loss function.
    :param epochs: [int], no. of training epoch.
    :param show_plots: [boolean], show the learning curve.
    :param metric: [str],  error used to plot curve.
    :return:
    x_train: [tensor], x value from data set.
    y_actual: [tensor], Target y value.
    y_pred: [tensor], Prediction by x_train using trained model.
    """
    test = False    # Using test data during training or not
    if isinstance(data, list):
        test = True
        train_data, test_data = data
        print('### Training for {} epochs, training size = {} cases, test size = {}'.
              format(epochs, train_data.dataset.tensors[0].shape[0], test_data.dataset.tensors[0].shape[0]))
    else:
        train_data = data
        print('### Training for {} epochs, training size = {} cases'.
              format(epochs, train_data.dataset.tensors[0].shape[0]))
    errors = {"mse": [], "rmse": [], "PE": []}  # Keep a dict of these for plotting afterwards

    for t in range(epochs):
        # Process each mini-batch in turn:
        for x_train, y_train_actual in train_data:  # Random x: data loader will shuffle the x, y pairs in every epoch
            y_train_pred = model(x_train)
            # Compute and print loss
            loss = criterion(y_train_pred, y_train_actual)
            # Zero gradients, perform a backward pass, and update the weights(Premise parameter).
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients(Consequent parameter) based on all data:
        x_train, y_train_actual = train_data.dataset.tensors    # Training data in sequence
        with torch.no_grad():
            model.fit_coeff(x_train, y_train_actual)
        # Get the error rate for the whole batch:
        y_train_pred = model(x_train)
        mse, rmse, perc_loss = calc_error(y_train_pred, y_train_actual)
        if test:
            x_test, y_test_actual = test_data.dataset.tensors   # test data in sequence
            # for x_test, y_test_actual in test_data:   random test data set
            y_test_pred = model(x_test)
            test_mse, test_rmse, test_perc_loss = calc_error(y_test_pred, y_test_actual)
            mse = (mse, test_mse)
            rmse = (rmse, test_rmse)
            perc_loss = (perc_loss, test_perc_loss)
        errors["mse"].append(mse)
        errors["rmse"].append(rmse)
        errors["PE"].append(perc_loss)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            if not test:
                print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f}, PE ={:.2f}%'
                      .format(t, mse, rmse, perc_loss))
            else:
                train_error = errors[metric][-1][0]
                test_error = errors[metric][-1][1]
                print(f'epoch {t}: Train {metric}={train_error:.5f}, Test {metric}={test_error:.5f}')
    x_train = train_data.dataset.tensors[0]
    y_actual = train_data.dataset.tensors[1]
    y_pred = model(x_train)
    if show_plots:
        plot_errors(errors, metric, test)
        plot_results(y_actual, y_pred)
    return x_train, y_actual, y_pred


def train_anfis(model, data, epochs=500, show_plots=False, metric="rmse"):
    '''
        Train the given model using the given (x,y) data.
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    return train_anfis_with(model, data, optimizer, criterion, epochs, show_plots, metric)


def train_anfis_cv(model, data, epochs=500, show_plots=False, metric="rmse"):
    """
    Train the given model using the given (x,y) data.
    :param model: [AnfisNet], the given model.
    :param data: [DataLoader] or [list], training data or list of training data and test data.
    :param epochs: [int], no. of training epoch.
    :param show_plots: [boolean], show the learning curve.
    :param metric: [str], error used to plot curve.
    :return:
    see return of train_anfis_with_cv.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    return train_anfis_with_cv(model, data, optimizer, criterion, epochs, show_plots, metric)


if __name__ == '__main__':
    # x = torch.arange(1, 100, dtype=dtype).unsqueeze(1)
    # y = torch.pow(x, 3)
    # model, errors = linear_model(x, y, 100)
    # plot_errors(errors)
    # plot_results(y, model(x))
    s = {"mse": [1], "rmse": [(2, 3), (5, 6)]}
    print(isinstance(s["mse"][0], int))
    print(isinstance(s["mse"][0], tuple))
    print(isinstance(s["rmse"][0], tuple))
    print(isinstance(s["rmse"][0], int))
