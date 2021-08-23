#!/usr/bin/env python3
# @ -*- coding: utf-8 -*-
# @Time:   2021/6/25 21:39
# @Author:  Wei XU <samxu0823@gmail.com>


import numpy as np

import anfis
from membership import BellMembFunc, make_bell_mfs, make_tri_mfs, make_gauss_mfs


def my_model_m1(rules='less'):
    """
    Initialization of the ANFIS regressor for m1 prediction. Single output.
    Fuzzy reasoning and rules can be modified according to human-expertise.
    :param rules: one rule for each case (less) or three rules for each case (more)
    :return:
    model: Initialized ANFIS
    """
    if rules == 'less':
        invardefs = [
            ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 5)))]

        rules = [[0, 2, 2, 6, 0, 2, 0, 6, 0, 0], [3, 3, 0, 2, 3, 0, 3, 2, 3, 3],
                 [2, 4, 1, 1, 2, 1, 2, 1, 2, 2], [1, 0, 1, 5, 1, 2, 1, 5, 1, 1],
                 [0, 3, 2, 4, 0, 2, 0, 4, 0, 0], [4, 2, 0, 0, 4, 0, 4, 0, 4, 4],
                 [1, 0, 1, 2, 1, 2, 1, 2, 1, 1], [2, 1, 1, 3, 2, 1, 2, 3, 2, 2]]

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
            ('x9', make_gauss_mfs(0.4, np.linspace(0, 1, 10))), ]

        rules = [[1, 8, 5, 2, 7, 4, 6, 2, 7, 7], [0, 6, 4, 1, 6, 4, 8, 1, 6, 6],
                 [1, 6, 4, 2, 5, 4, 8, 1, 5, 5], [1, 7, 5, 2, 6, 4, 7, 1, 6, 6],
                 [1, 7, 5, 2, 6, 4, 7, 2, 6, 6], [0, 6, 4, 1, 5, 4, 8, 1, 5, 5],
                 [1, 6, 4, 2, 6, 4, 7, 1, 6, 6], [1, 7, 4, 2, 6, 4, 7, 1, 6, 6],
                 [0, 5, 1, 1, 4, 1, 0, 1, 4, 4], [0, 5, 0, 0, 4, 0, 5, 0, 4, 4],
                 [1, 3, 0, 1, 3, 0, 5, 1, 3, 3], [0, 3, 1, 0, 2, 1, 1, 0, 2, 2],
                 [0, 2, 1, 1, 2, 0, 3, 1, 2, 2], [0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0, 0, 0, 1, 1], [1, 6, 0, 1, 5, 0, 5, 1, 5, 5],
                 [6, 9, 9, 7, 9, 9, 9, 7, 9, 9], [1, 8, 9, 3, 7, 7, 9, 2, 7, 7],
                 [1, 8, 9, 2, 7, 8, 9, 2, 7, 7], [4, 9, 9, 6, 9, 8, 9, 5, 9, 9],
                 [4, 9, 9, 5, 9, 7, 9, 5, 9, 9], [1, 8, 9, 3, 8, 8, 9, 2, 8, 8],
                 [9, 9, 9, 9, 9, 8, 9, 9, 9, 9], [1, 7, 8, 2, 7, 8, 9, 2, 7, 7]]
    outvars = ['m1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)
    model.set_rules(rules)
    return model


def my_model_k1():
    """
    Initialization of the ANFIS regressor for k1 prediction. Single output.
    Fuzzy reasoning and rules can be modified according to human-expertise.
    :return:
    model: Initialized ANFIS
    """
    invardefs = [
        ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 2))),
        ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
        ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
        ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
        ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
        ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
    ]
    outvars = ['k1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)

    rules = [[1, 1, 2, 2, 4, 1, 4, 0, 1, 6], [0, 0, 0, 0, 0, 0, 0, 3, 4, 2],
             [1, 3, 2, 1, 5, 2, 5, 2, 0, 0], [0, 0, 1, 1, 1, 0, 1, 1, 3, 1],
             [1, 3, 2, 2, 5, 2, 5, 0, 0, 3], [0, 2, 0, 0, 2, 1, 2, 4, 3, 4],
             [0, 2, 1, 1, 3, 1, 3, 1, 2, 4], [1, 1, 2, 1, 4, 1, 4, 2, 1, 5]]
    model.set_rules(rules)
    return model


def my_model_c1(rules='less'):
    """
    Initialization of the ANFIS regressor for c1 prediction. Single output.
    Fuzzy reasoning and rules can be modified according to human-expertise.
    :param rules: one rule for each case (less) or three rules for each case (more)
    :return:
    model: Initialized ANFIS
    """
    if rules == 'less':
        invardefs = [
            ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
            ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
            ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
            ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
        ]

        # 14.07
        rules = [[1, 0, 0, 6, 5, 1, 6, 0, 0, 6], [0, 4, 0, 2, 2, 0, 2, 0, 3, 2],
                 [2, 5, 3, 1, 1, 5, 1, 4, 5, 1], [0, 1, 0, 5, 4, 0, 5, 0, 1, 5],
                 [2, 2, 2, 4, 0, 4, 4, 3, 3, 4], [1, 6, 1, 0, 2, 3, 0, 1, 6, 0],
                 [1, 4, 1, 2, 3, 3, 2, 2, 4, 2], [1, 3, 0, 3, 4, 2, 3, 0, 2, 3]]

    elif rules == 'more':
        invardefs = [
            ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
            ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 10))),
            ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 8))),
            ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 8))),
            ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 9))),
            ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
            ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 11))), ]

        rules = [[4, 1, 6, 4, 0, 4, 0, 1, 4, 4], [3, 0, 8, 4, 3, 4, 3, 4, 4, 7],
                 [4, 5, 8, 4, 5, 4, 5, 6, 4, 8], [3, 0, 7, 4, 2, 4, 2, 3, 4, 6],
                 [4, 4, 8, 4, 4, 4, 4, 5, 4, 7], [4, 3, 9, 4, 7, 4, 7, 8, 4, 10],
                 [4, 3, 9, 4, 6, 4, 6, 7, 4, 9], [4, 2, 6, 4, 1, 4, 1, 2, 4, 5],
                 [0, 2, 5, 0, 0, 0, 1, 0, 0, 0], [4, 2, 1, 4, 0, 4, 1, 1, 4, 3],
                 [2, 2, 4, 2, 0, 2, 1, 1, 2, 2], [0, 2, 2, 0, 0, 0, 1, 1, 0, 2],
                 [0, 2, 5, 0, 0, 0, 1, 0, 0, 1], [3, 2, 0, 3, 0, 3, 1, 1, 3, 3],
                 [0, 2, 3, 0, 0, 0, 1, 0, 0, 2], [1, 2, 4, 1, 0, 1, 1, 1, 1, 2]]
    outvars = ['c1']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, grid=False)
    model.set_rules(rules)
    return model


def my_model_class1():
    """
    Initialization of the ANFIS classifier for virtual generic model. Multi-output.
    Fuzzy reasoning and rules can be modified according to human-expertise.
    :return:
    model: Initialized ANFIS
    """
    invardefs = [
        ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
        ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
        ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 7))),
        ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
        ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 5))),
        ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 2))),
        ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 7)))]

    outvars = ['dc1', 'dc2', 'dc3', 'dc4', 'dc5', 'dc6', 'dc7', 'dc8']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, hybrid=False, grid=False)
    # 20.07
    rules = [[0, 0, 1, 2, 6, 0, 0, 0, 1, 3], [3, 0, 0, 0, 2, 0, 3, 2, 0, 0],
             [2, 3, 2, 1, 1, 2, 2, 2, 1, 4], [1, 0, 0, 1, 5, 0, 1, 1, 0, 2],
             [0, 2, 2, 2, 4, 1, 0, 0, 1, 1], [4, 1, 1, 0, 0, 0, 4, 2, 0, 5],
             [1, 1, 1, 1, 2, 0, 1, 4, 0, 6], [2, 0, 1, 1, 3, 0, 2, 3, 1, 3]]

    model.set_rules(rules, hybrid=False)
    return model


def classifier_rig(window="small"):
    """
    Initialization of the ANFIS classifier for test rig. Multi-output.
    Fuzzy reasoning and rules can be modified according to human-expertise.
    :param window: "small", or "large" window size
    :return:
    model: Initialized ANFIS
    """
    if window == "small":
        invardefs = [
            ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 2))),
            ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 6))),
            ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 2))),
            ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 2))),
            ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ]
        # 22.07 small
        rules = [[1, 0, 0, 2, 2, 0, 0, 2, 0, 2], [0, 2, 2, 1, 0, 1, 1, 1, 1, 1],
                 [0, 5, 1, 0, 1, 2, 1, 0, 1, 0], [0, 1, 2, 0, 1, 2, 1, 0, 1, 0],
                 [0, 4, 3, 1, 0, 1, 1, 1, 1, 1], [0, 3, 3, 1, 0, 1, 1, 1, 1, 1]]
    else:
        invardefs = [
            ('x0', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x1', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x2', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x3', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x4', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x5', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x6', make_gauss_mfs(0.5, np.linspace(0, 1, 2))),
            ('x7', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
            ('x8', make_gauss_mfs(0.5, np.linspace(0, 1, 4))),
            ('x9', make_gauss_mfs(0.5, np.linspace(0, 1, 3))),
        ]
        # 22.07 Large
        rules = [[2, 0, 0, 3, 0, 0, 0, 2, 0, 2], [0, 2, 3, 0, 1, 1, 1, 1, 3, 1],
                 [1, 1, 1, 2, 3, 2, 1, 0, 1, 0], [1, 2, 2, 0, 2, 3, 1, 0, 2, 0],
                 [0, 2, 3, 1, 1, 1, 1, 1, 3, 1]]
    outvars = ['nc', 'dc1', 'dc2', 'dc3', 'dc4', 'dc5', 'dc6']
    model = anfis.AnfisNet('My_Anfis', invardefs, outvars, hybrid=False, grid=False)
    model.set_rules(rules, hybrid=False)
    return model
