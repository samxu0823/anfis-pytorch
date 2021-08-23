#!/usr/bin/env python3
# @ -*- coding: utf-8 -*-
# @Time:   2021/6/27 13:20
# @Author:  Wei XU <samxu0823@gmail.com>

import pickle
import os
import numpy as np


def read(path, window_size=4, label=0, save_folder="nc"):
    """
    Read the raw measurement data (pickle files) from raspberry pi and convert it to
    the csv according to different window sizes.
    :param path: The path of the raw measurement data
    :param window_size: The measurement will be divided into sub-signal by window (Time segment)
    :param label: Categorical label according to damage case
    :param save_folder: Save path
    :return:
    """
    files = os.listdir(path)
    no = len(files)
    n_dataset = no / window_size if no % window_size == 0 else (no // window_size) + 1
    print(f"{no} data set will be separated in to {int(n_dataset)} sub set.")
    data = np.zeros((1, 4))
    set_index = 1
    folder_path = f"my_data_set/rig/processed_data/{save_folder}"
    folder = os.path.exists(folder_path)
    if not folder:
        os.makedirs(folder_path)
    for i, file in enumerate(files):
        print(f"current iteration: {i+1}")
        f = open(f"{path}/{file}", "rb")
        a = pickle.load(f)
        data_label = np.hstack((a["data"], label * np.ones((a["data"].shape[0], 1))))
        data = np.vstack((data, data_label))
        if (i + 1) % window_size == 0 or i + 1 == no:    # generate the next batch or last data set
            data = np.delete(data, 0, 0)
            print(f"{set_index} data set has been generated with shape of {data.shape}.")
            np.savetxt(f"my_data_set/rig/processed_data/{save_folder}/data{set_index}", data)
            set_index += 1
            data = np.zeros((1, 4))


if __name__ == "__main__":
    path = "my_data_set/rig/raw_measurement/Screw_4_05Nm"
    read(path, window_size=2, label=6, save_folder="dc6")