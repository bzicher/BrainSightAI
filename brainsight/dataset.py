import torch
from torch.utils.data import Dataset, DataLoader
# import pandas as pd
import numpy as np
import os
import mat73

torch.manual_seed(42)


class CustomDataset(Dataset):
    """
    Converts data and returns it in Dataset format.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float)
        y = torch.tensor(self.labels[index], dtype=torch.float)
        return x, y

    def __len__(self):
        return len(self.data)


def sliding_window(data, window_size, step_size=128):
    """
    Cuts and appends data
    """
    subsequences = []
    for i in range(0, np.size(data, 1) - window_size + 1, step_size):
        subsequences.append(data[:, i:i + window_size])
    return np.array(subsequences)


def windowing(signals, window_size, step_size):
    """
    Applies sliding window to each trial to cut the data based on window size and step size.
    Output is in the shape: number of windows x channels x window size
    """
    windowed_signals = np.zeros((1, np.size(signals, 0), window_size))
    for trial in range(np.size(signals, 2)):
        windowed_signals = np.append(windowed_signals,
                                     sliding_window(signals[:, :, trial], window_size, step_size), 0)
    windowed_signals = windowed_signals[1:, :, :]

    return windowed_signals


def load_files(files, directory, data_type, window_size, step_size, channels):
    final_data = np.zeros((1, channels, window_size))
    final_labels = np.zeros((1, 1), dtype=int)

    for i in files:
        signals = mat73.loadmat(os.path.join(directory, i))[data_type]
        # signals = signals[100:105, :, :]
        np_windowed = windowing(signals, window_size, step_size)
        final_data = np.append(final_data, np_windowed, axis=0)

        file_prefix = i[0]
        # encode class
        np_windowed_labels = np.zeros((np.shape(np_windowed)[0], 1), dtype=int)
        if file_prefix == 'b':    # baseline
            np_windowed_labels[:] = 0
        elif file_prefix == 'p':    # preparatory
            np_windowed_labels[:] = 1

        final_labels = np.append(final_labels, np_windowed_labels, axis=0)

    final_data = final_data[1:, :, :]
    final_labels = np.squeeze(final_labels[1:])

    return final_data, final_labels


def get_data(directory_emg: str, subjects: list, data_type: str, cross_val: int,
             channels: int = 256, data_window: int = 1024, step_size: float = 128):
    """
    Loads all the data and labels from given subjects. Creates cuts based on cross-validation folds.
    """

    classes = ['base', 'prep']
    all_mat_files = [f"{i}_S{s}_all_{data_type}.mat" for s in subjects for i in classes]

    final_data_all, final_labels_all = load_files(all_mat_files, directory=directory_emg,
                                                  data_type=data_type, window_size=data_window,
                                                  step_size=step_size,
                                                  channels=channels)

    total_windows = np.size(final_data_all, 0)

    # shuffle before dividing into folds
    rand_idxs = np.random.permutation(total_windows)
    final_data = final_data_all[rand_idxs]
    final_label = final_labels_all[rand_idxs]

    size_fold = int(total_windows/cross_val)
    cuts = [size_fold*i for i in range(0, cross_val)]
    cuts.append(total_windows)

    return final_data, final_label, cuts


def get_dataloaders_fold(final_data, final_labels, fold, cuts_arr, batch_size):
    """
    Returns dataloaders for a specific fold.
    Cuts the full dataset into train and test based on fold number and creates
    Dataloaders.
    """
    test_idx_start = cuts_arr[fold-1]
    test_idx_end = cuts_arr[fold]

    test_data = final_data[test_idx_start:test_idx_end, :, :]
    test_labels = final_labels[test_idx_start:test_idx_end]

    train_data = np.delete(final_data, np.arange(test_idx_start, test_idx_end), 0)
    train_labels = np.delete(final_labels, np.arange(test_idx_start, test_idx_end), 0)

    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
