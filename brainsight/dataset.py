import torch
import numpy as np
import os
import mat73
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, sosfiltfilt

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


def bandpass_filter(data, bandwidths):
    """
    Applies bandpass filter to data.
    """

    fs = 2048
    order = 5
    bandw_trials = np.zeros((np.shape(data)[0], len(bandwidths), np.shape(data)[2]))
    for i in range(len(bandwidths)):
        lowcut = bandwidths[i][0]
        highcut = bandwidths[i][1]
        sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
        for trial in range(np.size(data, 0)):
            temp = sosfiltfilt(sos, data[trial, :, :], axis=1)
            bandw_trials[trial, i, :] = np.mean(temp, axis=0)

    return bandw_trials


def sliding_window(data, window_size, step_size=128):
    """
    Cuts data into windows of given size and appends them.
    """
    subsequences = []
    input_scaler = StandardScaler()
    for i in range(0, np.size(data, 1) - window_size + 1, step_size):
        seq = data[:, i:i + window_size]
        tr_seq = input_scaler.fit_transform(seq.transpose())
        tr_seq = tr_seq.transpose()
        subsequences.append(tr_seq)
    return np.array(subsequences)


def windowing(signals, labels, window_size, step_size):
    """
    Applies sliding window to each trial to cut the data based on window size
    and step size.
    Output is in the shape: number of windows x channels x window size
    """
    windowed_signals = np.zeros((1, np.size(signals, 1), window_size))
    windowed_labels = np.zeros((1, 1))
    for trial in range(np.size(signals, 0)):
        seq = sliding_window(signals[trial, :, :], window_size, step_size)
        windowed_signals = np.append(windowed_signals, seq, 0)
        windowed_labels = np.append(windowed_labels,
                                    np.ones((np.size(seq, 0), 1))*labels[trial], axis=0)

    windowed_signals = windowed_signals[1:, :, :]
    windowed_labels = np.squeeze(windowed_labels[1:])

    return windowed_signals, windowed_labels


def load_files(files, directory, data_type, channels):
    """
    Loads files from directory based on data type and creates labels.
    """
    final_data = np.zeros((channels, 2048, 1))
    final_labels = np.zeros((1, 1), dtype=int)

    for i in files:
        # load file
        signals = mat73.loadmat(os.path.join(directory, i))[data_type]
        if data_type == 'emg':
            # TODO: group channels to increase sample size
            chs = [35, 90, 150, 180, 230]
            signals = signals[chs, :, :]
        # append
        final_data = np.append(final_data, signals, axis=2)

        file_prefix = i[0]
        # encode class based on file
        np_labels = np.zeros((np.shape(signals)[2], 1), dtype=int)
        if file_prefix == 'b':    # baseline
            np_labels[:] = 0
        elif file_prefix == 'p':    # preparatory
            np_labels[:] = 1

        final_labels = np.append(final_labels, np_labels, axis=0)

    final_data = final_data[:, :, 1:]
    final_labels = np.squeeze(final_labels[1:])

    return final_data, final_labels


def get_data(directory: str, subjects: list, data_type: str, cross_val: int,
             channels: int = 256):
    """
    Loads all the data and labels from given subjects. Creates cuts based on cross-validation folds.
    Returns data and cut locations.
    """

    classes = ['base', 'prep']
    all_mat_files = [f"{i}_S{s}_all_{data_type}.mat" for s in subjects for i in classes]

    final_data_all, final_labels_all = load_files(all_mat_files, directory=directory,
                                                  data_type=data_type, channels=channels)

    total_trials = np.size(final_data_all, 2)

    # shuffle before dividing into folds
    rand_idxs = np.random.permutation(total_trials)
    final_data = final_data_all[:, :, rand_idxs]
    final_label = final_labels_all[rand_idxs]

    # reshape to have first dim nr of trials
    final_data = np.transpose(final_data, (2, 0, 1))

    size_fold = int(total_trials/cross_val)
    cuts = [size_fold*i for i in range(0, cross_val)]
    cuts.append(total_trials)

    return final_data, final_label, cuts


def get_dataloaders_fold(final_data, final_labels, fold, cuts_arr, batch_size,
                         data_window: int = 1024, step_size: float = 128):
    """
    Cuts the full dataset into train and test based on fold number and creates
    Dataloaders.
    Returns dataloaders for a specific fold.
    """
    test_idx_start = cuts_arr[fold-1]
    test_idx_end = cuts_arr[fold]

    test_data = final_data[test_idx_start:test_idx_end, :, :]
    test_labels = final_labels[test_idx_start:test_idx_end]

    train_data = np.delete(final_data, np.arange(test_idx_start, test_idx_end), 0)
    train_labels = np.delete(final_labels, np.arange(test_idx_start, test_idx_end), 0)

    # bandpass filter data in given bandwidths
    bandwidths = [[0.01, 10], [10, 14], [14, 18], [18, 22], [22, 26], [26, 32], [32, 40], [45, 55]]
    train_data = bandpass_filter(train_data, bandwidths)
    test_data = bandpass_filter(test_data, bandwidths)

    # window
    test_data_windowed, test_labels_windowed = windowing(test_data, test_labels, data_window, step_size)
    train_data_windowed, train_labels_windowed = windowing(train_data, train_labels, data_window, step_size)

    # transform into dataset
    train_dataset = CustomDataset(train_data_windowed, train_labels_windowed)
    test_dataset = CustomDataset(test_data_windowed, test_labels_windowed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    channels = np.size(train_data_windowed, 1)
    return train_loader, test_loader, channels
