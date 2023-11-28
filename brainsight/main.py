import torch.nn as nn
import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score
from brainsight.dataset import get_data, get_dataloaders_fold
from brainsight.model import CNNClassifier, GRUClassifier, LSTMClassifier


if __name__ == "__main__":
    dir_data = os.path.join(os.path.normpath(os.getcwd() + os.sep), "data\\processed")
    device = 'cuda'
    num_epochs = 50
    channels = 5
    window_size = 1024
    subjects = [5]
    cross_val = 5
    data_type = 'emg'

    data, labels, cuts = get_data(directory_emg=dir_data, subjects=subjects, data_type=data_type,
                                  cross_val=cross_val, channels=channels)

    accuracy_scores = []
    for fold in range(1, cross_val+1):
        # model = CNNClassifier(channels, window_size)
        model = GRUClassifier(channels, 2, 10, window_size, 0.1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        criterion = nn.BCEWithLogitsLoss()
        sigmoid = nn.Sigmoid()
        train_losses = []
        test_losses = []

        train_loader, test_loader = get_dataloaders_fold(data, labels, fold, cuts, batch_size=32,
                                                         data_window=window_size, step_size=128)
        print(f'Fold {fold} starting')
        for epoch in range(num_epochs):
            train_loss = 0.0
            model.train()
            for batch_idx, (train_data, train_target) in enumerate(train_loader):
                train_data, train_target = train_data.to(device), train_target.to(device)

                optimizer.zero_grad()
                outputs = model(train_data)
                predicted = torch.round(sigmoid(outputs))
                loss = criterion(outputs, train_target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * train_data.size(0)

            train_loss /= len(train_loader.dataset)
            print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}')

            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                true_labels = np.empty((1))
                predicted_labels = np.empty((1))
                output_arr = np.empty((1))

                for batch_idx, (test_data, test_target) in enumerate(test_loader):
                    test_data, test_target = test_data.to(device), test_target.to(device)

                    outputs = model(test_data)
                    output_labels = torch.round(sigmoid(outputs))

                    true_labels = np.append(true_labels, test_target.cpu().numpy(), 0)
                    predicted_labels = np.append(predicted_labels, output_labels.cpu().numpy(), 0)
                    output_arr = np.append(output_arr, sigmoid(outputs).cpu().numpy(), 0)

                    loss = criterion(outputs, test_target)
                    test_loss += loss.item() * test_data.size(0)

                test_loss /= len(test_loader.dataset)
                true_labels = true_labels[1:].astype(int)
                predicted_labels = predicted_labels[1:].astype(int)
                output_arr = output_arr[1:]

                accuracy = accuracy_score(true_labels, predicted_labels)

            print(f'Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}, accuracy: {accuracy:.4f}')

        accuracy_scores.append(accuracy)
    print(f'Accuracies: {accuracy_scores}')
