import os
import torch
import wandb
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from brainsight.dataset import get_data, get_dataloaders_fold
from brainsight.model import CNNClassifier, GRUClassifier, LSTMClassifier


def train_func_wandb(subject, data_type, model_type, data_window, step_size, optimiser, lr, batch_size,
                     hidden_dim=None, num_layers=None):
    """ Train a model on a given data set and track results on wandb. """

    device = "cuda"
    cross_val = 5
    channels = 5
    num_epochs = 50
    dir_data = os.path.join(os.path.normpath(os.getcwd() + os.sep), "data\\processed")
    optimizers = {'Adam': (torch.optim.Adam, {})}
    criterion = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    if len(subject) > 1:
        name = "All_" + data_type
    else:
        name = "S" + str(subject[0]) + "_" + data_type

    data, labels, cuts = get_data(directory=dir_data, subjects=subject, data_type=data_type,
                                  cross_val=cross_val, channels=channels)
    accuracy_scores = []
    for fold in range(1, cross_val+1):

        train_loader, test_loader, channels = get_dataloaders_fold(data, labels, fold,
                                                                   cuts, batch_size=batch_size,
                                                                   data_window=data_window,
                                                                   step_size=step_size)
        print(f'Fold {fold} starting')
        if model_type == "CNN":
            model = CNNClassifier(channels, data_window)
        elif model_type == "LSTM":
            model = LSTMClassifier(channels, hidden_dim, num_layers, data_window)
        elif model_type == "GRU":
            model = GRUClassifier(channels, hidden_dim, num_layers,
                                  data_window, 0.2)

        model.to(device)
        optimizer_class, extra_kwargs = optimizers.get(optimiser)
        optimizer = optimizer_class(model.parameters(), lr=lr, **extra_kwargs)
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            train_loss = 0.0
            model.train()
            for (train_data, train_target) in train_loader:
                train_data, train_target = train_data.to(device), train_target.to(device)

                optimizer.zero_grad()
                outputs = model(train_data)
                # predicted = torch.round(sigmoid(outputs))
                loss = criterion(outputs, train_target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * train_data.size(0)

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            # print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}')
            wandb.log({"Train loss: " + name: train_loss})

            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                true_labels = np.empty((1))
                predicted_labels = np.empty((1))
                output_arr = np.empty((1))

                for (test_data, test_target) in test_loader:
                    test_data, test_target = test_data.to(device), test_target.to(device)

                    outputs = model(test_data)
                    output_labels = torch.round(sigmoid(outputs))

                    true_labels = np.append(true_labels, test_target.cpu().numpy(), 0)
                    predicted_labels = np.append(predicted_labels, output_labels.cpu().numpy(), 0)
                    output_arr = np.append(output_arr, sigmoid(outputs).cpu().numpy(), 0)

                    loss = criterion(outputs, test_target)
                    test_loss += loss.item() * test_data.size(0)

                test_loss /= len(test_loader.dataset)
                test_losses.append(test_loss)
                true_labels = true_labels[1:].astype(int)
                predicted_labels = predicted_labels[1:].astype(int)
                output_arr = output_arr[1:]

                accuracy = accuracy_score(true_labels, predicted_labels)
                wandb.log({"Test loss: " + name: test_loss})
                wandb.log({"Test accuracy: " + name: accuracy})

            # print(f'Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}, accuracy: {accuracy:.4f}')

        accuracy_scores.append(accuracy)
    print(f'Accuracies: {accuracy_scores}')

    # save to wandb table
    table_summary = wandb.Table(
        columns=["Subject", "data", "model", "optimiser", "lr", "batch size", "window size", "step size",
                 "accuracies", "epochs", "loss type"])

    table_summary.add_data(subject, data_type, model_type, optimiser, lr, batch_size, data_window, step_size,
                           accuracy_scores, num_epochs, criterion)

    wandb.log({"Model results": table_summary})

    wandb.log({"accuracy_mean": np.mean(accuracy_scores)})
