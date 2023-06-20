import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
from brainsight.dataset import CustomDataset, sliding_window
from brainsight.model import CNNClassifier


def create_data():
    window_size = 1024
    step_size = 128

    # create dataset
    raw_signals = np.random.rand(256, 2048, 41)  # channels x samples x trials

    signals = np.zeros((1, 256, window_size))
    for trial in range(np.size(raw_signals, 2)):
        signals = np.append(signals, sliding_window(raw_signals[:, :, trial], window_size, step_size), 0)
    signals = signals[1:, :, :]  # number of windows x channels x window size

    labels = np.zeros(np.shape(signals)[0])  # number of windows
    labels[0:int(np.size(signals, 0)/2)] = 1

    return signals, labels, window_size, step_size


if __name__ == "__main__":
    signals, labels, window_size, step_size = create_data()
    num_epochs = 20

    dataset = CustomDataset(signals, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNNClassifier(256, window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for train_signals, train_labels in train_loader:
            # signals is batch_size (number of samples in 1 batch) x channels x window size
            train_labels = train_labels.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = model(train_signals)
            _, predicted = torch.max(outputs, dim=1)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_signals.size(0)

        train_loss /= len(train_loader.dataset)
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}')

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            true_labels = np.empty((1))
            predicted_labels = np.empty((1))

            for test_signals, test_labels in train_loader:
                test_labels = test_labels.type(torch.LongTensor)
                outputs = model(test_signals)
                output_labels = np.argmax(outputs, axis=1)

                true_labels = np.append(true_labels, test_labels.numpy(), 0)
                predicted_labels = np.append(predicted_labels, output_labels.numpy(), 0)

                loss = criterion(outputs, test_labels)
                test_loss += loss.item() * test_signals.size(0)

            test_loss /= len(test_loader.dataset)
            true_labels = true_labels[1:].astype(int)
            predicted_labels = predicted_labels[1:].astype(int)

            accuracy = accuracy_score(true_labels, predicted_labels)
        print(f'Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}, accuracy: {accuracy:.4f}')
