import torch.nn as nn
import torch


class CNNClassifier(nn.Module):
    def __init__(self, n_channels, window_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=(2, 50), stride=1, padding=1)
        # keep track of dimensions
        H = (n_channels + 1 + 1 - 2)/1 + 1
        W = (window_size + 1 + 1 - 50)/1 + 1
        self.dropout1 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        W = int(W/2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                               stride=2, padding=1)
        H = int((H + 1 + 1 - 3)/2 + 1)
        W = int((W + 1 + 1 - 3)/2 + 1)
        self.dropout2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        W = int(W/3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3),
                               stride=2, padding=1)
        H = int((H + 1 + 1 - 3)/2 + 1)
        W = int((W + 1 + 1 - 3)/2 + 1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        W = int(W/3)

        self.fc1 = nn.Linear(int(64*H*W), 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.dropout1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.dropout1(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.squeeze(x)
        return x


class GRUClassifier(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_layers, window_size, dropout=0.1):
        super().__init__()

        self.gru = nn.GRU(input_channels, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim*window_size, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        # Input shape: (batch_size, input_channels, sequence_length)
        # Transpose the input to have the format:
        # (batch_size, sequence_length, input_channels)
        x = x.transpose(1, 2)
        x, h = self.gru(x)  # batch x sequence length x hidden layers
        x = torch.reshape(x, (x.size(0), -1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.squeeze(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_layers, window_size, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(input_channels, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Transpose the input to have the format:
        # (batch_size, sequence_length, input_channels)
        x = x.transpose(1, 2)
        x, (hn, cn) = self.lstm(x)  # batch x sequence length x hidden layers
        out = torch.reshape(hn[-1], (x.size(0), -1))
        out = self.fc(out)
        out = torch.squeeze(out)
        return out
