import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, n_channels, window_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=(3, 5), stride=1, padding=1)
        H = (n_channels + 1 + 1 - 3)/1 + 1
        W = (window_size + 1 + 1 - 5)/1 + 1
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        W = W/2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                               stride=2, padding=1)
        H = int((H + 1 + 1 - 3)/2 + 1)
        W = (W + 1 + 1 - 3)/2 + 1
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        W = int(W/3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3),
                               stride=2, padding=1)
        H = int((H + 1 + 1 - 3)/2 + 1)
        W = (W + 1 + 1 - 3)/2 + 1
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        W = int(W/3)

        self.fc1 = nn.Linear(int(32*H*W), 512)
        self.fc2 = nn.Linear(512, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)
        return x
