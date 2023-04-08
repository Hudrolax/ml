import torch.nn as nn
import torch


class CustomNN(nn.Module):
    def __init__(self, observation_space) -> None:
        super(CustomNN, self).__init__()
        channels = observation_space.shape[1]

        # define convolution layers
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=16, kernel_size=8, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        # dopout layer
        self.dropout = nn.Dropout(p=0.2)

        # maxpooling layer
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)

        # flatten layer
        self.flatten = nn.Flatten()

        # Set network output size
        test_tensor = torch.randn(1, observation_space.shape[0], channels)
        x = self.forward(test_tensor)
        self.output_size = x.shape[1] 

    def forward(self, x):
        x = x.transpose(1, 2)

        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.maxpool1d(x)
        x = self.flatten(x)
        return x


class CustomCNN(nn.Module):
    def __init__(self, observation_space) -> None:
        super(CustomCNN, self).__init__()
        channels = observation_space.shape[1]

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=16, kernel_size=8, stride=1),
            nn.Dropout1d(p = 0.5),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.Dropout1d(p = 0.5),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.Dropout1d(p = 0.5),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # Set network output size
        test_tensor = torch.randn(1, observation_space.shape[0], channels)
        x = self.forward(test_tensor)
        self.output_size = x.shape[1] 

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.cnn(x)


def mlp_net(in_dim) -> nn.Sequential:
    return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
        )