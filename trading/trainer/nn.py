import torch.nn as nn
import torch


class CustomCNN(nn.Module):
    def __init__(self, observation_space) -> None:
        super(CustomCNN, self).__init__()
        channels = observation_space.shape[-1]

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=16, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(p = 0.1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(p = 0.1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(p = 0.1),
            nn.Flatten()
        )

        # Set network output size
        test_tensor = torch.randn(1, observation_space.shape[0], channels)
        x = self.forward(test_tensor)
        self.output_size = x.shape[1] 

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.cnn(x)


class CustomCNN2d(nn.Module):
    def __init__(self, observation_space) -> None:
        super(CustomCNN2d, self).__init__()
        channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1), 
            nn.Dropout2d(p = 0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Dropout2d(p = 0.1),
            nn.Flatten()
        )

        # Set network output size
        test_tensor = torch.randn(1, *observation_space.shape)
        x = self.forward(test_tensor)
        self.output_size = x.shape[1] 

    def forward(self, x):
        # x = x.transpose(1, 2)
        return self.cnn(x)


def mlp_net(in_dim) -> nn.Sequential:
    return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )