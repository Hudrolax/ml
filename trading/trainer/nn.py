import torch.nn as nn
import torch


class CustomCNN1d(nn.Module):
    def __init__(self, observation_space) -> None:
        super(CustomCNN1d, self).__init__()
        channels = observation_space.shape[-1]

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

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
            nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1), 

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),

            nn.Flatten()
        )

        # Set network output size
        test_tensor = torch.randn(1, *observation_space.shape)
        x = self.forward(test_tensor)
        self.output_size = x.shape[1] 

    def forward(self, x):
        # x = x.transpose(1, 2)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        return self.cnn(x)


class CustomCNN1dReverse(nn.Module):
    def __init__(self, observation_space) -> None:
        super(CustomCNN1dReverse, self).__init__()
        channels = observation_space.shape[-1]

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten()
        )

        # Set network output size
        test_tensor = torch.randn(1, observation_space.shape[0], channels)
        x = self.forward(test_tensor)
        self.output_size = x.shape[1] 

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.cnn(x)


class CustomFlatten(nn.Module):
    def __init__(self, observation_space) -> None:
        super().__init__()
        channels = observation_space.shape[-1]

        self.flatten = nn.Flatten()

        # Set network output size
        test_tensor = torch.randn(1, observation_space.shape[0], channels)
        x = self.forward(test_tensor)
        self.output_size = x.shape[1] 

    def forward(self, x):
        return self.flatten(x)


def mlp_128_64(in_dim) -> nn.Sequential:
    return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )