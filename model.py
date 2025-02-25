import torch
import torch.nn as nn
import torch.nn.functional as F

class GuitarTabNet(nn.Module):
    def __init__(self, input_shape, num_frets=19):
        super(GuitarTabNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size after convolutions
        with torch.no_grad():
            self.flatten_size = self._get_flatten_size(input_shape)

        # Fully Connected Branches for each guitar string
        self.branches = nn.ModuleList([self._create_branch(self.flatten_size, num_frets) for _ in range(6)])

    def _get_flatten_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        x = self.pool(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(dummy_input)))))))
        return x.view(1, -1).size(1)

    def _create_branch(self, input_dim, num_frets):
        return nn.Sequential(
            nn.Linear(input_dim, 152),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(152, 76),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(76, num_frets)
        )

    def forward(self, x):
        # Shared CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten for fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Forward pass for each guitar string
        outputs = [F.log_softmax(branch(x), dim=1) for branch in self.branches]

        return outputs


def get_model(input_shape, learning_rate=0.01, momentum=0.8, epochs=30):
    model = GuitarTabNet(input_shape)

    # Optimizer with SGD and learning rate decay
    decay = learning_rate / epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)

    return model, optimizer
