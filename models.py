import torch
import torch.nn as nn
import torch.nn.functional as F

class mnist_arch(nn.Module):
    def __init__(self, num_classes):
        super(mnist_arch, self).__init__()
        self._num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 30, 5, 1)
        self.conv2 = nn.Conv2d(30, 70, 5, 1)
        self.conv3 = nn.Conv2d(70, 100, 1, 1)
        self.fc1 = nn.Linear(4*4*100, 500)
        self.fc2 = nn.Linear(500, self._num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)