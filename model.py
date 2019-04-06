# File: model.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 2:18 PM

from torch import nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act = nn.Tanh()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.act(self.conv1(x))
        x = self.avg_pool(x)
        x = self.act(self.conv2(x))
        x = self.avg_pool(x)
        x = self.conv3(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x)
        return x
