import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_FC(nn.Module):
    def __init__(self, serie_size, num_classes):
        super(conv_FC, self).__init__()
        self._num_classes = num_classes
        # self.num_segments = num_segments
        # self.segment_size = segment_size
        self.signal_lenght = serie_size
        self.conv1 = nn.Conv1d(1, 2, 3, 1,padding=1)
        self.conv2 = nn.Conv1d(2, 4, 3, 1,padding=1)
        # self.conv4 = nn.Conv1d(70, 70, 3, 1,padding=1)
        # self.conv5 = nn.Conv1d(70, 100, 1, 1,padding=0)
        self.fc1 = nn.Linear(int(self.signal_lenght*4/16),20)
        self.fc2 = nn.Linear(20, self._num_classes)
        self.drop_layer = nn.Dropout()


    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.max_pool1d(x, 4)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # x = self.drop_layer(x)
        x = F.max_pool1d(x, 4)
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        # # print(x.shape)
        # x = F.relu(self.conv4(x))
        # print(x.shape)
        # x = F.relu(self.conv5(x))
        # print(x.shape)
        # print(x.shape)
        # x = F.max_pool1d(x, 2)
        # print(x.shape)
        x = x.view(-1, int(self.signal_lenght*4/16 ))
        # x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return Fc.log_softmax(x, dim=1)
        return x    

class FC(nn.Module):
    def __init__(self, signal_lenght, num_classes):
        super(FC, self).__init__()
        self._num_classes = num_classes
        self.signal_lenght = signal_lenght


        self.fc1 = nn.Linear(int(signal_lenght ) , int(signal_lenght*4/16))
        self.drop_layer = nn.Dropout()
        self.fc2 = nn.Linear(int(signal_lenght*4/16), self._num_classes)

    def forward(self, x):
        # print(x.shape)
        # print(x.shape)
        x = x.view(-1, int(self.signal_lenght*4/16))
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = self.fc2(x)
        return x