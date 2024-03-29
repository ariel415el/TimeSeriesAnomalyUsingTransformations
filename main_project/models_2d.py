import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self,num_classes, t_dim=64, img_x=128, img_y=128, drop_p=0.2, fc_hidden1=256, fc_hidden2=128):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = x_3d.unsqueeze(1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.drop(x)
        # print(x.shape)
        # Conv 2
        x = self.conv2(x)
        # print(x.shape)
        x = self.bn2(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.drop(x)
        # print(x.shape)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = x.squeeze(1)
        # x = nn.functional.softmax(x, dim=1)
        return  x

class multi_head_FC(nn.Module):
    def __init__(self,num_classes, t_dim=64, img_x=128, img_y=128):
        super(multi_head_FC, self).__init__()
        intermediate_repr = 500
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        self.head = nn.Sequential(
            nn.Conv2d(1, 2, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, 3, 2, padding=1)
        )
        self.features_size = int(img_x/8*img_y/8)
        self.fc1 = nn.Linear(t_dim*self.features_size, intermediate_repr)
        self.fc2 = nn.Linear(intermediate_repr, num_classes)

    def forward(self, x_3d):
        features = []
        # print(x_3d.shape)
        for i in range(x_3d.shape[1]):
            features.append(self.head(x_3d[:,i].unsqueeze(1)))
        # print(len(features))
        # print(features[0].shape)
        features = torch.stack(features, dim=1)
        # print(features.shape)
        x = features.view(-1, self.features_size*self.t_dim)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x
