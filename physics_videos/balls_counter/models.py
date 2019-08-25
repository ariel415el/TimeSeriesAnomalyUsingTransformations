import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )   


class conv_FC(nn.Module):
    def __init__(self, num_classes):
        super(conv_FC, self).__init__()
        self._num_classes = num_classes

        self.conv1 = nn.Conv2d(48, 64, 3, 1,groups=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1,groups=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1,groups=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, 1,groups=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, 1,groups=1, padding=1)
        self.conv3 = nn.Conv2d(512, 48, 3, 1,groups=1, padding=1)
        # self.conv4 = nn.Conv1d(70, 70, 3, 1,padding=1)
        # self.conv5 = nn.Conv1d(70, 100, 1, 1,padding=0)
        self.fc1 = nn.Linear(48*8*8, 20)
        self.fc2 = nn.Linear(20, self._num_classes)
        self.drop_layer = nn.Dropout()


    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 4)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 4)
        # print(x.shape)
        x = F.relu(self.conv3(x))

        x = x.view(-1,48*8*8)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x    


class sep_conv_FC(nn.Module):
    def __init__(self, num_classes):
        super(sep_conv_FC, self).__init__()
        self._num_classes = num_classes

        self.features = nn.Sequential(
            conv_bn(  48,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 2),
            conv_dw(512, 256, 1),
            conv_dw(256, 128, 1),
            conv_dw(128, 48, 1)
        )
        self.fc1 = nn.Linear(48*8*8, 32)
        self.fc2 = nn.Linear(32, self._num_classes)
        self.drop_layer = nn.Dropout()


    def forward(self, x):
        # print(x.shape)
        x = self.features(x)

        # print(x.shape)
        x = x.view(-1,48*8*8)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x  

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self,num_classes, t_dim=48, img_x=128, img_y=128, drop_p=0.2, fc_hidden1=32, fc_hidden2=32):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 4, 4
        self.k1, self.k2 = (3, 3, 3), (3, 3, 3)  # 3d kernel size
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
        return x