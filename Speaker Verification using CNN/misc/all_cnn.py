import torch.nn as nn
from torch.nn import Sequential
import torch


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def forward(self,x):
        return x.view(x.size()[0],-1)

def all_cnn_module(n_1 = 3048):
    """
    Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
    https://arxiv.org/pdf/1412.6806.pdf
    Use a AvgPool2d to pool and then your Flatten layer as your final layers.
    You should have a total of exactly 23 layers of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """
    return ResNet(ResnetBuildingBlock, [1,1,1,1], num_classes = n_1)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class Flatten2(nn.Module):
    def forward(self, x):
        return x.view(1,-1)

class ResnetBuildingBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(ResnetBuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, padding = 1, kernel_size = 3, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride


    def forward(self, x):
        residual = x
        # first convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        # second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        # add residual
        out += residual
        out = self.elu(out)
        return out

def conv5x5(in_channels, out_channels, stride = 2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=0,bias=False)


class ResNet(nn.Module):
    def __init__(self,block,layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = conv5x5(1,32)
        self.elu1 = nn.ELU(inplace=True)
        self.res1 = self.make_layer(block, 32, layers[0])
        self.conv2 = conv5x5(32,64)
        self.elu2 = nn.ELU(inplace=True)
        self.res2 = self.make_layer(block, 64, layers[1])
        self.conv3 = conv5x5(64,128)
        self.elu3= nn.ELU(inplace=True)
        self.res3 = self.make_layer(block, 128, layers[2])
        self.conv4 = conv5x5(128,256)
        self.elu4 = nn.ELU(inplace=True)
        self.res4 = self.make_layer(block, 256, layers[3])
        self.avgpool = nn.AvgPool2d(kernel_size=(309,1))
        self.fc = nn.Linear(256, num_classes)
        self.flat = Flatten()
        self.alpha = 16
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def make_layer(self, block, out_channels, blocks):
        layers=[]
        for i in range(0,blocks):
            layers.append(block(out_channels,out_channels))
        return nn.Sequential(*layers)

    def forward(self,x,classify):
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.res1(out)
        out = self.conv2(out)
        out = self.elu2(out)
        out = self.res2(out)
        out = self.conv3(out)
        out = self.elu3(out)
        out = self.res3(out)
        out = self.conv4(out)
        out = self.elu4(out)
        out = self.res4(out)
        out = self.avgpool(out)
        out = self.flat(out)
        out = torch.div(out,torch.norm(out, p=2, dim = 0)) * self.alpha
        if classify:
            out = self.fc(out)
        return out
