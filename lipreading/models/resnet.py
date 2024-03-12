import math
import torch.nn as nn
import pdb

from lipreading.models.swish import Swish
from .layers import PHMConv2d

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

def downsample_basic_block_v2( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )

def PHMconv3x3(in_planes, out_planes, stride=1, n=4):
    return PHMConv2d(n, in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def PHMdownsample_basic_block( inplanes, outplanes, stride, n=4):
    return  nn.Sequential(
                PHMConv2d(n, inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

def PHMdownsample_basic_block_v2( inplanes, outplanes, stride , n=4):
    return  nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                PHMConv2d(n, inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type = 'prelu', phm=False, n=4):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu','prelu', 'swish']

        if not phm:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = PHMconv3x3(inplanes, planes, stride, n=n)
        self.bn1 = nn.BatchNorm2d(planes)

        # type of ReLU is an input option
        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == 'swish':
            self.relu1 = Swish()
            self.relu2 = Swish()
        else:
            raise Exception('relu type not implemented')
        # --------

        if not phm:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = PHMconv3x3(planes, planes, n=n)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, relu_type = 'relu', gamma_zero = False, avg_pool_downsample = False, phm=False, n=4):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        if not phm:
            self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block
        else:
            self.downsample_block = PHMdownsample_basic_block_v2 if avg_pool_downsample else PHMdownsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0], phm=phm, n=n)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, phm=phm, n=n)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, phm=phm, n=n)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, phm=phm, n=n)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #nn.init.ones_(m.weight)
                #nn.init.zeros_(m.bias)

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock ):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, phm=False, n=4):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not phm:
                downsample = self.downsample_block( inplanes = self.inplanes,
                                                    outplanes = planes * block.expansion,
                                                    stride = stride )
            else:
                downsample = self.downsample_block( inplanes = self.inplanes,
                                                    outplanes = planes * block.expansion,
                                                    stride = stride, n=n)

        layers = []
        if not phm:
            layers.append(block(self.inplanes, planes, stride, downsample, relu_type = self.relu_type))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, relu_type = self.relu_type))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, relu_type = self.relu_type, phm=phm, n=n))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, relu_type = self.relu_type, phm=phm, n=n))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
