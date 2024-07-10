import torch.nn as nn
#import torchvision.transforms as transforms
import math
from .quantize_util import  BinarizeLinear, BinarizeConv2d, Conv2d 

__all__ = ['resnet18_binary']

def Binaryconv3x3(in_planes, out_planes, stride=1, ao_bit=1, w_bit=1, adc_bit=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                          padding=1, bias=True, ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, do_bntan=True, ao_bit=1, w_bit=1, adc_bit=1):
        super(BasicBlock, self).__init__()

        self.conv1      = Binaryconv3x3(inplanes, planes, stride, ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit)
        self.bn1        = nn.BatchNorm2d(planes)
        self.tanh1      = nn.Hardtanh(inplace=True)

        self.conv2      = Binaryconv3x3(planes, planes, ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit)
        self.tanh2      = nn.Hardtanh(inplace=True)
        self.bn2        = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan   = do_bntan
        self.stride     = stride

    def forward(self, x):
        residual = x.clone()

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)

        if self.downsample is not None:
            if residual.data.max()>1:
                import pdb; pdb.set_trace()
            residual = self.downsample(residual)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1      = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = BinarizeConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes * 4)
        self.tanh       = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x
        import pdb; pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1, do_bntan=True, ao_bit=1, w_bit=1, adc_bit=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit))
        self.inplanes = planes * block.expansion
        for _ in range(blocks-2):
            layers.append(block(self.inplanes, planes, ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit))
        layers.append(block(self.inplanes, planes, do_bntan=do_bntan, ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)

        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000, block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1    = BinarizeConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.tanh     = nn.Hardtanh(inplace=True)
        self.maxpool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1   = self._make_layer(block,  64, layers[0])
        self.layer2   = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3   = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4   = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool  = nn.AvgPool2d(7)
        self.fc       = BinarizeLinear(True, 512 * block.expansion, num_classes)

        init_model(self)
        self.regime = {
            0:  {'optimizer': 'SGD', 'lr': 1e-1, 'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3, 'weight_decay': 0},
            90: {'lr': 1e-4}
        }


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18, ao_bit=1, w_bit=1, adc_bit=1):
        super(ResNet_cifar10, self).__init__()
        self.inflate  = 5
        self.inplanes = 16*self.inflate
        self.ao_bit  = ao_bit
        self.w_bit   = w_bit
        self.adc_bit = adc_bit
        n = int((depth - 2) / 6)

        self.conv1      = Conv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool    = lambda x: x

        self.bn1        = nn.BatchNorm2d(16*self.inflate)
        self.tanh1      = nn.Hardtanh(inplace=True)
        self.tanh2      = nn.Hardtanh(inplace=True)

        self.layer1     = self._make_layer(block,  16*self.inflate, 3,
                                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit)
        self.layer2     = self._make_layer(block,  32*self.inflate, 3, stride=2,
                                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit)
        self.layer3     = self._make_layer(block,  64*self.inflate, 2, stride=2, do_bntan=False,
                                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit)
        self.layer4     = lambda x: x 

        self.avgpool    = nn.AvgPool2d(8)
        self.bn2        = nn.BatchNorm1d(64*self.inflate)
        self.bn3        = nn.BatchNorm1d(10)

        self.logsoftmax = nn.LogSoftmax()
        self.fc         = BinarizeLinear(64*self.inflate, 10,
                                         last_layer=True, ao_bit=self.ao_bit, w_bit=w_bit, adc_bit=self.adc_bit)
        init_model(self)
        self.regime = {
            0:   {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }


def resnet18_binary(**kwargs):
    num_classes, depth, dataset, ao_bit, w_bit, adc_bit = map(kwargs.get, ['num_classes', 'depth', 'dataset', 'ao_bit',  'w_bit', 'adc_bit'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2,  2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[3, 4, 6,  3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 4, 6,  3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 8, 36, 3])
    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth       = depth       or 18
        return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, depth=depth, ao_bit=ao_bit, w_bit=w_bit, adc_bit=adc_bit)
