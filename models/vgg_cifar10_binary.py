import torch
import torch.nn as nn
#import torchvision.transforms as transforms
from torch.autograd import Function
from .quantize_util import BinarizeLinear,BinarizeConv2d

class VGG_Cifar10(nn.Module):
    def __init__(self, num_classes=1000, ao_bit=1, w_bit=1, adc_bit=1):
        super(VGG_Cifar10, self).__init__()
        self.infl_ratio=3
        self.ao_bit  = ao_bit
        self.w_bit   = w_bit
        self.adc_bit = adc_bit
        self.features = nn.Sequential(
            BinarizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=True,
                           ao_bit=self.ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),
            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True,
                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True,
                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),
            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True,
                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True,
                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),
            BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True,
                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512 * 4 * 4, 1024, bias=True, last_layer=False, 
                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),

            BinarizeLinear(1024, 1024, bias=True, last_layer=False, 
                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),

            BinarizeLinear(1024, num_classes, bias=True, last_layer=True, 
                           ao_bit=ao_bit, w_bit=w_bit, adc_bit=self.adc_bit),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0:   {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40:  {'lr': 1e-3},
            80:  {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def vgg_cifar10_binary(**kwargs):
    num_classes, ao_bit, w_bit, adc_bit = map(kwargs.get, ['num_classes', 'ao_bit', 'w_bit', 'adc_bit'])
    num_classes=10
    return VGG_Cifar10(num_classes, ao_bit, w_bit, adc_bit)
