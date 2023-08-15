import numpy as np
import torch

from torchvision import models

from const import *

def build_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=len(ALL_CHAR_SET)*MAX_CAPTCHA, bias=True)
    model.cuda()
    return model

class MyResNet(models.ResNet):
    def __init__(self, out_features, **kwargs):
        super().__init__(models.resnet.BasicBlock, [2,2,2,2], **kwargs)
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #out_features=len(ALL_CHAR_SET)*MAX_CAPTCHA
        self.fc = torch.nn.Linear(512, out_features, bias=True)
    
    def forward(self, x, middle_required=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if not middle_required:
            x = self.fc(x)
            return x
        else:
            middle = x
            return middle, self.fc(x)