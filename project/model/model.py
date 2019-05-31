import torch.nn as nn
from torchvision.models import resnet18
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        std_resnet = resnet18(pretrained=True)
        # input 2 * 180 * 240
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 5, stride=1, padding=(8, 2), bias=False),
            # now 64 * 192 * 240
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # now 64 * 96 * 120,
        self.resnet = nn.Sequential(
            std_resnet.layer1,
            std_resnet.layer2,
            std_resnet.layer3,
            std_resnet.layer4
        )
        # now 512 * 12 * 15, which is put into global average
        self.average = nn.AvgPool2d((12, 15), stride=1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),#raises an error with batch size 1 (last batch)
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        x = self.average(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'
