
import torch
import torch.nn as nn
import torchvision.models as models

class YOLOResNet(nn.Module):
    def __init__(self, num_classes, num_anchors=3):
        super(YOLOResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.detector = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, num_anchors * (5 + num_classes), 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.detector(x)
        return x
