import torch.nn as nn
import torch
import timm

class FaceNetRes50(nn.Module):
    def __init__(self):
        super(FaceNetRes50, self).__init__()

        model = timm.create_model('resnet50', pretrained='imagenet')
        model.fc = nn.Linear(2048, 7)
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out

class FaceNetEb2(nn.Module):
    def __init__(self):
        super(FaceNetEb2, self).__init__()

        model = timm.create_model('efficientnet_b2', pretrained='imagenet')
        model.classifier = nn.Linear(1408, 7)
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out

class FaceNetIBN50(nn.Module):
    def __init__(self):
        super(FaceNetIBN50, self).__init__()

        self.model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        self.model.fc = nn.Linear(2048, 7)

    def forward(self, img):
        out = self.model(img)
        return out