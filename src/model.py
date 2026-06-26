import torch
import torch.nn as nn
import torchvision.models as models

def build_model():
    print("Building model...")
    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 2)  # Binary classification: crack / no crack
    return net
