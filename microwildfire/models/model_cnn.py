import torch, torch.nn as nn
from torchvision import models

def build_resnet18(in_ch=3):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if in_ch != 3:
        w = m.conv1.weight.clone()
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            m.conv1.weight[:, :3] = w
            if in_ch > 3:
                mean_w = w.mean(dim=1, keepdim=True)
                m.conv1.weight[:, 3:] = mean_w.repeat(1, in_ch-3, 1, 1)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )
    return m

