import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        z = F.normalize(self.fc(x), p=2, dim=1)
        return z

class SiameseBin(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.enc = SmallCNN(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim*2,128), nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, x1,x2):
        z1,z2 = self.enc(x1),self.enc(x2)
        feat = torch.cat([torch.abs(z1-z2), z1*z2],1)
        logit = self.head(feat).squeeze(1)
        return logit,(z1,z2)