# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamFC(nn.Module):
    def __init__(self):
        super(SiamFC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, groups=2),
            nn.BatchNorm2d(256),
        )
        self.adjust = nn.BatchNorm2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def xcorr(self, z, x):
        batch_size, _, H, W = x.shape
        x = torch.reshape(x, (1, -1, H, W))
        out = F.conv2d(x, z, groups=batch_size)
        xcorr_out = out.transpose(0,1)
        return xcorr_out
    
    def init_template(self, z):
        z_feat = self.features(z)
        return torch.cat([z_feat for _ in range(5)], dim=0)
    
    def forward_corr(self, z_feat, x):
        x_feat = self.features(x)
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)
        return score
                
    def forward(self, z, x):
        z_feat = self.features(z)
        x_feat = self.features(x)
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)
        
        return score
    
if __name__ == '__main__':
    model = SiamFC()
    exemplar = torch.randn(1, 3, 127, 127)
    search = torch.randn(1, 3, 255, 255)
    
    score = model(exemplar, search)
    # 1 x 1 x 17 x 17
    print(score.shape)