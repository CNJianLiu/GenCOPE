import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(
            features * (len(sizes) + 1), out_features, kernel_size=1
        )
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.upsample(input=stage(feats), size=(h, w), mode='bilinear')
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class Modified_PSPNet(nn.Module):
    def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet18', pretrained=True):
        super(Modified_PSPNet, self).__init__()
        self.feats = getattr(resnet, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        return self.final(p)


# main modules

modified_psp_models = {
    'resnet18': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, backend='resnet18'),
    'resnet34': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, backend='resnet34'),
    'resnet50': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet50'),
    'resnet101': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet101'),
    'resnet152': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet152')
}


class ModifiedResnet(nn.Module):
    def __init__(self):
        super(ModifiedResnet, self).__init__()
        self.model = modified_psp_models['resnet18'.lower()]()

    def forward(self, x):
        x = self.model(x)
        return x


