import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, input_size=64):
        super(Generator, self).__init__()

        self.init_size = input_size // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.main_module = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.main_module(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_size=64, sn=False):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, sn=False, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if sn:
                block[0] = spectral_norm(block[0])
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32, bn=False),
            *discriminator_block(32, 64, bn=False),
            *discriminator_block(64, 128, bn=False),
        )

        # The height and width of downsampled image
        ds_size = input_size // 2 ** 4
        fc = nn.Linear(128 * ds_size ** 2, 1)
        if sn:
            fc = spectral_norm(fc)
        self.adv_layer = nn.Sequential(fc, nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

