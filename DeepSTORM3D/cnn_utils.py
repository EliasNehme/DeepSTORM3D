# Import modules and libraries
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from DeepSTORM3D.physics_utils import PhysicalLayer


# Define the basic Conv-LeakyReLU-BN
class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out


# Localization architecture
class LocalizationCNN(nn.Module):
    def __init__(self, setup_params):
        super(LocalizationCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        if setup_params['dilation_flag']:
            self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (8, 8), (8, 8), 0.2)
            self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (16, 16), (16, 16), 0.2)
        else:
            self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, im):

        # extract multi-scale features
        im = self.norm(im)
        out = self.layer1(im)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=2)
        out = self.deconv1(out)
        out = interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


# Phase mask learning architecture
class OpticsDesignCNN(nn.Module):
    def __init__(self, setup_params):
        super(OpticsDesignCNN, self).__init__()
        self.physicalLayer = PhysicalLayer(setup_params)
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(64, setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, mask, phase_emitter, Nphotons):

        # generate input image given current mask
        im = self.physicalLayer(mask, phase_emitter, Nphotons)
        im = self.norm(im)

        # extract depth features
        out = self.layer1(im)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer7(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer8(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer9(features) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)

        return out


