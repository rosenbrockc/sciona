"""Pinned custom HuBMAP U-Net, CBAM, hypercolumns and auxiliary heads.

Copyright (c) 2021 Tom; MIT, see docs/licenses/HuBMAP-MIT.txt.
Source commit615444e86f4fb5ab916c0b66d311a307f4e4dd27.
Source bodies retained with explicit encoder construction/state input, no download,
and rejection of the source optional threshold branch's integer placeholders.
The published training configurations use clf_threshold=None.
"""
import torch
from torch import nn
import torch.nn.functional as F
from sciona.hubmap_encoder import build_encoder

def conv3x3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

def conv1x1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

class ChannelAttentionModule(nn.Module):

    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(conv1x1(in_channel, in_channel // reduction).apply(init_weight), nn.ReLU(True), conv1x1(in_channel // reduction, in_channel).apply(init_weight))

    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x = torch.sigmoid(x1 + x2)
        return x

class SpatialAttentionModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2, 1).apply(init_weight)

    def forward(self, inputs):
        x1, _ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3x3(x)
        x = torch.sigmoid(x)
        return x

class CBAM(nn.Module):

    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x

class CenterBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x

class DecodeBlock(nn.Module):

    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1 = conv1x1(in_channel, out_channel).apply(init_weight)

    def forward(self, inputs):
        x = F.relu(self.bn1(inputs))
        x = self.upsample(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(F.relu(self.bn2(x)))
        x = self.cbam(x)
        x += self.conv1x1(self.upsample(inputs))
        return x

class UNET_SERESNEXT101(nn.Module):

    def __init__(self, resolution, deepsupervision, clfhead, clf_threshold, load_weights=False, encoder_state=None):
        if load_weights:
            raise ValueError('Supply explicit encoder_state instead of downloading model weights')
        if clf_threshold is not None:
            raise ValueError('Thresholded source eval placeholders are not an executable output contract')
        if len(resolution) != 2 or any((isinstance(n, bool) or not isinstance(n, int) or n < 32 or n % 32 for n in resolution)):
            raise ValueError('Positive spatial multiples of32 required')
        if not isinstance(deepsupervision, bool) or not isinstance(clfhead, bool):
            raise ValueError('Boolean head flags required')
        super().__init__()
        h, w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        self.clf_threshold = clf_threshold
        model_name = 'se_resnext101_32x4d'
        seresnext101 = build_encoder()
        if encoder_state is not None:
            seresnext101.load_state_dict(encoder_state, strict=True)
        self.encoder0 = nn.Sequential(seresnext101.layer0.conv1, seresnext101.layer0.bn1, seresnext101.layer0.relu1)
        self.encoder1 = nn.Sequential(seresnext101.layer0.pool, seresnext101.layer1)
        self.encoder2 = seresnext101.layer2
        self.encoder3 = seresnext101.layer3
        self.encoder4 = seresnext101.layer4
        self.center = CenterBlock(2048, 512)
        self.decoder4 = DecodeBlock(512 + 2048, 64, upsample=True)
        self.decoder3 = DecodeBlock(64 + 1024, 64, upsample=True)
        self.decoder2 = DecodeBlock(64 + 512, 64, upsample=True)
        self.decoder1 = DecodeBlock(64 + 256, 64, upsample=True)
        self.decoder0 = DecodeBlock(64, 64, upsample=True)
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deep4 = conv1x1(64, 1).apply(init_weight)
        self.deep3 = conv1x1(64, 1).apply(init_weight)
        self.deep2 = conv1x1(64, 1).apply(init_weight)
        self.deep1 = conv1x1(64, 1).apply(init_weight)
        self.final_conv = nn.Sequential(conv3x3(320, 64).apply(init_weight), nn.ELU(True), conv1x1(64, 1).apply(init_weight))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(nn.BatchNorm1d(2048).apply(init_weight), nn.Linear(2048, 512).apply(init_weight), nn.ELU(True), nn.BatchNorm1d(512).apply(init_weight), nn.Linear(512, 1).apply(init_weight))

    def forward(self, inputs):
        x0 = self.encoder0(inputs)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1))
        if (not self.training) & (self.clf_threshold is not None):
            if (torch.sigmoid(logits_clf) > self.clf_threshold).sum().item() == 0:
                bs, _, h, w = inputs.shape
                logits = torch.zeros((bs, 1, h, w))
                if self.clfhead:
                    if self.deepsupervision:
                        return (logits, _, _)
                    else:
                        return (logits, _)
                elif self.deepsupervision:
                    return (logits, _)
                else:
                    return logits
        y5 = self.center(x4)
        y4 = self.decoder4(torch.cat([x4, y5], dim=1))
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))
        y0 = self.decoder0(y1)
        y4 = self.upsample4(y4)
        y3 = self.upsample3(y3)
        y2 = self.upsample2(y2)
        y1 = self.upsample1(y1)
        hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)
        logits = self.final_conv(hypercol)
        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return (logits, logits_deeps, logits_clf)
            else:
                return (logits, logits_clf)
        elif self.deepsupervision:
            s4 = self.deep4(y4)
            s3 = self.deep3(y3)
            s2 = self.deep2(y2)
            s1 = self.deep1(y1)
            logits_deeps = [s4, s3, s2, s1]
            return (logits, logits_deeps)
        else:
            return logits
