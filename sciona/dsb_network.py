"""Source architecture port for candidate competition execution.

Copyright (c) 2023 lfz. MIT notice: docs/licenses/DSB2017-MIT.txt.
Pinned source commit: 0ac3eb9f383bf0127c587e0502c59ac84d9ba6a2.
Only PostRes, detector Net, classifier Net and CaseNet are included. No source
configuration paths, datasets, checkpoints or file readers are embedded.
Port changes: center-feature slice halves use Python 3 integer division;
detector class renamed DetectorNet to coexist with classifier feature Net.
Random initialization is not a trained or validated prediction model.
"""
import torch
from torch import nn

config = {'anchors': [10, 30, 60]}

class PostRes(nn.Module):

    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)
        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride), nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.preBlock = nn.Sequential(nn.Conv3d(1, 24, kernel_size=3, padding=1), nn.BatchNorm3d(24), nn.ReLU(inplace=True), nn.Conv3d(24, 24, kernel_size=3, padding=1), nn.BatchNorm3d(24), nn.ReLU(inplace=True))
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))
        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.path1 = nn.Sequential(nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.2, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1), nn.ReLU(), nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))

    def forward(self, x, coord):
        out = self.preBlock(x)
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)
        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))
        rev2 = self.path2(comb3)
        feat = self.back2(torch.cat((rev2, out2, coord), 1))
        comb2 = self.drop(feat)
        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        return (feat, out)

class CaseNet(nn.Module):

    def __init__(self, topk, nodulenet=None):
        super(CaseNet, self).__init__()
        self.NoduleNet = Net() if nodulenet is None else nodulenet
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
        self.Relu = nn.ReLU()

    def forward(self, xlist, coordlist):
        xsize = xlist.size()
        corrdsize = coordlist.size()
        xlist = xlist.view(-1, xsize[2], xsize[3], xsize[4], xsize[5])
        coordlist = coordlist.view(-1, corrdsize[2], corrdsize[3], corrdsize[4], corrdsize[5])
        noduleFeat, nodulePred = self.NoduleNet(xlist, coordlist)
        nodulePred = nodulePred.contiguous().view(corrdsize[0], corrdsize[1], -1)
        featshape = noduleFeat.size()
        centerFeat = self.pool(noduleFeat[:, :, featshape[2] // 2 - 1:featshape[2] // 2 + 1, featshape[3] // 2 - 1:featshape[3] // 2 + 1, featshape[4] // 2 - 1:featshape[4] // 2 + 1])
        centerFeat = centerFeat[:, :, 0, 0, 0]
        out = self.dropout(centerFeat)
        out = self.Relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.view(xsize[0], xsize[1])
        base_prob = torch.sigmoid(self.baseline)
        casePred = 1 - torch.prod(1 - out, dim=1) * (1 - base_prob.expand(out.size()[0]))
        return (nodulePred, casePred, out)

class DetectorNet(nn.Module):

    def __init__(self):
        super(DetectorNet, self).__init__()
        self.preBlock = nn.Sequential(nn.Conv3d(1, 24, kernel_size=3, padding=1), nn.BatchNorm3d(24), nn.ReLU(inplace=True), nn.Conv3d(24, 24, kernel_size=3, padding=1), nn.BatchNorm3d(24), nn.ReLU(inplace=True))
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))
        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.path1 = nn.Sequential(nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.2, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1), nn.ReLU(), nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))

    def forward(self, x, coord):
        out = self.preBlock(x)
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)
        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))
        rev2 = self.path2(comb3)
        feat = self.back2(torch.cat((rev2, out2, coord), 1))
        comb2 = self.drop(feat)
        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        return out
