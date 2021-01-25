import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sparse_anchors(batch_size, feature_size, downsample, joint_num):
    mesh_x = (torch.arange(feature_size) * downsample + downsample//2).unsqueeze(0).expand(feature_size, feature_size).float()
    mesh_y = (torch.arange(feature_size) * downsample + downsample//2).unsqueeze(1).expand(feature_size, feature_size).float()
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (B, 2, F, F)
    default_z = torch.zeros((batch_size, 1, feature_size, feature_size), dtype=torch.float)  # (B, 1, F, F)
    anchors = torch.cat((coords, default_z), dim=1).cuda()  # (B, 3, F, F)
    return anchors


class JointL2Loss(nn.Module):
    def __init__(self):
        super(JointL2Loss, self).__init__()

    def forward(self, joint_pred, joint_gt):
        batch_size, joint_num, _ = joint_gt.shape
        joint_pred = joint_pred.view(batch_size * joint_num, -1)
        joint_gt = joint_gt.view(batch_size * joint_num, -1)
        offset = torch.sum(torch.pow(joint_gt - joint_pred, 2), dim=1)
        return torch.sqrt(offset).mean()


class A2JProcess(nn.Module):
    def __init__(self, img_size, joint_num, downsample):
        super(A2JProcess, self).__init__()
        self.img_size = img_size
        self.downsample = downsample
        self.feature_size = int(img_size // downsample)
        self.joint_num = joint_num

    def forward(self, heads):
        classifications, regressions = heads
        batch_size = regressions.shape[0]
        joint_num = self.joint_num
        coords = get_sparse_anchors(batch_size, self.feature_size, self.downsample, self.joint_num)
        anchors = coords.repeat(1, joint_num, 1, 1)
        joint_pred = (anchors + regressions).view(batch_size, joint_num, 3, -1)
        reg = F.softmax(classifications.view(batch_size, joint_num, -1), dim=-1)
        joint_pred = torch.sum(joint_pred * reg.unsqueeze(2), dim=-1)

        return joint_pred


class A2JLoss(nn.Module):
    def __init__(self, img_size, joint_num, downsample):
        super(A2JLoss, self).__init__()
        self.l1_loss = torch.nn.SmoothL1Loss()
        # self.loss = torch.nn.MSELoss()
        self.jl2_loss = JointL2Loss()
        self.img_size = img_size
        self.downsample = downsample
        self.feature_size = int(img_size // downsample)
        self.joint_num = joint_num

    def forward(self, heads, labels):
        classifications, regressions = heads
        batch_size, joint_num, _ = labels.shape
        coords = get_sparse_anchors(batch_size, self.feature_size, self.downsample, self.joint_num)
        anchors = coords.repeat(1, joint_num, 1, 1)
        joint_pred = (anchors + regressions).view(batch_size, joint_num, 3, -1)  # (B, joint_num, 3, F*F)
        reg = F.softmax(classifications.view(batch_size, joint_num, -1), dim=-1)  # (B, joint_num, F*F)
        joint_pred = torch.sum(joint_pred * reg.unsqueeze(2), dim=-1)
        joint_loss = self.jl2_loss(labels, joint_pred)

        dense_offset = labels.view(batch_size, -1, 1, 1).repeat(1, 1, self.feature_size, self.feature_size) - anchors
        dense_loss = self.l1_loss(dense_offset, regressions)

        return dense_loss, joint_loss
