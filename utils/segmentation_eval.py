import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
import numpy as np

class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return dice_score

class Accuracy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Accuracy, self).__init__()

    def forward(self, inputs, targets):

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        inputs = (inputs>= 0.5).float()
        #targets = targets[:,:,:,0]
        targets = targets.view(-1)

        accuracy = torch.mean((inputs == targets).float())
        return accuracy

#適合率
class Precision(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Precision, self).__init__()

    def forward(self, inputs, targets):

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        inputs = (inputs>= 0.5).float()
        #targets = targets[:,:,:,0]
        targets = targets.view(-1)
        tp = torch.sum((inputs == 1) & (targets == 1)).float()
        fp = torch.sum((inputs == 1) & (targets == 0)).float()

        # precisionを計算する
        precision = tp / (tp + fp)
        return precision

#再現率
class Recall(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Recall, self).__init__()

    def forward(self, inputs, targets):

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        inputs = (inputs>= 0.5).float()
        #targets = targets[:,:,:,0]
        targets = targets.view(-1)
        tp = torch.sum((inputs == 1) & (targets == 1)).float()
        fn = torch.sum((inputs == 0) & (targets == 1)).float()

        recall = tp / (tp + fn)
        return recall

#特異度
class Specificity(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Specificity, self).__init__()

    def forward(self, inputs, targets):

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        inputs = (inputs>= 0.5).float()
        #targets = targets[:,:,:,0]
        targets = targets.view(-1)

        tn = torch.sum((inputs == 0) & (targets == 0)).float()
        fp = torch.sum((inputs == 1) & (targets == 0)).float()

        specificity = tn / (tn + fp)
        return specificity