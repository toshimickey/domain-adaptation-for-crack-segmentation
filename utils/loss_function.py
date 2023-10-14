import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # model output = (X,1,224,224)
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, 1e-7, 1-1e-7)
        inputs = inputs.view(-1)
        # annotation mask = (X,1,224,224)
        targets = targets.view(-1)

        bce_weight = 0.5
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final

# class BayesMSELoss(nn.Module):
#     def __init__(self):
#         super(BayesMSELoss, self).__init__()

#     def forward(self, inputs, targets, vars):
#         inputs = torch.sigmoid(inputs).view(-1)
#         #inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         vars = vars + 1
#         # バッチ内の各データごとに値の平均を求め(16)のベクトルを作る。これを(16,1,1,1)にresize
#         mean_vars = torch.mean(vars.view(vars.size(0), -1), dim=1).view(vars.size(0), 1, 1, 1)
#         mean_vars = torch.ones(vars.size(0), 1, 256, 256).to(device) * mean_vars
#         weights = mean_vars / vars
#         weights = weights.view(-1)
#         mse_loss = torch.mean(weights * (inputs - targets) ** 2)
#         return mse_loss

# class BayesMSELoss2(nn.Module):
#     def __init__(self, threshold):
#         super(BayesMSELoss2, self).__init__()
#         self.threshold = threshold

#     def forward(self, inputs, targets, vars):
#         inputs = torch.sigmoid(inputs)
#         below_threshold = vars < self.threshold
#         masked_inputs = inputs[below_threshold]
#         masked_targets = targets[below_threshold]
#         loss = F.mse_loss(masked_inputs, masked_targets)
#         return loss

# class BayesBCELoss(nn.Module):
#     def __init__(self):
#         super(BayesBCELoss, self).__init__()

#     def forward(self, inputs, targets, vars):
#         inputs = torch.sigmoid(inputs)
#         inputs = torch.clamp(inputs, 1e-7, 1-1e-7)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         vars = vars + 1
#         # バッチ内の各データごとに値の平均を求め(16)のベクトルを作る。これを(16,1,1,1)にresize
#         mean_vars = torch.mean(vars.view(vars.size(0), -1), dim=1).view(vars.size(0), 1, 1, 1)
#         mean_vars = torch.ones(vars.size(0), 1, 256, 256).to(device) * mean_vars
#         # ゼロ除算への対応
#         weights = mean_vars / vars
#         weights = weights.view(-1)
#         bce_loss = - torch.mean(weights * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs)))
#         return bce_loss

# class BayesBCELoss(nn.Module):
#     def __init__(self):
#         super(BayesBCELoss, self).__init__()

#     def forward(self, inputs, targets, vars_t, vars_s, alpha=0.001):
#         inputs = torch.sigmoid(inputs)
#         inputs = torch.clamp(inputs, 1e-7, 1-1e-7)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         vars_t = vars_t.view(-1)
#         vars_s = vars_s.view(-1)

#         bce_loss = torch.mean(-torch.exp(-vars_t) * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))) + (alpha * F.relu(torch.mean(vars_s) - 1.0))
#         return bce_loss

class BayesBCELoss(nn.Module):
    def __init__(self, alpha=1000, beta=10):
        super(BayesBCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, vars):
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, 1e-7, 1-1e-7)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        vars = vars.view(-1)

        bce_loss = torch.mean(-torch.exp(-vars*self.alpha) * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs)))
        return bce_loss*self.beta