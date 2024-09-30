import torch
import torch.nn as nn
import torch.nn.functional as F

# Soft Target Cross Entropy Loss for Mixup
class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

# Label Smoothing Loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# 손실 함수 선택
class Loss(nn.Module):
    def __init__(self, num_classes, label_smoothing=0.0, mixup=False):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.mixup = mixup
        
        if self.mixup:
            self.criterion = SoftTargetCrossEntropy()
        elif self.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(classes=num_classes, smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)