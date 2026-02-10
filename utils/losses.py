import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0, device='cuda'):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)  # [B, K]

        label_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(self.device)

        nce = -1.0 * torch.sum(label_one_hot * pred, dim=1) / (
            -1.0 * pred.sum(dim=1)
        )
        
        return self.scale * nce.mean()


class ReverseCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0, device='cuda'):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device 
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes,device):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes,device=device)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes,device=device)
        

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)
