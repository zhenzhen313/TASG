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
        # pred: [batch_size, num_classes], 未经过 softmax
        # labels: [batch_size], 真实标签 id
        # Step 1: 先做 log_softmax
        pred = F.log_softmax(pred, dim=1)  # [B, K]

        # Step 2: 生成 one-hot 标签
        label_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(self.device)

        # Step 3: 按照 NCE 公式计算
        #   分子 = -1 * sum(label_one_hot * pred)
        #   分母 = -1 * sum(pred) （对同一个样本在 num_classes 维度上求和）
        #   具体实现中： nce_i = (分子_i) / (分母_i)
        nce = -1.0 * torch.sum(label_one_hot * pred, dim=1) / (
            -1.0 * pred.sum(dim=1)
        )
        
        return self.scale * nce.mean()


class ReverseCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0, device='cuda'):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device  # 假设在你的代码里提前定义好了 device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        # pred: [B, num_classes], 未过 softmax 的网络输出
        # labels: [B], 每个位置是真实类别 id

        # 1) 计算预测分布 p(k|x)
        pred = F.softmax(pred, dim=1)
        # 避免出现 log(0) 的数值问题
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        # 2) 生成 one-hot 标签并做截断，避免 log(0)
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