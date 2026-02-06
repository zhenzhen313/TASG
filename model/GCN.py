
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias=bias
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
    
    def forward(self, text, adj,attention_mask):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=-1, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            output=output + self.bias
        output=torch.einsum('ijk,ij->ijk',output,attention_mask)
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5,alpha=0.7):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)  # 添加 BatchNorm
        self.gc2 = GraphConvolution(nhid, nclass)
        self.bn2 = nn.BatchNorm1d(nclass)  # 添加 BatchNorm
        self.dropout = dropout
        # self.leakyrelu = nn.LeakyReLU(alpha)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj,attention_mask):
        x = self.gc1(x, adj, attention_mask)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm 需要特定维度
        x = self.leakyrelu(x)
        x = torch.einsum('ijk,ij->ijk', x, attention_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, attention_mask)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.leakyrelu(x)
        x = torch.einsum('ijk,ij->ijk', x, attention_mask)

        return x