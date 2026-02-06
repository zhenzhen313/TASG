import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DotProductAttention(nn.Module):
  def __init__(self, dropout, **kwargs):
    super(DotProductAttention,self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)


  def masked_softmax(self, X, valid_lens): 
    if valid_lens is None: 
      return nn.functional.softmax(X, dim=-1)
    else:
      shape = X.shape
      if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
      else:
        valid_lens = valid_lens.reshape(-1)
      masked_value = -1e6
      X = X.reshape(-1, shape[-1])
      maxlen = X.size(1)
      mask = torch.arange((maxlen),dtype=torch.float32,device=X.device)[None,:] < valid_lens[:,None] #
      X[~mask] = masked_value
      X = X.reshape(shape)
      m, _ = torch.max(X, dim=-1)
      m, _ = torch.max(m, dim=-1) 
      return nn.functional.softmax(X, dim=-1), m 

  def forward(self, q, k, v, valid_lens=None, threshold=0.8):

    d = q.shape[-1]
    scores = torch.bmm(q, k.transpose(1,2)) / math.sqrt(d) 
    self.attention_weights, self.max_score = self.masked_softmax(scores, valid_lens)
    return torch.bmm(self.dropout(self.attention_weights), v), self.max_score * math.sqrt(d) > threshold 

def masked_mean(X, valid_lens=None):
    if valid_lens is None:
      return X.sum(dim=1)
    else:
      X = X.permute(0,2,1)
      shape = X.shape
      valid_lens = torch.repeat_interleave(valid_lens, shape[1])
      masked_value = 0
      X = X.reshape(-1, shape[-1])
      maxlen = X.size(1)
      mask = torch.arange((maxlen),dtype=torch.float32,device=X.device)[None,:] < valid_lens[:,None]
      X[~mask] = masked_value
      valid_lens = valid_lens.reshape(shape[0], -1)
      return X.reshape(shape).sum(dim=-1)/valid_lens
    

class Attention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(Attention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, q,k,v):
        key = self.key_layer(k)
        query = self.query_layer(q)
        value = self.value_layer(v)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context
    

class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
      
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.alpha = nn.Parameter(torch.tensor(0.2))

        self.learnablePara = False
        self.attentionPara = False
        self.gatingPara = False
       
        

    def forward(self, query_vec, context_vec, context_mask=None):
       
       
        cross_attn_out, attn_weights = self.mha(
            query=query_vec, 
            key=context_vec, 
            value=context_vec, 
            key_padding_mask=(~context_mask) if context_mask is not None else None
        )
       
        return cross_attn_out, attn_weights    