import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import init
import numpy as np



class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        _, b_q, _, nq, _ = queries.shape
        b_k, _, _, nk, _ = keys.shape

        q = queries.permute(0,1, 2, 3, 4)  # (b_s, h, nq, d_k)
        k = keys.permute(0,1, 2, 4, 3)  # (b_s, h, d_k, nk)
        v = values.permute(0, 1, 2, 3)  # (b_s, h, nk, d_v)


        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)


        att = torch.softmax(att, -1)
        att = att.mean(1)

        out = torch.matmul(att,v)
        out = out.contiguous().view(b_k, nk, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        out = out.transpose(-1, -2)
        return out

class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)



    def forward(self, queries, keys, values):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        _,b_q,_, nq,_= queries.shape
        b_k,_,_,nk,_ = keys.shape

        q = queries.permute(0,1, 2, 3, 4)  # (b_s, h, nq, d_k)
        k = keys.permute(0,1, 2, 4, 3)  # (b_s, h, d_k, nk)
        v = values.permute(0, 1, 2, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k)

        att = att / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1)
        att = att.mean(1)
        out = torch.matmul(att, v)
        out = out.contiguous().view(b_k, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out


class PositionAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7,head=1):
        super().__init__()
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)
        self.gamma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.head = head
    def forward(self, query_a,key_a,value_a,query_b,key_b,value_b):


        query_sq = query_a.permute(0,1,3,2)  # bs,bq,h*w,c
        key_sq = key_b.permute(0,1,3,2)
        value_sq = value_b.permute(0,1, 3, 2)  # bs,bq,h*w,c


        query_sq = query_sq.unsqueeze(0)
        key_sq = key_sq.unsqueeze(1)



        y = self.pa(query_sq, key_sq, value_sq)  # bs,h*w,c
        value_b_res = value_b.contiguous().view(value_b.size(0),value_b.size(2)*self.head ,-1)


        pam_x = self.gamma * y + value_b_res

        return pam_x


class ChannelAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7,head=1):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=1)
        self.gamma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.head = head
    def forward(self, query_a,key_a,value_a,query_b,key_b,value_b):

        query_sq = query_a.unsqueeze(0)
        key_sq = key_b.unsqueeze(1)
        value_sq = value_b

        y = self.pa(query_sq, key_sq, value_sq)  # bs,c,h*w

        value_b_res = value_b.contiguous().view(value_b.size(0), value_b.size(2) , value_b.size(3)*self.head)
        cam_x = self.gamma * y+ value_b_res
        return cam_x


class CSCAModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7,h=1):
        super().__init__()
        self.position_attention_module = PositionAttentionModule(d_model=d_model, kernel_size=kernel_size, H=5, W=5,head=h)
        self.channel_attention_module = ChannelAttentionModule(d_model=d_model, kernel_size=kernel_size, H=5, W=5,head=h)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.num_heads = h


        self.to_qkv_p = nn.Sequential(
            nn.Linear(d_model, d_model * 3, bias=False),
            )

        self.to_qkv_c = nn.Sequential(
            nn.Linear(H*W, H*W * 3, bias=False),
            )

        self.dropout_p = nn.Dropout2d(0.1)
        self.dropout_c = nn.Dropout2d(0.1)
        self.dropout_o = nn.Dropout2d(0.1)


    def forward(self,support, query):
        b_s, c_s, h_s, w_s = support.shape
        b_q, c_q, h_q, w_q = query.shape

        s = support.contiguous().view(b_s,c_s, -1) # b_s,c_s,h_s*w_s
        q = query.contiguous().view(b_q, c_q, -1) # b_q,c_q,h_q*w_q


        # credit to https://github.com/PRIS-CV/Bi-FRN/
        qkv_s_c = self.to_qkv_c(s)
        # (3,b,num_heads,l,dim_per_head)
        qkv_a_c = qkv_s_c.contiguous().view(b_s, c_s, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        query_a_c, key_a_c, value_a_c = qkv_a_c.chunk(3)

        query_a_c, key_a_c, value_a_c = query_a_c.squeeze(0), key_a_c.squeeze(0), value_a_c.squeeze(0)

        # (b,l,dim_all_heads x 3)
        qkv_b_c = self.to_qkv_c(q)
        # (3,b,num_heads,l,dim_per_head)
        qkv_b_c = qkv_b_c.contiguous().view(b_q, c_q, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()

        # 3 x (1,b,num_heads,l,dim_per_head)
        query_b_c, key_b_c, value_b_c = qkv_b_c.chunk(3)

        query_b_c, key_b_c, value_b_c = query_b_c.squeeze(0), key_b_c.squeeze(0), value_b_c.squeeze(0)

        s = s.permute(0,2, 1)
        q = q.permute(0, 2, 1)

        # (b,l,dim_all_heads x 3)
        qkv_s_p = self.to_qkv_p(s)
        # (3,b,num_heads,l,dim_per_head)
        qkv_a_p = qkv_s_p.contiguous().view(b_s, c_s, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        query_a_p, key_a_p, value_a_p = qkv_a_p.chunk(3)

        query_a_p, key_a_p, value_a_p = query_a_p.squeeze(0), key_a_p.squeeze(0), value_a_p.squeeze(0)

        # (b,l,dim_all_heads x 3)
        qkv_b_p = self.to_qkv_p(q)
        # (3,b,num_heads,l,dim_per_head)
        qkv_b_p = qkv_b_p.contiguous().view(b_q, c_q, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # 3 x (1,b,num_heads,l,dim_per_head)
        query_b_p, key_b_p, value_b_p = qkv_b_p.chunk(3)
        #[450, 1, 25, 64]
        query_b_p, key_b_p, value_b_p = query_b_p.squeeze(0), key_b_p.squeeze(0), value_b_p.squeeze(0)


        p_out_q = self.position_attention_module(query_a_p, key_a_p, value_a_p,query_b_p, key_b_p, value_b_p)
        p_out_q = self.dropout_p(p_out_q)
        c_out_q = self.channel_attention_module(query_a_c, key_a_c, value_a_c,query_b_c, key_b_c, value_b_c)
        c_out_q = self.dropout_c(c_out_q)


        sum_pc_q =  p_out_q +c_out_q
        sum_pc_q = sum_pc_q.contiguous().view(b_q,c_q,h_q,w_q)
        sum_pc_q = self.gamma*sum_pc_q + query
        sum_pc_q = self.dropout_o(sum_pc_q)/3


        sum_pc_s = value_a_p + value_a_c
        sum_pc_s = sum_pc_s.contiguous().view(b_s, c_s, h_s,w_s)
        sum_pc_s = self.beta * sum_pc_s + support
        sum_pc_s = self.dropout_o(sum_pc_s)/3

        return sum_pc_s,sum_pc_q

class CSCAM(nn.Module):
    def __init__(self,
                 sequence_length=25,
                 embedding_dim=64,
                 *args, **kwargs):
        super(CSCAM, self).__init__()
        self.CSCAModule = CSCAModule(d_model=embedding_dim, kernel_size=3, H=5, W=5,h=1)

    def forward(self, support,query):
        att_s,att_q = self.CSCAModule(support,query)
        return att_s,att_q

