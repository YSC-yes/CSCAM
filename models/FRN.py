import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
from .backbones import Conv_4,ResNet,CSCAM


class FRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False,num_cat=None,self_attention_model=None,cross_attention_model=None):
        
        super().__init__()
        self.change_channel = 256
        if resnet:
            num_channel = 640
            self.change = nn.Sequential(
                nn.Conv2d(self.d, self.change_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.change_channel),
                nn.ReLU(inplace=True)
            )
            # 通道交叉注意力模块+空间交叉注意力模块（混合域）
            self.CSCAM = CSCAM.CSCAM(sequence_length=self.resolution, embedding_dim=self.change_channel)
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.change = nn.Sequential(
                nn.Conv2d(self.d, self.d, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.d),
                nn.ReLU(inplace=True)
            )
            self.CSCAM = CSCAM.CSCAM(sequence_length=self.resolution, embedding_dim=self.d)
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.shots = shots
        self.way = way
        self.resnet = resnet
        self.self_attention_model = self_attention_model
        self.cross_attention_model = cross_attention_model

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel

        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        self.num_heads = 1
        self.qkv_bias = None
        self.attn_drop = 0
        self.proj_drop = 0
        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25 
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        self.s1 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        self.s2 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)




        if is_pretraining:

            self.num_cat = num_cat

            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)   
    

    def get_feature_map(self,inp):

        batch_size = inp.size(0)

        feature_map = self.feature_extractor(inp)

        if self.resnet:
            feature_map = feature_map/np.sqrt(640)
        
        return feature_map.contiguous().view(batch_size,self.d,-1).permute(0,2,1).contiguous() # N,HW,C
    

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)
        
        # correspond to lambda in the paper
        lam = reg*alpha.exp()+1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution
        if Woodbury:
            # correspond to Equation 10 in the paper
            # print("support",support.shape)   support torch.Size([10, 125, 640])
            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d
        
        else:
            # correspond to Equation 8 in the paper
            
            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d
        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        return dist

    
    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=False):
        
        resolution = self.resolution
        alpha = self.r[0]
        beta = self.r[1]

        feature_map = self.get_feature_map(inp)

        feature_map = feature_map.transpose(-1,-2).contiguous().view((way*(shot+query_shot)),self.d,5,5)
        feature_map = self.change(feature_map)

        support = feature_map[:way*shot]
        query = feature_map[way*shot:]
        support,query = self.CSCAM(support,query)




        if self.resnet:
            support = support.contiguous().view(way * shot, self.change_channel, self.resolution).transpose(-1, -2).contiguous().view(
                way, shot * resolution, self.change_channel)
            query = query.contiguous().view(way * query_shot, self.change_channel, self.resolution).transpose(-1, -2).contiguous().view(
                way * query_shot * resolution, self.change_channel)
            recon_dist = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta,Woodbury=False) # way*query_shot*resolution, way
        else:
            support = support.contiguous().view(way * shot, self.d, self.resolution).transpose(-1, -2).contiguous().view(
                way, shot * resolution, self.d)
            query = query.contiguous().view(way * query_shot, self.d, self.resolution).transpose(-1, -2).contiguous().view(
                way * query_shot * resolution, self.d)
            recon_dist = self.get_recon_dist(query=query, support=support, alpha=alpha, beta=beta,Woodbury=True)  # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().contiguous().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way


        # exit()
        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist


    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index
    

    def forward_pretrain(self,inp):

        feature_map = self.get_feature_map(inp)

        batch_size = feature_map.size(0)

        feature_map = feature_map.contiguous().view(batch_size*self.resolution,self.d)
        
        alpha = self.r[0]
        beta = self.r[1]
        
        recon_dist = self.get_recon_dist(query=feature_map,support=self.cat_mat,alpha=alpha,beta=beta) # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().contiguous().view(batch_size,self.resolution,self.num_cat).mean(1) # batch_size,num_cat
        
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction

    def forward(self, inp):

        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction, support
