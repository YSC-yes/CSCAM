import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
import Conv_4,ResNet,CSCAM
import argparse
import cv2
import numpy as np



def pdist(x,y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return dist


class Proto(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False):
        
        super().__init__()
        self.change_channel = 256
        if resnet:
            self.dim = 640
            self.change = nn.Sequential(
                nn.Conv2d(self.dim, self.change_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.change_channel),
                nn.ReLU(inplace=True)
            )
            self.CSCAM = CSCAM.CSCAM(sequence_length=25, embedding_dim=self.change_channel)
            self.feature_extractor = ResNet.resnet12()
        else:
            num_channel = 64
            self.change = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_channel),
                nn.ReLU(inplace=True)
            )
            self.CSCAM = CSCAM.CSCAM(sequence_length=25, embedding_dim=num_channel)
            self.feature_extractor = Conv_4.BackBone(num_channel)            
            self.dim = num_channel*5*5

        self.shots = shots
        self.way = way
        self.resnet = resnet
        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
            

    def get_feature_vector(self,inp):

        feature_map = self.feature_extractor(inp)

        return feature_map
    

    def get_neg_l2_dist(self,inp,way,shot,query_shot):
        feature_map = self.get_feature_vector(inp)
        feature_map = self.change(feature_map)

        support = feature_map[:way*shot]
        query = feature_map[way*shot:]

        support, query = self.CSCAM(support, query)

        if self.resnet:

            support = F.avg_pool2d(input=support,kernel_size=support.size(-1))
            query = F.avg_pool2d(input=query,kernel_size=query.size(-1))

            support = support.contiguous().view(way,shot,-1)
            query = query.contiguous().view(way*query_shot, -1)
        else:
            support = support.contiguous().view(way,shot,-1)
            query = query.contiguous().view(way*query_shot, -1)
        centroid = torch.mean(support,1) # way,dim
        neg_l2_dist = pdist(query,centroid).neg().view(way*query_shot,way) #way*query_shot,way
        
        return neg_l2_dist



    
    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index


    def forward(self,inp):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=self.way,
                                        shot=self.shots[0],
                                        query_shot=self.shots[1])
        
        logits = neg_l2_dist/self.dim*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction