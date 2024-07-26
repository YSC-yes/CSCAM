import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from datasets import dataloaders
from tqdm import tqdm


def get_score(acc_list):

    mean = np.mean(acc_list)
    interval = 1.96*np.sqrt(np.var(acc_list)/len(acc_list))

    return mean,interval


def meta_test(data_path,model,way,shot,pre,transform_type,query_shot=16,trial=10000,return_list=False):

    eval_loader = dataloaders.meta_test_dataloader(data_path=data_path,
                                                way=way,
                                                shot=shot,
                                                pre=pre,
                                                transform_type=transform_type,
                                                query_shot=query_shot,
                                                trial=trial)
    
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()

    acc_list = []

    file_path = './Matrix_index_%ds'%shot

    for i, (inp,_) in tqdm(enumerate(eval_loader)):

        inp = inp.cuda()
        max_index, neg_12_dist = model.meta_test(inp,way=way,shot=shot,query_shot=query_shot)

        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way
        acc_list.append(acc)

        if(acc>95):
            neg_12_dist_npy = F.softmax(neg_12_dist, dim = 0).cpu().numpy()
            np.save(file_path, neg_12_dist_npy)


    if return_list:
        return np.array(acc_list)
    else:
        mean,interval = get_score(acc_list)
        return mean,interval