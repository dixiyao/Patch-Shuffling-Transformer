import copy

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from  einops import rearrange
import timm
import utilsenc
from timm.models.layers import trunc_normal_

from transformers import AdamW,get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
class t2t_pretrain(torch.nn.Module):
    def __init__(self,net,k=1):
        super(t2t_pretrain, self).__init__()
        self.model = net
        self.k=k

    def forward_features(self, x):
        x = self.model.tokens_to_token(x)
        x =utilsenc.BatchPatchPartialShuffle(x,self.k)
        cls_token = self.model.cls_token.expand(x.shape[0], -1,-1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.blocks[:1](x)
        x =x[:,1:,:]

        return x

    def forward(self, x):
        x= self.forward_features(x)
        return x

def WBBatchPatchShuffle(x,row_perm,batch_perm,k=0.1):
    percent = int(row_perm.shape[1] * k)
    for _ in range(x.ndim - 2): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(2)], *(x.shape[2:]))  # reformat this for the gather operation
    shuffle_part = x.gather(1, row_perm)
    keep_part = shuffle_part[:, :percent, :]

    random_part = shuffle_part[:, percent:, :]
    b, n, d = random_part.shape
    random_part = random_part.reshape(b * n, d)
    random_part = random_part[batch_perm, :]
    random_part = random_part.reshape(b, n, d)
    input = torch.cat((keep_part, random_part), dim=1)
    perm_back = row_perm.argsort(1)
    x = input.gather(1, perm_back)
    return x

class WB(torch.nn.Module):
    def __init__(self,net,batch_size,k=1):
        super(WB, self).__init__()
        self.model = net
        self.k1 = k
        self.row_perm= torch.rand((batch_size,196)).argsort(1).cuda()
        self.batch_perm=torch.randperm(batch_size*int(196-int(196*self.k1)))

    def forward_features(self, x):
        x = self.model.tokens_to_token(x)
        patch_token=x
        patch_token = WBBatchPatchShuffle(patch_token,copy.deepcopy(self.row_perm),copy.deepcopy(self.batch_perm),self.k1)  # BatchPatchPartialShuffle(patch_token,self.k1,self.k2)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, patch_token), dim=1)
        x = self.model.blocks[:1](x)
        x = x[:, 1:, :]
        return x

    def forward(self, x, output_intermediate=False):
        x= self.forward_features(x)
        return x