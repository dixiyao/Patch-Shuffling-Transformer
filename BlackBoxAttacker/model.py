import timm
import torch
import torch.nn as nn
from PIL import Image
import requests
from  einops import rearrange

from models import *
import utilsenc
from timm.models.layers import trunc_normal_
#hugging face
from transformers import ViTFeatureExtractor, ViTForImageClassification

class F(torch.nn.Module):
    def __init__(self,net,k=1):
        super(F, self).__init__()
        self.model=net
        self.k=k

    def forward_features(self, x,output_intermediate=False):
        x = self.model.tokens_to_token(x)
        #do something
        x =utilsenc.BatchPatchPartialShuffle(x,self.k)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.blocks[:1](x)
        return x

    def forward(self, x):
        x= self.forward_features(x)
        return x[:,1:,:]

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=10)
        self.tanh=torch.nn.Tanh()
        self.mlp=timm.models.layers.mlp.Mlp(512,out_features=768)

    def forward_features(self, x):
        x = self.mlp(x)
        x =self.model.pos_drop(x+self.model.pos_embed[:,1:,:])
        x  = self.model.blocks(x)
        x = self.model.norm(x)
        x = self.tanh(x)
        x = rearrange(x,'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=3,p1=16,h=14)
        return  x


    def forward(self, x):
        x=self.forward_features(x)
        return x
