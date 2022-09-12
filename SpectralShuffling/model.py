import timm
import torch
from PIL import Image
import requests

import utilsenc
import math

from timm.models.vision_transformer import Block
import torch.nn as nn
from timm.models.layers import trunc_normal_,Mlp,PatchEmbed
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

class SpectralPooling2d(nn.Module):
    def __init__(self, H, W):
        super(SpectralPooling2d, self).__init__()
        self.H = H
        self.W = W

    def crop_spectrum(self, z, H, W):
        M, N = z.size(-2), z.size(-1)
        return z[..., (M - H)//2:(M + H)//2, (N - W)//2:(N +W)//2]

    def pad_spectrum(self, z, M, N):
        H, W = z.size(-2), z.size(-1)
        pad = torch.nn.ZeroPad2d((M - H)//2, (M - H)//2, (N - W)//2, (N - W)//2)
        return pad(z)

    def forward(self, x):
        x=torch.fft.fftshift(x,dim=(-2,-1))
        M, N = x.size(-2), x.size(-1)
        self.M = M
        self.N = N
        crop_x_fft = self.crop_spectrum(x, self.H, self.W)
        pool_x = crop_x_fft
        return pool_x

    def backward(self, gRgx):
        H, W = gRgx.size(-2), gRgx.size(-1)
        M, N = self.M, self.N
        z = gRgx
        z = self.pad_spectrum(z, M, N)
        gRx = z
        return gRx


# classes
class ToSpectral(nn.Module):
    def __init__(self):
        super(ToSpectral, self).__init__()

    def forward(self,x):
        x=torch.fft.fft2(x)
        x=torch.fft.fftshift(x,dim=(-2,-1))
        real=x.real
        image=x.imag
        x=torch.cat([real,image],dim=1)
        return x

class ToTime(nn.Module):
    def __init__(self):
        super(ToTime, self).__init__()

    def forward(self,x):
        a=x[:,:3,...]
        b=x[:,3:,...]
        x=torch.complex(a,b)
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x=torch.fft.ifft2(x).real
        return x

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

class t2t_pretrain(torch.nn.Module):
    def __init__(self,k=1):
        super(t2t_pretrain, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224',num_classes=40,pretrained=True)
        self.k=k
        self.pos_embed=torch.nn.Parameter(torch.randn(1, 197, 768) * .02)
        trunc_normal_(self.pos_embed,.02)
        self.MLP=torch.nn.Sequential(Mlp(768*2,out_features=768))
        self.FFT=ToSpectral()
        self.pool=SpectralPooling2d(56,56)

    def forward_features(self, x):
        with torch.no_grad():
            x = self.model.patch_embed(x)
            x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=16, p2=16, h=14)
            x=self.FFT(x)
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16,p2=16)
            x = utilsenc.BatchPatchPartialShuffle(x, self.k)
        x=self.MLP(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x= x+self.pos_embed
        x  = self.model.blocks(x)
        x = self.model.norm(x)
        return x[:, 0]

    def forward(self, x):
        x= self.forward_features(x)
        x = self.model.head(x)
        return x
