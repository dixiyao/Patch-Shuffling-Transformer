import torch
from torch.autograd import Variable
import time
import copy
import numpy as np
import pytorch_lightning as pl
import argparse
from einops import rearrange
from timm.models.layers import Mlp

import DataLoaders
import model
import torchvision
from models import t2t_vit_24
from utils import load_for_transfer_learning
import utilsenc
from facenet_pytorch import MTCNN, InceptionResnetV1
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='cifar10 or cifar100 or celeba or imdb')
parser.add_argument('--b', type=int, default=8,
                    help='batch size')
parser.add_argument('--k', type=float, default=1.,
                    help='BatchPatchShuffle Parameter k1 not using class attention')
parser.add_argument('--datapath', type=str, default='../../../data',
                    help='Default path for placing dataset')

args = parser.parse_args()

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1 / torch.sqrt(mse))

class deep_feature(torch.nn.Module):
    def __init__(self):
        super(deep_feature, self).__init__()
        self.model=InceptionResnetV1(pretrained='vggface2',classify=False)

    def forward(self,x):
        x=self.model(x)
        return x


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return (x ** 2).mean()

def main():

    batch_size=args.b
    valid_loader = DataLoaders.get_loader(args.dataset, args.datapath, batch_size, 'valid', num_workers=8,
                                          pin_memory=True)

    net_kernel = t2t_vit_24(num_classes=10)
    print('transfer learning, load t2t-vit pretrained model')
    load_for_transfer_learning(net_kernel, "../checkpoints/T2t_vit_24_pretrained.pth.tar", use_ema=True, strict=False,num_classes=10)
    F=model.t2t_pretrain(net_kernel,k=args.k)
    WB = model.WB(copy.deepcopy(net_kernel),args.b,k=args.k)

    img=None
    for img, _ in valid_loader:
        break

    vis = torchvision.utils.make_grid(img, nrow=8, padding=1, normalize=True)
    vis = torchvision.transforms.ToPILImage()(vis)
    vis.save('org.png')
    F=F.cuda()
    WB=WB.cuda()
    F=F.train()
    WB=WB.train()
    x_pred = torch.nn.Parameter(torch.zeros(img.shape).cuda(),requires_grad=True)#torch.rand(img.shape)
    img=img.cuda()
    input_opt = torch.optim.Adam([x_pred], lr=0.001, amsgrad=True)
    # model_opt = torch.optim.Adam(WB.parameters(), lr=0.001, amsgrad=True)
    mse = torch.nn.MSELoss().cuda()
    target=F(img)

    loss_x=0
    best=100

    for main_iter in range(5000):
        for i in range(100):
            input_opt.zero_grad()
            pred = WB(x_pred)
            loss = mse(pred, target)#+ 0.1 * TV(x_pred)+ l2loss(x_pred)
            loss_x=loss.item()
            loss.backward(retain_graph=True)
            input_opt.step()

        if main_iter%10==0:
            print('current iter %d, loss x %f, loss mdoel %f, pic mse %f'%(main_iter,loss_x,0,mse(x_pred,img).item()))

        if main_iter%100==0:
            if mse(x_pred,img).item()<best:
                best=mse(x_pred,img).item()
                x_output=copy.deepcopy(x_pred)
                x_out=x_output.detach().cpu()
                np.save(f'whitebox_{args.k}_{args.transform}_2.npy',x_out.numpy())
                vis = torchvision.utils.make_grid(x_out, nrow=8, padding=1, normalize=True)
                vis = torchvision.transforms.ToPILImage()(vis)
                vis.save(f'whitebox_{args.k}_{args.transform}_2.png')


    metric_mse = torch.nn.MSELoss()
    metric_ssim = SSIM(data_range=1., size_average=True, channel=3)
    psnr = PSNR()
    deep_feature_net = deep_feature()
    deep_feature_net = deep_feature_net.cuda()
    mse = metric_mse(x_pred, img).item()
    ssim = metric_ssim(x_pred, img).item()
    psnr_score = psnr(x_pred, img).item()

    input_deep_feature = deep_feature_net(img)
    output_deep_feature = deep_feature_net(x_pred)
    feature_sim = torch.mean(torch.abs(torch.cosine_similarity(input_deep_feature, output_deep_feature, dim=1)))
    Fsim = feature_sim.item()
    print('MSE: %f SSIM: %f PSNR: %f Fsim: %f' % (mse,ssim,psnr_score,Fsim))

def valid():
    batch_size = args.b
    valid_loader = DataLoaders.get_loader(args.dataset, args.datapath, batch_size, 'valid')#, num_workers=8, pin_memory=True)
    img = None

    for img, _ in valid_loader:
        break

    x_pred=torch.from_numpy(np.load('whitebox_1.0_gaussian_2.npy'))
    x_pred-=torch.min(x_pred)
    x_pred/=torch.max(x_pred)
    metric_mse = torch.nn.MSELoss()
    metric_ssim = SSIM(data_range=1., size_average=True, channel=3)
    psnr = PSNR()
    deep_feature_net = deep_feature()
    deep_feature_net = deep_feature_net
    mse = metric_mse(x_pred, img).item()
    ssim = metric_ssim(x_pred, img).item()
    psnr_score = psnr(x_pred, img).item()

    input_deep_feature = deep_feature_net(img)
    output_deep_feature = deep_feature_net(x_pred)
    feature_sim = torch.mean(torch.abs(torch.cosine_similarity(input_deep_feature, output_deep_feature, dim=1)))
    Fsim = feature_sim.item()
    print('MSE: %f SSIM: %f PSNR: %f Fsim: %f' % (mse,ssim,psnr_score,Fsim))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()