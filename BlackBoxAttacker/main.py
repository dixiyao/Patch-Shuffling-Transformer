import torch
from torch.autograd import Variable
import time
import copy
import numpy as np
import pytorch_lightning as pl

import DataLoaders
import trainer

def train_G(batch_size,max_epochs,net,wrapper):
    train_loader = DataLoaders.get_loader('cifar10', '../../../data', batch_size, 'train', num_workers=8, pin_memory=True)
    valid_loader = DataLoaders.get_loader('cifar10', '../../../data', batch_size, 'valid', num_workers=8, pin_memory=True)

    net.total_steps = ((len(train_loader.dataset) // (batch_size)) // 1 * float(max_epochs))
    wrapper.fit(net, train_loader, valid_loader)
    wrapper.test(net, valid_loader)

def train():
    batch_size=50
    max_epochs = 1

    net = trainer.TrainWrapper(checkpoints='./checkpoints/t2t_vit_24_ImageNet2Cifar10_Raw.pth',num_labels=40,k1=1.,k2=1)
    wrapper = pl.Trainer(gpus=1,precision=32,max_epochs=max_epochs,default_root_dir='./log/')
    train_G(batch_size,max_epochs,net,wrapper)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()
