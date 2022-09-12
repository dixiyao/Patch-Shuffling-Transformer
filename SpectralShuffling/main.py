# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

'''Tranfer pretrained T2T-ViT to downstream dataset: CIFAR10/CIFAR100.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utilsenc

import os
import argparse

import math
import DataLoaders
from timm.models import *
from utils import progress_bar
from timm.models import create_model
from utils import load_for_transfer_learning

import model

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='cifar10 or cifar100 or celeba or imdb')
parser.add_argument('--b', type=int, default=64,
                    help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--model', default='T2t_vit_24', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
# Transfer learning
parser.add_argument('--transfer-learning', default=False,
                    help='Enable transfer learning')
parser.add_argument('--transfer-model', type=str, default="./checkpoints/T2t_vit_24_pretrained.pth.tar",
                    help='Path to pretrained model for transfer learning')
parser.add_argument('--transfer-ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--k', type=float, default=1.,
                    help='BatchPatchShuffle Parameter k1 not using class attention')
parser.add_argument('--transform', type=str, default='None',
                    help='gdp or blur or None')
parser.add_argument('--transform-value', type=int, default=4,
                    help='Parameter for gdp for blur transformation')
parser.add_argument('--epoch', type=int, default=60, metavar='N',
                    help='Training Epoch')
parser.add_argument('--datapath', type=str, default='../../data',
                    help='Default path for placing dataset')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

if args.dataset=='cifar10':
    args.num_classes = 10
elif args.dataset=='svhn':
    args.num_classes = 10
elif args.dataset=='cifar100':
    args.num_classes = 100
elif args.dataset=='celeba':
    args.num_classes = 40
elif args.dataset=='imdb':
    args.num_classes = 2
else:
    print('Please use cifar10 or cifar100 or celeba dataset.')

trainloader = DataLoaders.get_loader(args.dataset,args.datapath, args.b, attr='train', num_workers=8)
testloader = DataLoaders.get_loader(args.dataset, args.datapath, args.b, attr='valid', num_workers=8)

#重跑 random, patch batch shuffle
print(f'learning rate:{args.lr}, weight decay: {args.wd}')
# create T2T-ViT Model
print('==> Building model..')
net=model.t2t_pretrain(k=args.k)

net = net.to(device)

if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.dataset=='cifar10' or args.dataset=='cifar100' or args.dataset=='imdb' or args.dataset=='svhn':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

# set optimizer
parameters = [{'params': net.MLP.parameters()},
                {'params': net.pos_embed},
                {'params': net.model.blocks.parameters(), 'lr': args.transfer_ratio * args.lr},
                {'params': net.model.head.parameters()}]


optimizer = optim.SGD(parameters, lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=args.epoch)

log_loss=[]
log_acc=[]

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs= net(inputs)
        if args.dataset=='celeba':
            targets=targets.float()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if args.dataset=='cifar10' or args.dataset=='cifar100' or args.dataset=='imdb' or args.dataset=='svhn':
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        else:
            predicted = (outputs > 0.5).long()
            correct += predicted.eq(targets).float().mean(dim=1).sum().item()
        total += targets.size(0)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        break

    global log_loss
    log_loss.append(train_loss/(batch_idx+1))

def confusion(prediction, truth):
    confusion_vector = prediction / truth
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    return true_positives, false_positives, true_negatives, false_negatives


def mcc(prediction, truth):
    TP,FP,TN,FN=confusion(prediction, truth)
    mcc=(TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return mcc

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    mccs=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if args.dataset == 'celeba':
                targets = targets.float()
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            if args.dataset=='cifar10' or args.dataset=='cifar100' or args.dataset=='imdb' or args.dataset=='svhn':
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            else:
                predicted = (outputs > 0.5).long()
                correct += predicted.eq(targets).float().mean(dim=1).sum().item()
                mccs += mcc(predicted, targets)
            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            break

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'checkpoint_{args.dataset}_{args.model}_{args.k}'):
            os.mkdir(f'checkpoint_{args.dataset}_{args.model}_{args.k}')
        torch.save(state, f'./checkpoint_{args.dataset}_{args.model}_{args.k}/ckpt_{args.lr}_{args.wd}.pth')
        best_acc = acc

    global log_acc
    log_acc.append(acc)

    print('testepoch %3d,acc: %.3f' % (epoch, acc))
    if args.dataset == 'celeba':
        print('testepoch %3d,mcc: %.3f' % (epoch, mccs / len(testloader)))


for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
