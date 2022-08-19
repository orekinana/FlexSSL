import argparse
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from collections import OrderedDict
import pickle
import numpy as np
from re import search
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from data.cifar import get_cifar10, get_cifar100
from utils import AverageMeter, accuracy
from utils.utils import *
from utils.train_util import train_initial, train_regular, train_sup
from utils.evaluate import test
from utils.pseudo_labeling_util import pseudo_labeling


def main():
    run_started = datetime.today().strftime('%d-%m-%y_%H%M') #start time to create unique experiment name
    parser = argparse.ArgumentParser(description='UPS Training')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset names')
    parser.add_argument('--n-lbl', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='cnn13', type=str,
                        choices=['wideresnet', 'cnn13', 'shakeshake'],
                        help='architecture name')
    parser.add_argument('--iterations', default=20, type=int,
                        help='number of total pseudo-labeling iterations to run')
    parser.add_argument('--epchs', default=1024, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batchsize', default=128, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate, default 0.03')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='dropout probs')
    parser.add_argument('--num-classes', default=10, type=int,
                        help='total classes')
    parser.add_argument('--class-blnc', default=10, type=int,
                        help='total number of class balanced iterations')
    parser.add_argument('--tau-p', default=0.70, type=float,
                        help='confidece threshold for positive pseudo-labels, default 0.70')
    parser.add_argument('--tau-n', default=0.05, type=float,
                        help='confidece threshold for negative pseudo-labels, default 0.05')
    parser.add_argument('--kappa-p', default=0.05, type=float,
                        help='uncertainty threshold for positive pseudo-labels, default 0.05')
    parser.add_argument('--kappa-n', default=0.005, type=float,
                        help='uncertainty threshold for negative pseudo-labels, default 0.005')
    parser.add_argument('--temp-nl', default=2.0, type=float,
                        help='temperature for generating negative pseduo-labels, default 2.0')
    parser.add_argument('--no-uncertainty', action='store_true',
                        help='use uncertainty in the pesudo-label selection, default true')
    parser.add_argument('--split-txt', default='run1', type=str,
                        help='extra text to differentiate different experiments. it also creates a new labeled/unlabeled split')
    parser.add_argument('--model-width', default=2, type=int,
                        help='model width for WRN-28')
    parser.add_argument('--model-depth', default=28, type=int,
                        help='model depth for WRN')
    parser.add_argument('--test-freq', default=10, type=int,
                        help='frequency of evaluations')

    args = parser.parse_args()
    #print key configurations
    print('########################################################################')
    print('########################################################################')
    print(f'dataset:                                  {args.dataset}')
    print(f'number of labeled samples:                {args.n_lbl}')
    print(f'architecture:                             {args.arch}')
    print(f'number of pseudo-labeling iterations:     {args.iterations}')
    print(f'number of epochs:                         {args.epchs}')
    print(f'batch size:                               {args.batchsize}')
    print(f'lr:                                       {args.lr}')
    print(f'value of tau_p:                           {args.tau_p}')
    print(f'value of tau_n:                           {args.tau_n}')
    print(f'value of kappa_p:                         {args.kappa_p}')
    print(f'value of kappa_n:                         {args.kappa_n}')
    print('########################################################################')
    print('########################################################################')

    args.batch_size = 64
    args.epochs = 1024

    DATASET_GETTERS = {'cifar10': get_cifar10, 'cifar100': get_cifar100}
    exp_name = f'sup_{args.dataset}_{args.n_lbl}_{args.arch}_{args.split_txt}_{args.epchs}_{args.class_blnc}_{args.tau_p}_{args.tau_n}_{args.kappa_p}_{args.kappa_n}_{run_started}'
    device = torch.device('cuda', args.gpu_id)
    args.device = device
    args.exp_name = exp_name
    args.dtype = torch.float32
    if args.seed != -1:
        set_seed(args)
    args.out = os.path.join(args.out, args.exp_name)

    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100

    lbl_dataset, nl_dataset, unlbl_dataset, test_dataset = \
        DATASET_GETTERS[args.dataset](
            'data/datasets', args.n_lbl, None, None, 0, args.split_txt
        )

    model = create_model(args)
    model.to(args.device)

    nl_batchsize = int((float(args.batch_size) * len(nl_dataset))/(len(lbl_dataset) + len(nl_dataset)))

    lbl_batchsize = args.batch_size
    args.iteration = len(lbl_dataset) // args.batch_size

    lbl_loader = DataLoader(
        lbl_dataset,
        sampler=RandomSampler(lbl_dataset),
        batch_size=lbl_batchsize,
        num_workers=args.num_workers,
        drop_last=True)

    nl_loader = DataLoader(
        nl_dataset,
        sampler=RandomSampler(nl_dataset),
        batch_size=nl_batchsize,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    unlbl_loader = DataLoader(
        unlbl_dataset,
        sampler=SequentialSampler(unlbl_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    args.total_steps = args.epochs * args.iteration
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)
    start_epoch = 0

    print(len(lbl_dataset))

    model.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_sup(args, lbl_loader, model, optimizer, scheduler, epoch, 0)

        test_model = model
        test_loss, test_acc = test(args, test_loader, test_model)
        writer.add_scalar('test/1.test_acc', test_acc, epoch)
        writer.add_scalar('test/2.test_loss', test_loss, epoch)
        writer.add_scalar('train/1.train_loss', train_loss, epoch)

        print(f'epoch: {epoch} train_loss: {train_loss:.8f} test_loss: {test_loss:8f} test_acc: {test_acc:.8f}')

    writer.close()


if __name__ == '__main__':
    cudnn.benchmark = True
    main()