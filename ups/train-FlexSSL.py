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
from utils.train_util import train_initial, train_regular, train_sup, train_scl
from utils.evaluate import test
from utils.pseudo_labeling_util import pseudo_labeling

from models.scl import Discriminator


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

    parser.add_argument('--weight-factor', default=0.1, type=float,
                        help='weight factor (alpha)')
    parser.add_argument('--freq-update-label', default=10, type=int,
                        help='frequency of updating label')
    parser.add_argument('--pretrain-epochs', default=1024, type=int,
                        help='number of epochs of pretrain')
    parser.add_argument('--load-pretrain', action='store_true',
                        help="load pretrain")
    parser.add_argument('--exp-name', default='scl', type=str,
                        help='exp name')
    parser.add_argument('--num-unlabel', default=46000, type=int,
                        help='number of unlabel data')
    parser.add_argument('--unlabel-batch-size', default=128, type=int,
                        help='batch size of unlabel data')
    parser.add_argument('--weight-mean', action='store_true',
                        help="weight mean")
    parser.add_argument('--optimizer', default='Adam', type=str,
                        help='Optimizer')
    parser.add_argument('--nce', action='store_true',
                        help="NCE Loss")
    parser.add_argument('--freq-enlarge-unlabel-data-size', default=-1, type=int,
                        help="enlarge size of unlabel dataset")
    parser.add_argument('--freq-enlarge-unlabel-batch-size', default=-1, type=int,
                        help="enlarge batch size of unlabel dataloader")
    parser.add_argument('--freq-update-weight-factor', default=-1, type=int,
                        help="freq of update weight factor")
    parser.add_argument('--cos-unlabel-bs', action='store_true',
                        help="cosine unlabel batch size")



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

    args.batch_size = args.batchsize
    args.epochs = args.epchs

    if args.weight_mean:
        args.exp_name += '_weightmean'
    if args.nce:
        args.exp_name += '_nce'
    if args.freq_enlarge_unlabel_data_size > 0:
        args.exp_name += f'_enunds{args.freq_enlarge_unlabel_data_size}'
    if args.freq_enlarge_unlabel_batch_size > 0:
        args.exp_name += f'_enunbs{args.freq_enlarge_unlabel_batch_size}'
    if args.freq_update_weight_factor > 0:
        args.exp_name += f'_updatealpha{args.freq_update_weight_factor}'
    if args.cos_unlabel_bs:
        args.exp_name += f'_cosunbs'

    DATASET_GETTERS = {'cifar10': get_cifar10, 'cifar100': get_cifar100}
    exp_name = f'{args.exp_name}_pretrain_discriminatorcnn13_alpha{args.weight_factor:.1f}_frequpdatelabel{args.freq_update_label}_numunlabel{args.num_unlabel}_unlabelbatchsize{args.unlabel_batch_size}_optimizer{args.optimizer}_pretrainepochs{args.pretrain_epochs}_{args.dataset}_{args.n_lbl}_{args.arch}_{args.split_txt}_{args.epchs}_{args.class_blnc}_{args.tau_p}_{args.tau_n}_{args.kappa_p}_{args.kappa_n}_{run_started}'
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

    origin_unlbl_dataset = deepcopy(unlbl_dataset)

    unlbl_dataset.data = origin_unlbl_dataset.data[:args.num_unlabel]
    unlbl_dataset.targets = origin_unlbl_dataset.targets[:args.num_unlabel]

    model = create_model(args)
    model.to(args.device)
    old_model = create_model(args)
    old_model.to(args.device)
    discriminator = Discriminator()
    discriminator.to(args.device)

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
        batch_size=args.unlabel_batch_size,
        num_workers=args.num_workers)

    print(len(lbl_dataset))
    print(len(unlbl_dataset))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    args.total_steps = args.epochs * args.iteration
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)

    if args.resume and os.path.exists(args.resume):
        print(f'loading state dict from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=torch.device('cuda:0'))
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_satet_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            pass
        old_model.load_state_dict(checkpoint['state_dict'])
    else:
        if args.load_pretrain:
            print('loading pretrain')
            checkpoint = torch.load(f'./outputs/scl_pretrain_discriminatorcnn13_alpha0.6_frequpdatelabel256_cifar10_4000_cnn13_run1_10240_7_0.7_0.05_0.05_0.005_15-11-21_1230/checkpoint_epoch_1024.pth.tar', map_location=torch.device('cuda:0'))
            model.load_state_dict(checkpoint['state_dict'])
            old_model.load_state_dict(checkpoint['state_dict'])
        else:
            print('starting pretrain')
            model.zero_grad()
            for pretrain_epoch in range(args.pretrain_epochs):
                train_loss = train_sup(args, lbl_loader, model, optimizer, scheduler, pretrain_epoch, 0)
                test_model = model
                test_loss, test_acc = test(args, test_loader, test_model)
                writer.add_scalar('test/1.test_acc', test_acc, pretrain_epoch)
                writer.add_scalar('test/2.test_loss', test_loss, pretrain_epoch)
                writer.add_scalar('train/1.train_loss', train_loss, pretrain_epoch)
                print(f'epoch: {pretrain_epoch} train_loss: {train_loss:.8f} test_loss: {test_loss:8f} test_acc: {test_acc:.8f}')
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint({
                'epoch': args.pretrain_epochs,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, False, args.out, f'pretrain_epoch_{args.pretrain_epochs}')
            checkpoint = torch.load(f'{args.out}/checkpoint_pretrain_epoch_{args.pretrain_epochs}.pth.tar', map_location=torch.device('cuda:0'))
            old_model.load_state_dict(checkpoint['state_dict'])

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD([
            { 'params': model.parameters() },
            { 'params': discriminator.parameters() },
        ], lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    else:
        optimizer = optim.Adam([
            { 'params': model.parameters() },
            { 'params': discriminator.parameters() },
        ])
    args.total_steps = args.epochs * args.iteration
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)
    model.zero_grad()
    now_num_unlabel = args.num_unlabel
    now_bs_unlabel = args.unlabel_batch_size
    unlabel_batch_sizes = [64, 80, 96, 112, 96, 80]
    for epoch in range(0, args.epochs):

        if args.freq_enlarge_unlabel_data_size > 0 and epoch % args.freq_enlarge_unlabel_data_size == (args.freq_enlarge_unlabel_data_size - 1):
            now_num_unlabel += args.num_unlabel
            unlbl_dataset.data = origin_unlbl_dataset.data[:now_num_unlabel]
            unlbl_dataset.targets = origin_unlbl_dataset.targets[:now_num_unlabel]
            print(f'now_num_unlabel: {now_num_unlabel}')
        if args.freq_enlarge_unlabel_batch_size > 0 and epoch % args.freq_enlarge_unlabel_batch_size == (args.freq_enlarge_unlabel_batch_size - 1):
            now_bs_unlabel += 16
            now_bs_unlabel = min(112, now_bs_unlabel)
            print(f'now_bs_unlabel: {now_bs_unlabel}')
        if args.cos_unlabel_bs:
            now_bs_unlabel = unlabel_batch_sizes[(epoch // 128) % 6]
            print(f'now_bs_unlabel: {now_bs_unlabel}')
        unlbl_loader = DataLoader(
            unlbl_dataset,
            sampler=SequentialSampler(unlbl_dataset),
            batch_size=now_bs_unlabel,
            num_workers=args.num_workers)

        if args.freq_update_weight_factor > 0 and epoch % args.freq_update_weight_factor == (args.freq_update_weight_factor - 1):
            args.weight_factor -= 0.1
            args.weight_factor = max(0.1, args.weight_factor)
            print(f'now alpha: {args.weight_factor}')

        if epoch < args.start_epoch:
            continue

        if epoch % args.freq_update_label == (args.freq_update_label - 1):

            print(f'saving model to {args.out}/checkpoint_epoch_{epoch}.pth.tar')
            model_to_save = model.module if hasattr(model, "module") else model
            discriminator_to_save = discriminator.module if hasattr(discriminator, "module") else discriminator

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                'discriminator_satet_dict': discriminator_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, False, args.out, f'epoch_{epoch}')

            print(f'loading state dict from {args.out}/checkpoint_epoch_{epoch}.pth.tar')
            checkpoint = torch.load(f'{args.out}/checkpoint_epoch_{epoch}.pth.tar', map_location=torch.device('cuda:0'))
            model.load_state_dict(checkpoint['state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_satet_dict'])
            if args.optimizer.lower() == 'sgd':
                optimizer = optim.SGD([
                    { 'params': model.parameters() },
                    { 'params': discriminator.parameters() },
                ], lr=args.lr, momentum=0.9, nesterov=args.nesterov)
            else:
                optimizer = optim.Adam([
                    { 'params': model.parameters() },
                    { 'params': discriminator.parameters() },
                ])
            args.total_steps = args.epochs * args.iteration
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)
            model.zero_grad()
            discriminator.zero_grad()
            # try:
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            #     scheduler.load_state_dict(checkpoint['scheduler'])
            # except:
            #     pass
            print(f'updating label by loading model from {args.out}/epoch_{epoch}.pth.tar')
            old_model.load_state_dict(checkpoint['state_dict'])

        train_loss, loss_B, loss_C = train_scl(args, lbl_loader, unlbl_loader, model, discriminator, optimizer, scheduler, epoch, old_model)

        test_model = model
        test_loss, test_acc = test(args, test_loader, test_model)
        writer.add_scalar('test/1.test_acc', test_acc, epoch)
        writer.add_scalar('test/2.test_loss', test_loss, epoch)
        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        writer.add_scalar('train/1.loss_B', loss_B, epoch)
        writer.add_scalar('train/1.loss_C', loss_C, epoch)
        print(f'epoch: {epoch} train_loss: {train_loss:.8f} loss_B: {loss_B:.8f} loss_C: {loss_C:.8f} test_loss: {test_loss:8f} test_acc: {test_acc:.8f}')

    writer.close()


if __name__ == '__main__':
    cudnn.benchmark = True
    main()