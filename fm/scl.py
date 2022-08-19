import argparse

import numpy as np

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.mnist import FashionMNIST

from data import MaskFashionMNIST, mask_dataset_to_label_dataset, mislabel_dataset, noise_dataset, update_mask_dataset
from model import resnet18, Discriminator
from utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser('SCL Vision - Fashion MNIST')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--pretrain-model-epochs', type=int, default=50, help='number of epoches')
    parser.add_argument('--pretrain-discriminator-epochs', type=int, default=100, help='number of epoches')
    parser.add_argument('--epochs', type=int, default=300, help='number of epoches')
    parser.add_argument('--update-model-freq', type=int, default=10, help='number of epoches')
    parser.add_argument('--data-ratio', type=float, default=1.0, help='ratio of dataset.')
    parser.add_argument('--mislabel-sample', action='store_true', help='mislabel sample')
    parser.add_argument('--noise-sample', action='store_true', help='noise sample')
    parser.add_argument('--noise-sample-ratio', type=float, default=0.5, help='ratio of dataset.')
    parser.add_argument('--sample-noise-ratio', type=float, default=0.05, help='ratio of dataset.')
    parser.add_argument('--weight-factor', type=float, default=0.5, help='ratio of dataset.')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0')
    return args


def main(args):
    ROOT_DIR = 'datasets/fashion_mnist'

    model = resnet18().to(args.device)
    discriminator = Discriminator().to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam([
        { 'params': model.parameters() },
        { 'params': discriminator.parameters() },
    ])

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = MaskFashionMNIST(root=ROOT_DIR, partial=0.1, ratio=args.data_ratio, train=True, transform=train_transforms)
    if args.mislabel_sample:
        print('mislabel sample')
        train_set = mislabel_dataset(train_set, args.noise_sample_ratio)
    elif args.noise_sample:
        print('noise sample')
        train_set = noise_dataset(train_set, args.noise_sample_ratio, args.sample_noise_ratio)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_set = FashionMNIST(root=ROOT_DIR, train=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    pretrain_dataset = mask_dataset_to_label_dataset(train_set)
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    pretrain_model_optimizer = torch.optim.Adam(model.parameters())
    print(f'len(pretrain_dataset): {len(pretrain_dataset)}')
    for epoch in range(args.pretrain_model_epochs):
        train_loss, train_acc = pretrain_model(args, model, pretrain_loader, criterion, pretrain_model_optimizer)
        val_loss, val_acc = test(args, model, test_loader, criterion)
        stats = f'[Suervised Pretrain] [Epoch]: {epoch} [train_loss]: {train_loss:.3f} [train_acc]: {train_acc:.3f} [val_loss]: {val_loss:.3f} [val_acc]: {val_acc:.3f}'
        print(stats)

    pretrain_discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    for epoch in range(args.pretrain_model_epochs, args.pretrain_discriminator_epochs):
        loss_B = pretrain_disctiminator(args, model, discriminator, train_loader, pretrain_discriminator_optimizer)
        stats = f'[Discriminator Pretrain] [Epoch]: {epoch} [loss_B]: {loss_B:.3f}'
        print(stats)

    print(f'len(train_set): {len(train_set)}, labeled train_set: {sum(train_set.mask)}')
    print(f'len(test_set): {len(test_set)}')
    best_acc = -1
    cnt = 0
    now_train_set = train_set
    now_train_loader = torch.utils.data.DataLoader(now_train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    for epoch in range(args.pretrain_discriminator_epochs, args.epochs):

        if cnt % args.update_model_freq == 0:
            print('update labels')
            now_train_loader_for_infer = torch.utils.data.DataLoader(now_train_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
            predict_labels = infer(args, model, now_train_loader_for_infer)
            now_train_set = update_mask_dataset(now_train_set, predict_labels)
            now_train_loader = torch.utils.data.DataLoader(now_train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)

        train_loss, train_loss_B, train_loss_C, train_acc, label_p, unlabel_p = train(args, epoch, model, discriminator, now_train_loader, criterion, optimizer)
        val_loss, val_acc = test(args, model, test_loader, criterion)
        stats = f'[Epoch]: {epoch} [train_loss]: {train_loss:.3f} [loss_B]: {train_loss_B:.3f} [loss_C]: {train_loss_C:.3f} [label_p]: {label_p:.3f} [unlabel_p]: {unlabel_p:.3f} [train_acc]: {train_acc:.3f} [val_loss]: {val_loss:.3f} [val_acc]: {val_acc:.3f}'
        print(stats)

        if val_acc > best_acc:
            best_acc = val_acc

        cnt += 1

    print(f'best acc: {best_acc}')


def infer(args, model: torch.nn.Module, data_loader):
    model.eval()
    with torch.set_grad_enabled(False):
        outputs = []
        for i, (x, y, _) in enumerate(data_loader):
            x, y = x.to(args.device), y.to(args.device)
            output = model(x)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        return outputs.cpu()


def pretrain_model(args, model: torch.nn.Module, data_loader, criterion, optimizer):
    running_loss = 0
    running_accuracy = 0

    model.train(mode=True)

    for i, (x, y, _) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(x)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())

    return running_loss / len(data_loader), running_accuracy.double() / len(data_loader.dataset)


def pretrain_disctiminator(args, model: torch.nn.Module, discriminator: torch.nn.Module, data_loader, optimizer):
    running_loss_B = 0
    model.train(mode=False)
    discriminator.train(mode=True)
    for i, (x, y, mask) in enumerate(data_loader):
        x, y, mask = x.to(args.device), y.to(args.device), mask.to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y_hat = model(x)
            y_hat = torch.nn.Softmax(dim=1)(y_hat)

            loss_A = torch.nn.CrossEntropyLoss(reduction='none')(y_hat, y)

            p = discriminator(x, y_hat.detach(), loss_A.detach())
            loss_B = torch.nn.BCELoss(reduction='mean')(p, mask.unsqueeze(dim=-1).float())

        loss_B.backward()
        optimizer.step()

        running_loss_B += loss_B.item()

    return running_loss_B / len(data_loader)


def train(args, epoch, model: torch.nn.Module, discriminator: torch.nn.Module, data_loader, criterion, optimizer):
    running_loss = 0
    running_loss_B = 0
    running_loss_C = 0
    running_accuracy = 0

    model.train(mode=True)
    discriminator.train(mode=True)

    ps = []
    masks = []

    for i, (x, y, mask) in enumerate(data_loader):
        x, y, mask = x.to(args.device), y.to(args.device), mask.to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y_hat = model(x)
            y_hat = torch.nn.Softmax(dim=1)(y_hat)

            loss_A = torch.nn.CrossEntropyLoss(reduction='none')(y_hat, y)

            p = discriminator(x, y_hat.clone().detach(), loss_A.clone().detach())
            soft_mask = mask.unsqueeze(dim=-1).float().clone().detach()
            soft_mask[soft_mask > 0.9] = 0.9
            soft_mask[soft_mask < 0.1] = 0.1
            loss_B = torch.nn.BCELoss(reduction='mean')(p, soft_mask)

            p1 = p.clone().detach().squeeze()
            # debug_log = ' '.join([ f'{d.item():.4f}' for d in (max(p1), min(p1)) ])
            # print(f'  ==> [Batch]: {i:3d} p: [ {debug_log} ]')
            # p1[p1 > 0.9] = 0.9
            # p1[p1 < 0.1] = 0.1
            ps.append(p1.detach().cpu())
            masks.append(mask.detach().cpu())

            sample_weights = torch.zeros_like(p1, dtype=torch.float, device=args.device)
            num_label = (mask > 0.5).sum()
            num_unlabel = (mask < 0.5).sum()
            sample_weights[mask > 0.5] = (1 + 1 / p1[mask > 0.5]) / num_label
            sample_weights[mask < 0.5] = (1 - args.weight_factor * 1 / (1 - p1[mask < 0.5])) / num_unlabel
            loss_C = (sample_weights * loss_A).sum() / sample_weights.sum()

            loss = loss_B + loss_C

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_B += loss_B.item()
        running_loss_C += loss_C.item()
        running_accuracy += torch.sum(y_hat.argmax(dim=1).detach() == y.detach())

    ps = torch.cat(ps, dim=0)
    masks = torch.cat(masks, dim=0)
    label_ps = ps[masks > 0.5]
    unlabel_ps = ps[masks < 0.5]
    mean_label_p = label_ps.mean()
    mean_unlabel_p = unlabel_ps.mean()
    np.save(f'logs/discriminator_output/{epoch}_label_p.npy', label_ps.numpy())
    np.save(f'logs/discriminator_output/{epoch}_unlabel_p.npy', unlabel_ps.numpy())

    return running_loss / len(data_loader), running_loss_B / len(data_loader), running_loss_C / len(data_loader), running_accuracy.double() / len(data_loader.dataset), mean_label_p, mean_unlabel_p


def test(args, model: torch.nn.Module, data_loader, criterion):
    running_loss = 0
    running_accuracy = 0

    model.train(mode=False)
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)

        with torch.set_grad_enabled(False):
            output = model(x)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())

    return running_loss / len(data_loader), running_accuracy.double() / len(data_loader.dataset)


# def run_model(args, model: torch.nn.Module, discriminator: torch.nn.Module, data_loader, criterion, optimizer, train=True):
#     running_loss = 0
#     running_accuracy = 0
#
#     model.train(mode=train)
#
#     for i, (x, y) in enumerate(data_loader):
#         x, y = x.to(args.device), y.to(args.device)
#
#         optimizer.zero_grad()
#
#         with torch.set_grad_enabled(train):
#             output = model(x)
#             _, pred = torch.max(output, 1)
#             loss = criterion(output, y)
#
#         if train:
#             loss.backward()
#             optimizer.step()
#
#         running_loss += loss.item()
#         running_accuracy += torch.sum(pred == y.detach())
#
#     return running_loss / len(data_loader), running_accuracy.double() / len(data_loader.dataset)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    setup_seed(args.seed)
    main(args)
