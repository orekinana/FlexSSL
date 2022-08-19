import argparse

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.mnist import FashionMNIST

from data import PartialFashionMNIST, noise_dataset, split_dataset, update_label_dataset, mislabel_dataset
from model import resnet18
from utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser('SCL Vision - Fashion MNIST')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--pretrain-epochs', type=int, default=50, help='number of epoches')
    parser.add_argument('--epochs', type=int, default=300, help='number of epoches')
    parser.add_argument('--update-model-freq', type=int, default=10, help='number of epoches')
    parser.add_argument('--data-ratio', type=float, default=1.0, help='ratio of dataset.')
    parser.add_argument('--mislabel-sample', action='store_true', help='mislabel sample')
    parser.add_argument('--noise-sample', action='store_true', help='noise sample')
    parser.add_argument('--noise-sample-ratio', type=float, default=0.5, help='ratio of dataset.')
    parser.add_argument('--sample-noise-ratio', type=float, default=0.05, help='ratio of dataset.')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0')
    return args


def main(args):
    ROOT_DIR = '/data/SCL/fm/datasets/fashion_mnist'

    model = resnet18().to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters())

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = PartialFashionMNIST(root=ROOT_DIR, partial=0.1, train=True, transform=train_transforms)
    if args.mislabel_sample:
        print('mislabel sample')
        train_set = mislabel_dataset(train_set, args.noise_sample_ratio)
    elif args.noise_sample:
        print('noise sample')
        train_set = noise_dataset(train_set, args.noise_sample_ratio, args.sample_noise_ratio)

    label_dataset, unlabel_dataset = split_dataset(train_set, ratio=args.data_ratio)
    label_loader = torch.utils.data.DataLoader(label_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    test_set = FashionMNIST(root=ROOT_DIR, train=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    for epoch in range(args.pretrain_epochs):
        train_loss, train_acc = run_model(args, model, label_loader, criterion, optimizer)
        val_loss, val_acc = run_model(args, model, test_loader, criterion, optimizer, False)
        stats = f'[Suervised Pretrain] [Epoch]: {epoch} [train_loss]: {train_loss:.3f} [train_acc]: {train_acc:.3f} [val_loss]: {val_loss:.3f} [val_acc]: {val_acc:.3f}'
        print(stats)

    best_acc = -1
    cnt = 0
    now_label_dataset = label_dataset
    now_unlabel_dataset = unlabel_dataset
    print(f'init: len(now_label_dataset): {len(now_label_dataset)}')
    print(f'init: len(now_unlabel_dataset): {len(now_unlabel_dataset)}')
    now_label_loader = torch.utils.data.DataLoader(now_label_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    now_unlabel_loader = torch.utils.data.DataLoader(now_unlabel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    for epoch in range(args.pretrain_epochs, args.epochs):
        if len(now_unlabel_dataset) > 1 and cnt % args.update_model_freq == 0:
            predict_labels = infer(args, model, now_unlabel_loader)
            now_label_dataset, now_unlabel_dataset = update_label_dataset(now_label_dataset, now_unlabel_dataset, predict_labels)
            print(f'len(now_label_dataset): {len(now_label_dataset)}')
            print(f'len(now_unlabel_dataset): {len(now_unlabel_dataset)}')
            now_label_loader = torch.utils.data.DataLoader(now_label_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
            now_unlabel_loader = torch.utils.data.DataLoader(now_unlabel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        train_loss, train_acc = run_model(args, model, now_label_loader, criterion, optimizer)
        val_loss, val_acc = run_model(args, model, test_loader, criterion, optimizer, False)
        stats = f'[Self-suervised] [Epoch]: {epoch} [train_loss]: {train_loss:.3f} [train_acc]: {train_acc:.3f} [val_loss]: {val_loss:.3f} [val_acc]: {val_acc:.3f}'
        if len(now_unlabel_dataset) == 0:
            stats = '[No Unlabeled Sample] ' + stats
        print(stats)

        if val_acc > best_acc:
            best_acc = val_acc

        cnt += 1

    print(f'best acc: {best_acc}')


def run_model(args, model: torch.nn.Module, data_loader, criterion, optimizer, train=True):
    running_loss = 0
    running_accuracy = 0

    model.train(mode=train)

    for i, (x, y) in enumerate(data_loader):
        if x.size(0) < 2:
            continue

        x, y = x.to(args.device), y.to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            try:
                output = model(x)
            except:
                print(f'ERROR!!!!! Batch: {i} {x.size()}')
                continue
                # import pdb; pdb.set_trace()
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())

    return running_loss / len(data_loader), running_accuracy.double() / len(data_loader.dataset)


def infer(args, model: torch.nn.Module, data_loader):
    model.eval()
    with torch.set_grad_enabled(False):
        outputs = []
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(args.device), y.to(args.device)
            output = model(x)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        return outputs.cpu()



if __name__ == '__main__':
    args = parse_args()
    print(args)
    setup_seed(args.seed)
    main(args)
