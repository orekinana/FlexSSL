import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .misc import AverageMeter, accuracy
import itertools


def train_regular(args, lbl_loader, nl_loader, model, optimizer, scheduler, epoch, itr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(lbl_loader, nl_loader)
    model.train()
    for batch_idx, (data_x, data_nl) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs_x, targets_x, _, nl_mask_x = data_x
        inputs_nl, targets_nl, _, nl_mask_nl = data_nl

        inputs = torch.cat((inputs_x, inputs_nl)).to(args.device)
        targets = torch.cat((targets_x, targets_nl)).to(args.device)
        nl_mask = torch.cat((nl_mask_x, nl_mask_nl)).to(args.device)

        #network outputs
        logits = model(inputs)

        positive_idx = nl_mask.sum(dim=1) == args.num_classes #the mask for negative learning is all ones
        nl_idx = (nl_mask.sum(dim=1) != args.num_classes) * (nl_mask.sum(dim=1) > 0)
        loss_ce = 0
        loss_nl = 0

        #positive learning
        if sum(positive_idx*1) > 0:
            loss_ce += F.cross_entropy(logits[positive_idx], targets[positive_idx], reduction='mean')

        #negative learning
        if sum(nl_idx*1) > 0:
            nl_logits = logits[nl_idx]
            pred_nl = F.softmax(nl_logits, dim=1)
            pred_nl = 1 - pred_nl
            pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
            nl_mask = nl_mask[nl_idx]
            y_nl = torch.ones((nl_logits.shape)).to(device=args.device, dtype=logits.dtype)
            loss_nl += torch.mean((-torch.sum((y_nl * torch.log(pred_nl))*nl_mask, dim = -1))/(torch.sum(nl_mask, dim = -1) + 1e-7))

        loss = loss_ce + loss_nl
        loss.backward()
        losses.update(loss.item())

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            p_bar.set_description("Train PL-Iter: {itr}/{itrs:4}. Epoch: {epoch}/{epochs:4}. BT-Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                itr=itr + 1,
                itrs=args.iterations,
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=scheduler.get_lr()[0],  #scheduler.get_last_lr()[0]
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()
    return losses.avg


def train_initial(args, train_loader, model, optimizer, scheduler, epoch, itr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    model.train()
    for batch_idx, (inputs, targets, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction='mean')
        loss.backward()
        losses.update(loss.item())

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            p_bar.set_description("Train PL-Iter: {itr}/{itrs:4}. Epoch: {epoch}/{epochs:4}. BT-Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                itr=itr + 1,
                itrs=args.iterations,
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=scheduler.get_lr()[0],  #scheduler.get_last_lr()[0]
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()

    return losses.avg


def train_sup(args, train_loader, model, optimizer, scheduler, epoch, itr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.train()
    for batch_idx, (inputs, targets, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction='mean')
        loss.backward()
        losses.update(loss.item())

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def train_scl(args, label_loader, unlabel_loader, model, discriminator, optimizer, scheduler, epoch, old_model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_B = AverageMeter()
    losses_C = AverageMeter()
    end = time.time()

    model.train()
    for batch_idx, ((label_x, label_y, _, _), (unlabel_x, unlabel_y, _, _)) in enumerate(zip(itertools.cycle(label_loader), unlabel_loader)):

        data_time.update(time.time() - end)

        label_x, label_y = label_x.to(args.device), label_y.to(args.device)
        unlabel_x, unlabel_y = unlabel_x.to(args.device), unlabel_y.to(args.device)

        if old_model is None:
            pseudo_label = torch.randint_like(unlabel_y, low=0, high=args.num_classes).to(args.device)
        else:
            pseudo_label = torch.argmax(old_model(unlabel_x), dim=-1)

        inputs = torch.cat([label_x, unlabel_x], dim=0)
        targets = torch.cat([label_y, pseudo_label], dim=0)

        logits = model(inputs)
        logits = torch.nn.Softmax(dim=-1)(logits)

        loss_A = F.cross_entropy(logits, targets, reduction='none')
        if args.nce:
            negative_labels = 1 - F.one_hot(targets, num_classes=args.num_classes)
            loss_nce = (negative_labels * -torch.log(1 - logits + 1e-7)).mean(dim=1)
            loss_A += loss_nce
            loss_A /= 2.0


        p = discriminator(inputs, logits.clone().detach(), loss_A.clone().detach())
        soft_mask = torch.tensor([0.9] * label_x.size(0) + [0.1] * unlabel_x.size(0), dtype=torch.float).unsqueeze(-1).to(args.device)
        loss_B = torch.nn.BCELoss(reduction='mean')(p, soft_mask)
        # print(' '.join([f'{pp.item():.2f}' for pp in p]))

        p1 = p.clone().detach().squeeze()
        soft_mask = soft_mask.squeeze()

        sample_weights = torch.zeros_like(p1, dtype=torch.float, device=args.device)
        sample_weights[soft_mask > 0.5] = (1 + 1 / p1[soft_mask > 0.5])
        sample_weights[soft_mask < 0.5] = (1 - args.weight_factor * 1 / (1 - p1[soft_mask < 0.5]))
        if args.weight_mean:
            sample_weights /= sample_weights.sum()
        # print('\n' + ' '.join([f'{w:.4f}' for w in sample_weights]))
        loss_C = (sample_weights * loss_A).sum()

        loss = loss_C

        loss.backward()
        loss_B.backward()

        optimizer.step()
        scheduler.step()
        model.zero_grad()
        discriminator.zero_grad()

        losses.update(loss.item())
        losses_B.update(loss_B.item())
        losses_C.update(loss_C.item())
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, losses_B.avg, losses_C.avg
