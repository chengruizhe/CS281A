import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import SGD


def fit_mlp(model, data, l2_penalty, tol=1e-4, theta_init=None):
    train_set = TensorDataset(data)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=l2_penalty)
    criterion = nn.BCELoss()

    prev_loss, acc = evaluate_mlp(data, model)

    loss_list = [prev_loss]
    acc_list = [acc]
    i = 0
    gap = 1e30

    while gap > tol:
        train_loss = train(train_loader, model, optimizer, criterion)

        val_loss, val_acc = evaluate_mlp(data, model)
        loss_list.append(val_loss)
        acc_list.append(val_acc)

        gap = prev_loss - val_loss
        prev_loss = val_loss

        print(i, " | gap: ", gap, " | train loss: ", train_loss, " | val loss: ", val_loss, " | acc: ", val_acc)
        i += 1

    return loss_list, acc_list


def train(train_loader, model, optimizer, criterion):
    model.train()
    losses = AverageMeter('Loss', ':.4e')

    for batch_idx, inputs in enumerate(train_loader):
        inputs = inputs[0]
        X = inputs[:, :-1]
        Y = inputs[:, -2:-1]
        # X, Y = X.cuda(), Y.cuda()

        outputs = model(X)
        loss = criterion(outputs, Y)
        losses.update(loss.item(), X.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def evaluate_mlp(data, model):
    """

    Parameters
    ----------
    X
    Y
    model
    l2_penalty

    Returns
    -------
    loss
    acc
    """

    val_set = TensorDataset(data)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    criterion = nn.BCELoss()

    loss, acc = eval(val_loader, model, criterion)
    return loss, acc


def eval(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')

    for batch_idx, inputs in enumerate(val_loader):
        inputs = inputs[0]

        X = inputs[:, :-1]
        Y = inputs[:, -2:-1]
        # X, Y = X.cuda(), Y.cuda()

        outputs = model(X)
        loss = criterion(outputs, Y)
        losses.update(loss.item(), X.size(0))
        acc = torch.sum((outputs > 0.5) == Y) / X.size(0)
        accs.update(acc.item(), X.size(0))

    return losses.avg, accs.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)