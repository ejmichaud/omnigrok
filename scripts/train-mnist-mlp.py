#!/usr/bin/env python
# coding: utf-8
"""
This script performs training runs on MNIST, with lots of configuration options!
    TODO: track weight norms throughout training

"""

from itertools import islice
import random

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("train-mnist-mlp")
ex.captured_out_filter = apply_backspaces_and_linefeeds


# --------------------------
#    ,   ,
#   /////|
#  ///// | Some definitions
# |~~~|  |    and helper
# |===|  |      functions
# |j  |  |
# | g |  |
# |  s| /
# |===|/
# '---'
# --------------------------

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}

def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    """Computes accuracy of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()

def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50):
    """Computes mean loss of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device)
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            y = network(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
        return total / points



# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------
@ex.config
def cfg():
    
    # training parameters
    train_points = 1000
    optimization_steps = 10000
    batch_size = 256
    loss_function = 'MSE' # 'MSE' or 'CrossEntropy'
    optimizer = 'AdamW'
    weight_decay = 0.1
    lr = 1e-3
    initialization_scale = 7.0
    download_directory = "data/"

    # architecture parameters
    depth = 3 # the number of nn.Linear modules in the model
    width = 200
    activation = 'ReLU' # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

    # logging parameters
    log_freq = 1 # log(steps) == log_freq --> do a logging step
    verbose = True

    # computing parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    dnn = True
    dataset_name = "MNIST"

    overwrite = None


# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(train_points,
        optimization_steps,
        batch_size,
        loss_function,
        optimizer,
        weight_decay,
        lr,
        initialization_scale,
        download_directory,
        depth,
        width,
        activation,
        log_freq,
        verbose,
        device,
        dtype,
        dnn,
        dataset_name,
        seed,
        _log):
   
    device = torch.device(device)
    
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # load dataset
    ds = eval(f"torchvision.datasets.{dataset_name}")
    if dataset_name == "MNIST":
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # thank you copilot for those numbers
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
      ])
      image_size = 28
      image_height = 28
      
    elif dataset_name == "CIFAR10":
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # thank you copilot for those numbers
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
      ])
      image_size = 32
        
    train = ds(root=download_directory, train=True, 
        transform=transforms, download=True)
    test = ds(root=download_directory, train=False, 
        transform=transforms, download=True)
    train = torch.utils.data.Subset(train, range(train_points))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    assert activation in activation_dict, f"Unsupported activation function: {activation}"
    activation_fn = activation_dict[activation]

    # create model
    def create_dnn():
      layers = [nn.Flatten()]
      for i in range(depth):
          if i == 0:
              layers.append(nn.Linear(image_size**2, width))
              layers.append(activation_fn())
          elif i == depth - 1:
              layers.append(nn.Linear(width, 10))
          else:
              layers.append(nn.Linear(width, width))
              layers.append(activation_fn())
      return nn.Sequential(*layers).to(device)

    def create_cnn():
      layers = []
      if dataset_name == "MNIST":
        in_channels = 1
      elif dataset_name == "CIFAR10":
        in_channels = 3
      n_channels = width//5
      for i in range(depth):
          if i == 0:
              layers.append(nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1))
              layers.append(activation_fn())
          else:
              layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
              layers.append(nn.MaxPool2d(2,2))
              layers.append(activation_fn())
      layers.append(nn.Flatten())
      resolution = image_size//2**(depth - 1)
      layers.append(nn.Linear(n_channels*resolution*resolution, 10))
      return nn.Sequential(*layers).to(device)
    
    if dnn:
      network_model = create_dnn()
    else:
      network_model = create_cnn()
    
    with torch.no_grad():
        for p in network_model.parameters():
            p.data = initialization_scale * p.data
    _log.debug("Created model.")

    # create optimizer
    assert optimizer in optimizer_dict, f"Unsupported optimizer choice: {optimizer}"
    optimizer = optimizer_dict[optimizer](network_model.parameters(), lr=lr, weight_decay=weight_decay)

    # define loss function
    assert loss_function in loss_function_dict
    loss_fn = loss_function_dict[loss_function]()

    # prepare for logging
    
    ex.info['log_steps'] = []
    ex.info['l2'] = []
    ex.info['last_layer_l2'] = []
    ex.info['train'] = {
        'loss': [],
        'accuracy': []
    }
    ex.info['val'] = {
        'loss': [],
        'accuracy': []
    }
    
    steps = 0
    one_hots = torch.eye(10, 10).to(device)
    with tqdm(total=optimization_steps, disable=not verbose) as pbar:
        for x, labels in islice(cycle(train_loader), optimization_steps):
            if steps == 0 or np.log2(steps) % log_freq == 0:
                ex.info['train']['loss'].append(compute_loss(network_model, train, loss_function, device, N=len(train)))
                ex.info['train']['accuracy'].append(compute_accuracy(network_model, train, device, N=len(train)))
                ex.info['val']['loss'].append(compute_loss(network_model, test, loss_function, device, N=len(test)))
                ex.info['val']['accuracy'].append(compute_accuracy(network_model, test, device, N=len(test)))
                ex.info['log_steps'].append(steps)
                with torch.no_grad():
                    total = sum(torch.pow(p, 2).sum() for p in network_model.parameters())
                    ex.info['l2'].append(np.sqrt(total.item()))
                    last_layer = sum(torch.pow(p, 2).sum() for p in network_model[-1].parameters())
                    ex.info['last_layer_l2'].append(np.sqrt(last_layer.item()))
                pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                    ex.info['train']['loss'][-1],
                    ex.info['val']['loss'][-1],
                    ex.info['train']['accuracy'][-1] * 100, 
                    ex.info['val']['accuracy'][-1] * 100))

            optimizer.zero_grad()
            y = network_model(x.to(device))
            if loss_function == 'CrossEntropy':
                loss = loss_fn(y, labels.to(device))
            elif loss_function == 'MSE':
                loss = loss_fn(y, one_hots[labels])
            loss.backward()
            optimizer.step()
            steps += 1
            pbar.update(1)

