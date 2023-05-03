#!/usr/bin/env python

"""
Example script to train a model to generate daily anatomical variations from a planning CT.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import h5py
import json
import torch
import numpy as np
import src as dam
import math

# parse training parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=16, help='batch size (default: 16)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs (default: 500)')
parser.add_argument('--dataset', default='data/train.h5', type=str, help='dataset path')
parser.add_argument('--load-model', type=str, help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
args = parser.parse_args()

# architecture and loss hyperparameters
with open('./hyperparam.json', 'r') as hfile:
    param = json.load(hfile)

# load and prepare training data
random.seed(333)
train_split = 0.95
fh = h5py.File(args.dataset, 'r')
listIDs = [*range(fh[param['xkey']].shape[-1])]
validIDs = listIDs[:24]
trainIDs = listIDs[24:]

# split data into training and validation
train_set = dam.generators.Volumes(trainIDs, fh,
    maxv=param['max'], minv=param['min'], masks=True)
valid_set = dam.generators.Volumes(validIDs, fh,
    maxv=param['max'], minv=param['min'], masks=True)

# initialize generators
gen_params={
    'batch_size': args.batch_size,
    'shuffle': param['shuffle'],
    'num_workers': param['num_workers']}
train_gen = torch.utils.data.DataLoader(train_set, **gen_params)
valid_gen = torch.utils.data.DataLoader(valid_set, **gen_params)

# extract shape from sampled input
inshape = tuple(reversed(fh[param['xkey']].shape[:-1]))

# prepare model folder
model_dir = 'models/' + time.strftime("%m%d-%H%M")
os.makedirs(model_dir, exist_ok=True)

# device handling
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# initilialize unet model
if args.load_model:
    # load initial model (if specified)
    model = dam.networks.DamBase.load(args.load_model, device)
else:
    # otherwise configure new model
    model = dam.networks.DamBase(
        inshape=inshape,
        latent_dim=param['latent_dim'],
        nb_unet_features=[param['enc_nf'], param['dec_nf']],
        int_steps=param['int_steps'],
        int_downsize=param['int_downsize'],
        nb_unet_conv_per_level=param['conv_per_level'],
        bidir=param['bidir'],
    )

# prepare the model for training
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if param['image_loss'] == 'ncc':
    image_loss_func = dam.losses.NCC(device=device).loss
elif param['image_loss'] == 'mse':
    image_loss_func = dam.losses.MSE().loss
else:
    raise ValueError("Invalid image loss")

# need two image loss functions if bidirectional
if param['bidir']:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5 * param['lambda'], 0.5 * param['lambda']]
else:
    losses = [image_loss_func]
    weights = [param['lambda']]

# prepare dice loss
if param['bidir']:
    losses += [dam.losses.Dice().loss, dam.losses.Dice().loss]
    weights += [param['kappa'], param['kappa']]

else:
    losses += [dam.losses.Dice().loss]
    weights += [param['kappa']]

# prepare deformation loss
losses += [dam.losses.Grad('l2', loss_mult=param['int_downsize']).loss]
weights += [param['beta']]

# add kl loss
losses += [dam.losses.KL(prior=param['prior']).loss]
weights += [param['kl_weight']]
best_loss = math.inf

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(model_dir, "%04d.pt" % epoch))

    epoch_loss = []
    epoch_total_loss = []
    val_loss = []
    val_total_loss = []
    start_time = time.time()

    for inputs, y_true, pmasks, rmasks in train_gen:

        # transfer to GPU
        inputs = inputs.to(device).float().permute(0, 4, 1, 2, 3)
        y_true = y_true.to(device).float().permute(0, 4, 1, 2, 3)
        pmasks = pmasks.to(device).to(torch.int64).permute(0, 4, 1, 2, 3)
        rmasks = rmasks.to(device).to(torch.int64).permute(0, 4, 1, 2, 3)

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(inputs, y_true, pmasks, rmasks)

        # convert organ masks to one-hot vectors
        pmasks = model.to_binary(pmasks, num_organs=1)
        rmasks = model.to_binary(rmasks, num_organs=1)

        # expand target list with inputs to different losses
        y_true = [y_true, inputs, rmasks, pmasks, 0, 0] if param['bidir'] else [y_true, rmasks, 0, 0]

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    with torch.set_grad_enabled(False):
        for inputs, y_true, pmasks, rmasks in valid_gen:

            # transfer to GPU
            inputs = inputs.to(device).float().permute(0, 4, 1, 2, 3)
            y_true = y_true.to(device).float().permute(0, 4, 1, 2, 3)
            pmasks = pmasks.to(device).to(torch.int64).permute(0, 4, 1, 2, 3)
            rmasks = rmasks.to(device).to(torch.int64).permute(0, 4, 1, 2, 3)

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(inputs, y_true, pmasks, rmasks)

            # convert organ masks to one-hot vectors
            pmasks = model.to_binary(pmasks, num_organs=1)
            rmasks = model.to_binary(rmasks, num_organs=1)

            # expand target list with inputs to different losses
            y_true = [y_true, inputs, rmasks, pmasks, 0, 0] if param['bidir'] else [y_true, rmasks, 0, 0]

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            val_loss.append(loss_list)
            val_total_loss.append(loss.item())

    # print epoch info
    epoch_info = "Epoch %d/%d" % (epoch + 1, args.epochs)
    time_info = "%.2f sec" % (time.time()-start_time)
    losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = "loss: %.4e  (%s)" % (np.mean(epoch_total_loss), losses_info)
    val_losses_info = ", ".join(["%.4e" % f for f in np.mean(val_loss, axis=0)])
    val_loss_info = "val loss: %.4e  (%s)" % (np.mean(val_total_loss), val_losses_info) 
    print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)

    # save best weights
    if np.mean(val_total_loss) < best_loss:
        best_loss = np.mean(val_total_loss)
        model.save(os.path.join(model_dir, "best.pt"))

# final model save
model.save(os.path.join(model_dir, "%04d.pt" % args.epochs))