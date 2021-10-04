import argparse
import json
import os
from os.path import join, exists

import numpy as np
from tqdm import tqdm
import cv2
import wandb
wandb.init(project="fcn")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cfm.dataset import DynamicsDataset
import cfm.models as cm
import cfm.utils as cu


def get_dataloaders(data_subset):
    train_subset = int(data_subset * 0.8)
    test_subset  = data_subset-train_subset
    train_dset = DynamicsDataset(root=join(args.root, 'train_data'))
    train_dset = torch.utils.data.Subset(train_dset,range(train_subset))
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4,
                                   pin_memory=True,)

    test_dset = DynamicsDataset(root=join(args.root, 'test_data'))
    test_dset = torch.utils.data.Subset(test_dset, range(test_subset))
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4,
                                  pin_memory=True)
    return train_loader, test_loader


def train(trans, optimizer, train_loader, epoch, device):
    trans.train()

    stats = cu.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    for batch in train_loader:
        obs, obs_pos, actions = [b.to(device) for b in batch]

        a = actions
        # Turn the actions into images
        obs, obs_pos, actions = trans.format_batch(obs, obs_pos, actions)
        # obs, obs_pos, actions = augment(obs, obs_pos, actions)

        # loss = F.mse_loss(trans(obs, actions), obs_pos)
        actions_h = trans(obs, obs_pos)
        # loss_nobs = torch.mean((obs_pos_h - obs_pos)**2)
        loss = F.mse_loss(actions_h, actions)

        # zn_h = trans.forward_emb(obs, actions)
        # zn   = trans.forward_emb(obs_pos, torch.zeros_like(actions))
        # loss_emb  = torch.mean((zn_h - zn)**2)

        # loss = loss_nobs# + loss_emb
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trans.parameters(), 20)
        optimizer.step()

        # img = trans.viz(obs, trans(obs_pos, torch.zeros_like(actions)).detach(), torch.zeros_like(actions), obs_pos_h.detach(), a)
        # cv2.imshow('img',img)
        # cv2.waitKey(1)
        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}')
        pbar.update(obs.shape[0])
    pbar.close()
    return stats


def test(trans, test_loader, epoch, device):
    trans.eval()

    test_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_pos, actions = [b.to(device) for b in batch]
            obs, obs_pos, actionsi = trans.format_batch(obs, obs_pos, actions)

            # loss = F.mse_loss(trans(obs, actions), obs_pos)
            actions_h = trans(obs, obs_pos)
            # loss_nobs = torch.mean((obs_pos_h - obs_pos)**2)
            loss = F.mse_loss(actions_h, actionsi)

            # zn_h = trans.forward_emb(obs, actionsi)
            # zn   = trans.forward_emb(obs_pos, torch.zeros_like(actionsi))
            #
            # loss_emb = torch.mean((zn_h - zn.detach())**2)
            # loss = loss_nobs# + loss_emb

            test_loss += loss * obs.shape[0]

    for i in range(obs.shape[0]):
        viz = trans.viz(obs[i], obs_pos[i], actions_h[i], actions[i])
    # cv2.imshow('viz',viz)
    # cv2.waitKey(1)
    wandb.log({"viz": [wandb.Image(viz, caption="viz_{}".format(epoch))]})
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    return test_loss.item()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)
    wandb.run.name = folder_name
    wandb.run.save()

    writer = SummaryWriter(join(folder_name, 'data'))

    save_args = vars(args)
    save_args['script'] = 'train_fcn_inv'
    with open(join(folder_name, 'params.json'), 'w') as f:
        json.dump(save_args, f)

    obs_dim = (3, 64, 64)
    if 'rope' in args.root:
        action_dim = 4
    elif 'cloth' in args.root:
        action_dim = 5
    else:
        raise Exception('Invalid environment, or environment needed in root name')

    device = torch.device('cuda')
    encoder = nn.Identity() # Just to keep the interface consistent for eval
    trans = cm.FCNInverse(obs_dim, action_dim, learn_delta=args.learn_delta).to(device)
    parameters = list(trans.parameters())

    optimizer = optim.Adam(parameters, lr=args.lr)
    train_loader, test_loader = get_dataloaders(args.data_subset)

    # Save training images
    batch = next(iter(train_loader))
    obs, obs_next, _ = batch
    imgs = torch.stack((obs, obs_next), dim=1).view(-1, *obs.shape[1:])
    cu.save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_seq_img.png'), nrow=8)

    best_test_loss = float('inf')
    itr = 0
    for epoch in range(args.epochs):
        stats = train(trans, optimizer, train_loader, epoch, device)
        test_loss = test(trans, test_loader, epoch, device)

        # Log metrics
        old_itr = itr
        for k, values in stats.items():
            itr = old_itr
            for v in values:
                writer.add_scalar(k, v, itr)
                wandb.log({k:v})
                itr += 1
        writer.add_scalar('test_loss', test_loss, epoch)
        wandb.log({"test_loss":test_loss})

        if epoch % args.log_interval == 0:
            if test_loss <= best_test_loss:
                best_test_loss = test_loss

                checkpoint = {
                    'encoder': encoder,
                    'trans': trans,
                    'optimizer': optimizer,
                }
                torch.save(checkpoint, join(folder_name, 'checkpoint'))
                print('Saved models with loss', best_test_loss)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    parser.add_argument('--root', type=str, default='data/rope', help='path to dataset (default: data/rope)')

    # Architecture Parameters
    parser.add_argument('--learn_delta', action='store_true', help='learn image delta instead of next image')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=50, help='default: 50')
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--batch_size', type=int, default=128, help='default 128')
    parser.add_argument('--data_subset', type=int, default=1000)

    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='fcn_inv')
    args = parser.parse_args()

    main()
