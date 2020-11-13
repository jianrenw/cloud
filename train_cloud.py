import argparse
import json
import os
from os.path import join, exists

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from tools.dataloader import SampleDataset
from tools.utils import Stats
import model.cloud_FI as cloud

def get_dataloader():
    dataloader = {}
    for name in ["train", "test"]:
        dataset = SampleDataset(root=join(args.datadir, name)) 
        dataloader[name] = data.DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=4,
                                    pin_memory=True)
    return dataloader

def get_cloud_loss(obs, obs_pos, encoder, acts_encoder, acts_decoder, trans, inverse_trans, actions, device):
    bs = obs.shape[0]
    actshape = actions.shape[0]

    z, z_pos = encoder(obs), encoder(obs_pos)
    actions_unsqueeze = torch.unsqueeze(actions, 2)
    qacts = acts_encoder(actions_unsqueeze)
    qacts_decode = acts_decoder(qacts)
    qacts_decode = torch.squeeze(qacts_decode, 2)
    acts_recon_loss = F.mse_loss(qacts_decode, actions)
    
    z_next = trans(z, qacts_decode)
    
    qacts_inv = inverse_trans(z, z_pos)
    loss_acts = F.mse_loss(qacts_inv, actions)

    neg_dot_products = torch.mm(z_next, z.t())
    neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2* neg_dot_products + (z ** 2).sum(1).unsqueeze(0))
    idxs = np.arange(bs)
    neg_dists[idxs, idxs] = float('-inf')

    pos_dot_products = (z_pos * z_next).sum(dim=1)
    pos_dists = -((z_pos ** 2).sum(1) - 2* pos_dot_products + (z_next ** 2).sum(1))
    pos_dists = pos_dists.unsqueeze(1)

    dists = torch.cat((neg_dists, pos_dists), dim=1)
    dists = F.log_softmax(dists, dim=1)
    loss = -dists[:, -1].mean()
    return loss, loss_acts, acts_recon_loss


def train(encoder, acts_encoder, acts_decoder, trans, inverse_trans, optimizer, train_loader, epoch, device):
    encoder.train()
    acts_encoder.train()
    acts_decoder.train()
    trans.train()
    inverse_trans.train()
    
    stats = Stats()
    pbar = tqdm(total=len(train_loader.dataset))

    parameters = list(encoder.parameters()) + list(trans.parameters()) + list(acts_encoder.parameters()) + list(inverse_trans.parameters()) + list(acts_decoder.parameters())
    for batch in train_loader:
        obs, obs_pos, actions = [b.to(device) for b in batch]
        loss_states, loss_acts, loss_acts_recon = get_cloud_loss(obs, obs_pos, encoder, acts_encoder, acts_decoder, trans, inverse_trans, actions, device)
        loss = loss_states + loss_acts + loss_acts_recon
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        stats.add('states_loss', loss_states.item())
        stats.add('acts_loss', loss_acts.item())
        stats.add('acts_recon_loss', loss_acts_recon.item())
        avg_loss = np.mean(stats['train_loss'][-50:])
        avg_states_loss = np.mean(stats['states_loss'][-50:])
        avg_acts_loss = np.mean(stats['acts_loss'][-50:])
        avg_acts_recon_loss = np.mean(stats['acts_recon_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f},'
                             f'States Loss {avg_states_loss:.4f},'
                             f'Acts Loss {avg_acts_loss:.4f},'
                             f'Acts Recon Loss {avg_acts_recon_loss:.4f}')
        pbar.update(obs.shape[0])
    pbar.close()
    return stats


def test(encoder, acts_encoder, acts_decoder, trans, inverse_trans, test_loader, epoch, device):
    encoder.eval()
    acts_encoder.eval()
    acts_decoder.eval()
    trans.eval()
    inverse_trans.eval()

    test_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_pos, actions = [b.to(device) for b in batch]
            loss_states, loss_acts, loss_acts_recon = get_cloud_loss(obs, obs_pos, encoder, acts_encoder, acts_decoder, trans, inverse_trans, actions, device)
            loss = loss_states + loss_acts + loss_acts_recon
            test_loss += loss * obs.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    return test_loss.item()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = "checkpoints"
    if not exists(folder_name):
        os.makedirs(folder_name)

    obs_dim = (3, 64, 64)
    action_dim = 4

    device = torch.device('cuda')

    encoder = cloud.Encoder(args.z_dim, obs_dim[0]).to(device)
    acts_encoder = cloud.ActionEncoder(args.z_dim, action_dim).to(device)
    acts_decoder = cloud.ActionDecoder(args.z_dim, action_dim).to(device)
    trans = cloud.Transition(args.z_dim, action_dim).to(device)
    inverse_trans = cloud.InverseModel(args.z_dim, action_dim).to(device)

    parameters = list(encoder.parameters()) + list(trans.parameters()) + list(acts_encoder.parameters()) + list(inverse_trans.parameters()) + list(acts_decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.load_checkpoint:
        checkpoint = torch.load(join(folder_name, 'checkpoint'))
        encoder.load_state_dict(checkpoint['encoder'])
        acts_encoder.load_state_dict(checkpoint['acts_encoder'])
        acts_decoder.load_state_dict(checkpoint['acts_decoder'])
        trans.load_state_dict(checkpoint['trans'])
        inverse_trans.load_state_dict(checkpoint['inverse_trans'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    dataloader = get_dataloader()

    best_test_loss = float('inf')
    itr = 0
    for epoch in range(args.epochs):
        
        stats = train(encoder, acts_encoder, acts_decoder, trans, inverse_trans, optimizer, dataloader["train"], epoch, device)
        test_loss = test(encoder, acts_encoder, acts_decoder, trans, inverse_trans, dataloader["test"], epoch, device)

        old_itr = itr
        for k, values in stats.items():
            itr = old_itr
            for v in values:
                itr += 1

        if test_loss <= best_test_loss:
            best_test_loss = test_loss

            checkpoint = {
                'encoder': encoder,
                'acts_encoder': acts_encoder,
                'acts_decoder': acts_decoder,
                'trans': trans,
                'inverse_trans': inverse_trans,
                'optimizer': optimizer,
            }
            torch.save(checkpoint, join(folder_name, 'checkpoint'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/sample', help='path to dataset')
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, help='default 0')
    parser.add_argument('--epochs', type=int, default=60, help='default: 50')
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128, help='default 128')
    parser.add_argument('--z_dim', type=int, default=4, help='dimension of the latents')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main()
