from __future__ import division
import argparse, pdb, os, numpy, imp
from datetime import datetime
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import models, utils

parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='poke', help='reacher | poke | rope')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-model', type=str, default='cloud', help='cloud | cloud_F | cloud_I')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-loss', type=str, default='l2', help='l2')
parser.add_argument('-gpu', type=int, default=1)
parser.add_argument('-datapath', type=str, default='./data/', help='data folder')
parser.add_argument('-save_dir', type=str, default='./results/', help='where to save the models')
parser.add_argument('-z_dim', type=int, default=8, help='dimension of the latents')

opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.set_default_tensor_type('torch.FloatTensor')

if opt.gpu > 0:
    torch.cuda.set_device(opt.gpu)

data_config = utils.read_config('config.json').get(opt.task)
data_config['batchsize'] = opt.batch_size
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])
opt.ncond = data_config['ncond']
opt.npred = data_config['npred']
opt.height = data_config['height']
opt.width = data_config['width']
opt.nc = data_config['nc']
ImageLoader=imp.load_source('ImageLoader', 'dataloaders/{}.py'.format(data_config.get('dataloader'))).ImageLoader
dataloader = ImageLoader(data_config)

opt.save_dir = '{}/{}/'.format(opt.save_dir, opt.task)
opt.model_filename = '{}/model={}-loss={}-lrt={}'.format(
                    opt.save_dir, opt.model, opt.loss, opt.lrt)
print("Saving to " + opt.model_filename)

def get_loss(obs, obs_target, encoder, acts_encoder, acts_decoder, trans, inverse_trans, actions, device):
    bs = obs.shape[0]
    actshape = actions.shape[0]

    z, z_target = encoder(obs), encoder(obs_target)

    actions_unsqueeze = torch.unsqueeze(actions, 2)
    qacts = acts_encoder(actions_unsqueeze)
    qacts_decode = acts_decoder(qacts)
    qacts_decode = torch.squeeze(qacts_decode, 2)
    acts_recon_loss = torch.nn.functional.mse_loss(qacts_decode, actions)
    
    # F(ht, d(e(at)))
    noise_factor = 0.2
    device = torch.device('cuda')
    z_noise = torch.from_numpy(noise_factor * numpy.random.normal(loc=0.0, scale=1.0, size=z.shape)).float().to(device)
    qacts_decode_noise = torch.from_numpy(noise_factor * numpy.random.normal(loc=0.0, scale=1.0, size=qacts_decode.shape)).float().to(device)
    z_next = trans(z, qacts_decode, z_noise)
    z_target_noise = torch.from_numpy(noise_factor * numpy.random.normal(loc=0.0, scale=1.0, size=z_target.shape)).float().to(device)

    # z: ht z_target: ht+1
    qacts_inv = inverse_trans(z, z_target)
    loss_acts = torch.nn.functional.mse_loss(qacts_inv, actions)

    # Maximize agreement between z_next
    neg_dot_products = torch.mm(z_next, z.t())
    neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2* neg_dot_products + (z ** 2).sum(1).unsqueeze(0))
    idxs = numpy.arange(bs)

    neg_dists[idxs, idxs] = float('-inf')

    pos_dot_products = (z_target * z_next).sum(dim=1)
    pos_dists = -((z_target ** 2).sum(1) - 2* pos_dot_products + (z_next ** 2).sum(1))
    pos_dists = pos_dists.unsqueeze(1)

    dists = torch.cat((neg_dists, pos_dists), dim=1)
    dists = torch.nn.functional.log_softmax(dists, dim=1)
    loss = -dists[:, -1].mean()
    
    total_loss = loss + loss_acts + acts_recon_loss
    return total_loss

def train_epoch(nsteps, encoder, acts_encoder, acts_decoder, trans, inverse_trans, optimizer):
    total_loss = 0
    encoder.train()
    acts_encoder.train()
    acts_decoder.train()
    trans.train()
    inverse_trans.train()

    parameters = list(encoder.parameters()) + list(trans.parameters()) + list(acts_encoder.parameters()) + list(inverse_trans.parameters()) + list(acts_decoder.parameters())
    for iter in range(0, nsteps):
        obs, obs_target, actions = dataloader.get_batch('train')
        loss_states, loss_acts, loss_acts_recon = get_loss(obs, obs_target, encoder, acts_encoder, acts_decoder, trans, inverse_trans, actions, device)
        loss = loss_states + loss_acts + loss_acts_recon
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()
    return total_loss / nsteps


def test_epoch(nsteps, encoder, acts_encoder, acts_decoder, trans, inverse_trans):
    total_loss = 0
    encoder.eval()
    acts_encoder.eval()
    acts_decoder.eval()
    trans.eval()
    inverse_trans.eval()
    
    for iter in range(0, nsteps):
        with torch.no_grad():
            obs, obs_target, action = dataloader.get_batch('valid')
            loss_states, loss_acts, loss_acts_recon = get_loss(obs, obs_target, encoder, acts_encoder, acts_decoder, trans, inverse_trans, action, device)
            loss = loss_states + loss_acts + loss_acts_recon
            total_loss += loss * obs.shape[0]
    return total_loss / nsteps

def train(n_epochs, encoder, acts_encoder, acts_decoder, trans, inverse_trans, optimizer):
    os.system("mkdir -p " + opt.save_dir)
    best_valid_loss = 1e7
    train_loss, valid_loss = [], []
    for i in range(0, n_epochs):        
        train_loss.append(train_epoch(opt.epoch_size, encoder, acts_encoder, acts_decoder, trans, inverse_trans, optimizer))
        valid_loss.append(test_epoch(opt.epoch_size, encoder, acts_encoder, acts_decoder, trans, inverse_trans))

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            torch.save({ 'epoch': i, 'encoder': encoder, 'acts_encoder': acts_encoder, 'acts_decoder': acts_decoder, 'trans': trans, 'inverse_trans': inverse_trans, 'train_loss': train_loss, 'valid_loss': valid_loss}, opt.model_filename + '.model')
            torch.save(optimizer, opt.model_filename + '.optim')

        log_string = ('iter: {:d}, train_loss: {:0.6f}, valid_loss: {:0.6f}, best_valid_loss: {:0.6f}, lr: {:0.5f}').format(
                      (i+1)*opt.epoch_size, train_loss[-1], valid_loss[-1], best_valid_loss, opt.lrt)
        print(log_string)
        utils.log(opt.model_filename + '.log', log_string)


if __name__ == '__main__':
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    obs_dim = (3, 64, 64)
    action_dim = 5

    device = torch.device('cuda')
    encoder = models.Encoder(opt.z_dim, obs_dim[0]).to(device)
    acts_encoder = models.ActionEncoder(opt.z_dim, action_dim).to(device)
    acts_decoder = models.ActionDecoder(opt.z_dim, action_dim).to(device)
    trans = models.TransitionNoise(opt.z_dim, action_dim, 4).to(device)
    inverse_trans = models.InverseModel(opt.z_dim, action_dim).to(device)

    parameters = list(encoder.parameters()) + list(trans.parameters()) + list(acts_encoder.parameters()) + list(inverse_trans.parameters()) + list(acts_decoder.parameters())
    optimizer = optim.Adam(parameters, lr=opt.lrt, weight_decay=1e-5)

    print('training...')
    utils.log(opt.model_filename + '.log', '[training]')
    train(500, encoder, acts_encoder, acts_decoder, trans, inverse_trans, optimizer)
