import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    prefix = 'encoder'

    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        self.out = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x


class ActionEncoder(nn.Module):
    prefix = 'encoder'

    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv1d(channel_dim, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(64, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(128, 256, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(256, 256, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        self.out = nn.Linear(256, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x


class Transition(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim):
        super().__init__()
        self.model = TransitionSimple(z_dim, action_dim)

    def forward(self, z, a):
        return self.model(z, a)


class TransitionSimple(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        x = self.model(x)
        return x


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class TransitionParam(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, hidden_sizes=[], orthogonalize_mode='reparam_w'):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.orthogonalize_mode = orthogonalize_mode

        if orthogonalize_mode == 'reparam_w_ortho_cont':
            self.model = MLP(z_dim + action_dim, z_dim * (z_dim - 1), hidden_sizes=hidden_sizes)
        else:
            self.model = MLP(z_dim + action_dim, z_dim * z_dim, hidden_sizes=hidden_sizes)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim)
        return torch.bmm(Ws, z.unsqueeze(-1)).squeeze(-1) 


class InverseTransition(nn.Module):
    prefix = 'invtrans'

    def __init__(self, z_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(2 * z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z, z_next):
        x = torch.cat((z, z_next), dim=1)
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[]):
        super().__init__()
        print(">>> int size : {} out : {}".format(input_size, output_size))
        model = []
        prev_h = input_size
        for h in hidden_sizes + [output_size]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop() 
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def quantize(x, n_bit):
    x = x * 0.5 + 0.5 
    x *= n_bit ** 2 - 1 
    x = torch.floor(x + 1e-4) 
    return x


class Decoder(nn.Module):
    prefix = 'decoder'

    def __init__(self, z_dim, channel_dim, discrete=False, n_bit=4):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim
        self.discrete = discrete
        self.n_bit = n_bit
        self.discrete_dim = 2 ** n_bit

        out_dim = self.discrete_dim * self.channel_dim if discrete else channel_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 256, 4, 1),
            
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, out_dim, 4, 2, 1),
        )

    def forward(self, z):
        x = z.view(-1, self.z_dim, 1, 1)
        output = self.main(x)

        if self.discrete:
            output = output.view(output.shape[0], self.discrete_dim,
                                 self.channel_dim, *output.shape[2:])
        else:
            output = torch.tanh(output)

        return output


    def loss(self, x, z):
        recon = self(z)
        if self.discrete:
            loss = F.cross_entropy(recon, quantize(x, self.n_bit).long())
        else:
            loss = F.mse_loss(recon, x)
        return loss


    def predict(self, z):
        recon = self(z)
        if self.discrete:
            recon = torch.max(recon, dim=1)[1].float()
            recon = (recon / (self.discrete_dim - 1) - 0.5) / 0.5
        return recon

class ActionDecoder(nn.Module):
    prefix = 'actiondecoder'

    def __init__(self, z_dim, channel_dim, discrete=False, n_bit=4):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim
        self.discrete = discrete
        self.n_bit = n_bit
        self.discrete_dim = 2 ** n_bit

        out_dim = self.discrete_dim * self.channel_dim if discrete else channel_dim
        self.main = nn.Sequential(
            nn.ConvTranspose1d(self.z_dim, 256, 1),
            
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(256, 256, 1),
            
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(256, 128, 1),
            
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(128, 64, 1),
            
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(64, out_dim, 1),
        )

    def forward(self, z):
        x = z.view(-1, self.z_dim, 1)
        output = self.main(x)

        if self.discrete:
            output = output.view(output.shape[0], self.discrete_dim,
                                 self.channel_dim, *output.shape[2:])
        else:
            output = torch.tanh(output)

        return output


    def loss(self, x, z):
        recon = self(z)
        if self.discrete:
            loss = F.cross_entropy(recon, quantize(x, self.n_bit).long())
        else:
            loss = F.mse_loss(recon, x)
        return loss


    def predict(self, z):
        recon = self(z)
        if self.discrete:
            recon = torch.max(recon, dim=1)[1].float()
            recon = (recon / (self.discrete_dim - 1) - 0.5) / 0.5
        return recon


class InverseModel(nn.Module):
    prefix = 'inv'

    def __init__(self, z_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(2 * z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z, z_next):
        
        x = torch.cat((z, z_next), dim=-1)
        return self.model(x)


class ForwardModel(nn.Module):
    prefix = 'forward'

    def __init__(self, z_dim, action_dim, mode='linear'):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        if mode == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        else:
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, z_dim),
            )

    def forward(self, z, action):
        x = torch.cat((z, action), dim=1)
        return self.model(x)
