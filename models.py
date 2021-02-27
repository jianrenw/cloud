import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
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


class ActionDecoder(nn.Module):
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


class TransitionNoise(nn.Module):
    def __init__(self, z_dim, action_dim, noise_dim):
        super().__init__()
        self.model = TransitionParamNoise(z_dim, action_dim, noise_dim, hidden_sizes=[64, 64])

    def forward(self, z, a, noise):
        return self.model(z, a, noise)


class TransitionParamNoise(nn.Module):
    def __init__(self, z_dim, action_dim=0, noise_dim=0, hidden_sizes=[]):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.model = MLP(z_dim + action_dim + noise_dim*2, z_dim * z_dim, hidden_sizes=hidden_sizes)

    def forward(self, z, a, noise):
        x = torch.cat((z, a, noise), dim=-1)
        Ws = torch.tanh(self.model(x)).view(x.shape[0], self.z_dim, self.z_dim) / math.sqrt(self.z_dim)
        return torch.bmm(Ws, z.unsqueeze(-1)).squeeze(-1) 

class InverseModel(nn.Module):
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

class InverseModelNoise(nn.Module):
    def __init__(self, z_dim, action_dim, noise_dim):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim

        self.model = nn.Sequential(
            nn.Linear(3 * z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z, z_next, noise):
        x = torch.cat((z, z_next, noise), dim=1)
        return self.model(x)


class GaussianNoise(nn.Module): def __init__(self, sigma=0.1, is_relative_detach=True):
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


class MLP(nn.Module):   def __init__(self, input_size, output_size, hidden_sizes=[]):
        super().__init__()
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
