import torch.utils.data as data
from os.path import join
import pickle as pkl
import h5py
import torch
import numpy as np

class SampleDataset(data.Dataset):

    def __init__(self, root):
        self.root = root

        with open(join(root, f'pos_neg_pairs.pkl'), 'rb') as f:
            data = pkl.load(f)
        self.pos_pairs = data['pos_pairs']
        self.image_paths = data['all_images']

        self.images = h5py.File(join(root, 'images.hdf5'), 'r')['images'][:]
        self.img2idx = {self.image_paths[i]: i for i in range(len(self.image_paths))}

        self.mean = np.array([0.5, 0.5, 0., 0.])
        self.std = np.array([0.5, 0.5, 1, 1])

    def _get_image(self, path):
        img = self.images[self.img2idx[path]]
        img = img.astype('float32') / 255
        img = (img - 0.5) / 0.5
        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index):
        obs_file, obs_next_file, action_file = self.pos_pairs[index]
        obs, obs_next = self._get_image(obs_file), self._get_image(obs_next_file)
        actions = np.load(action_file)

        fsplit = obs_next_file.split('_')
        t = int(fsplit[-2])
        k = int(fsplit[-1].split('.')[0])

        action = actions[t - 1, k]
        action = (action - self.mean) / self.std

        return obs, obs_next, torch.FloatTensor(action)
